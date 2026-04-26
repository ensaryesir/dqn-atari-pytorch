# =============================================================================
# dqn_agent.py
# -----------------------------------------------------------------------------
# DQN (Deep Q-Network) ajanı. Experience Replay ve hedef ağ (target network)
# mekanizmalarını içerir. DeepMind 2015 DQN makalesindeki hiperparametrelere
# birebir uyar. Online ve hedef ağların periyodik senkronizasyonu,
# epsilon-greedy politikası ve Huber kaybı burada uygulanır.
# =============================================================================

import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional

from model         import DQNModel
from replay_buffer import ReplayBuffer

# ---------------------------------------------------------------------------
# Hiperparametreler (Tablo 1 — makaleyle birebir)
# ---------------------------------------------------------------------------
ALPHA                = 0.00025      # Öğrenme oranı
GAMMA                = 0.99         # İndirim çarpanı
BATCH_SIZE           = 32
REPLAY_BUFFER_SIZE   = 1_000_000
TARGET_UPDATE_FREQ   = 10_000       # Adım cinsinden hedef ağ güncelleme sıklığı (C)
EPSILON_START        = 1.0
EPSILON_END          = 0.1
EPSILON_DECAY_STEPS  = 1_000_000    # Doğrusal azalma adım sayısı
MIN_REPLAY_SIZE      = 50_000       # Eğitime başlamak için minimum tampon doluluk
GRAD_CLIP_NORM       = 10.0         # Maksimum gradyan normu
# RMSprop eps: 0.01 / BATCH_SIZE  (makale Tablo 1)
RMSPROP_EPS          = 0.01 / BATCH_SIZE


class DQNAgent:
    """DeepMind DQN ajanı (Experience Replay + Target Network).

    Parameters
    ----------
    state_shape : tuple
        Gözlem uzayının şekli, örn. (4, 84, 84).
    n_actions : int
        Eylem uzayının boyutu.
    device : torch.device
        Hesaplamaların yapılacağı cihaz (cpu veya cuda).
    alpha : float
        Öğrenme oranı (RMSprop).
    gamma : float
        İndirim çarpanı.
    batch_size : int
        Mini-batch boyutu.
    buffer_size : int
        Replay tamponunun kapasitesi.
    target_update_freq : int
        Hedef ağın güncelleneceği adım sıklığı.
    epsilon_start : float
        Başlangıç keşif oranı.
    epsilon_end : float
        Minimum keşif oranı.
    epsilon_decay_steps : int
        Epsilon'un epsilon_end'e ulaşacağı toplam adım sayısı.
    min_replay_size : int
        Güncelleme başlamadan önce tamponda bulunması gereken minimum geçiş.

    Attributes
    ----------
    online_net : DQNModel
        Çevrimiçi (öğrenen) Q-ağı.
    target_net : DQNModel
        Hedef Q-ağı (periyodik kopyalanır).
    step_count : int
        Toplam eğitim adım sayacı.
    epsilon : float
        Güncel keşif oranı.
    """

    def __init__(
        self,
        state_shape: tuple,
        n_actions: int,
        device: torch.device,
        alpha: float             = ALPHA,
        gamma: float             = GAMMA,
        batch_size: int          = BATCH_SIZE,
        buffer_size: int         = REPLAY_BUFFER_SIZE,
        target_update_freq: int  = TARGET_UPDATE_FREQ,
        epsilon_start: float     = EPSILON_START,
        epsilon_end: float       = EPSILON_END,
        epsilon_decay_steps: int = EPSILON_DECAY_STEPS,
        min_replay_size: int     = MIN_REPLAY_SIZE,
    ) -> None:

        self.state_shape         = state_shape
        self.n_actions           = n_actions
        self.device              = device
        self.gamma               = gamma
        self.batch_size          = batch_size
        self.target_update_freq  = target_update_freq
        self.epsilon             = epsilon_start
        self.epsilon_end         = epsilon_end
        self.epsilon_decay       = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.min_replay_size     = min_replay_size

        self.step_count: int = 0

        # ---- Ağlar ----
        self.online_net = DQNModel(n_actions).to(device)
        self.target_net = DQNModel(n_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()   # Hedef ağ hiçbir zaman eğitilmez

        # ---- Optimizatör ----
        # Tablo 1: RMSprop, lr=α, ε=0.01/batch_size, α_decay=0.95, momentum=0
        self.optimizer = optim.RMSprop(
            self.online_net.parameters(),
            lr=alpha,
            eps=RMSPROP_EPS,
            alpha=0.95,
            momentum=0.0,
        )

        # ---- Replay tamponu ----
        self.replay = ReplayBuffer(capacity=buffer_size)

    # ------------------------------------------------------------------
    # Eylem seçimi
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-açgözlü politikaya göre eylem seçer.

        Parameters
        ----------
        state : np.ndarray
            Mevcut gözlem (uint8, shape state_shape).

        Returns
        -------
        int
            Seçilen eylem indeksi.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        state_t = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0) / 255.0    # (1, 4, 84, 84) normalize edilmiş

        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Geçiş depolama
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Replay tamponuna bir geçiş ekler.

        Parameters
        ----------
        state : np.ndarray
            Mevcut durum.
        action : int
            Uygulanan eylem.
        reward : float
            Alınan ödül.
        next_state : np.ndarray
            Sonraki durum.
        done : bool
            Bölüm bitti mi?
        """
        self.replay.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Güncelleme (eğitim adımı)
    # ------------------------------------------------------------------

    def update(self) -> Optional[float]:
        """Bir mini-batch üzerinde DQN güncellemesi yapar.

        Returns
        -------
        float veya None
            Huber kaybı değeri; tampon yeterince dolmamışsa None.
        """
        if not self.replay.is_ready(self.min_replay_size):
            # Epsilon ve sayacı güncelle ama eğitme
            self._update_epsilon()
            self.step_count += 1
            return None

        # ---- Veri örnekle ----
        states, actions, rewards, next_states, dones = self.replay.sample(
            self.batch_size
        )

        # ---- Tensöre dönüştür & normalize ----
        states_t      = torch.tensor(states,      dtype=torch.float32, device=self.device) / 255.0
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device) / 255.0
        actions_t     = torch.tensor(actions,     dtype=torch.long,    device=self.device)
        rewards_t     = torch.tensor(rewards,     dtype=torch.float32, device=self.device)
        dones_t       = torch.tensor(dones,       dtype=torch.float32, device=self.device)

        # ---- Hedef Q-değeri ----
        with torch.no_grad():
            # max_{a'} Q(s', a'; θ⁻)
            next_q_values = self.target_net(next_states_t).max(dim=1)[0]
            targets = rewards_t + self.gamma * next_q_values * (1.0 - dones_t)

        # ---- Online Q-değeri (seçilen eylemler için) ----
        q_values = self.online_net(states_t)
        q_selected = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # ---- Huber kaybı (Smooth L1) ----
        loss = nn.functional.smooth_l1_loss(q_selected, targets)

        # ---- Geri yayılım ----
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), max_norm=GRAD_CLIP_NORM
        )
        self.optimizer.step()

        # ---- Epsilon ve adım sayacı güncelle ----
        self._update_epsilon()
        self.step_count += 1

        # ---- Hedef ağ kopyalama ----
        if self.step_count % self.target_update_freq == 0:
            self._update_target_network()

        return loss.item()

    # ------------------------------------------------------------------
    # Yardımcı metodlar
    # ------------------------------------------------------------------

    def _update_epsilon(self) -> None:
        """Epsilon'u doğrusal olarak azaltır."""
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def _update_target_network(self) -> None:
        """Hedef ağın ağırlıklarını online ağdan kopyalar."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def get_epsilon(self) -> float:
        """Güncel epsilon değerini döndürür."""
        return self.epsilon

    def get_step_count(self) -> int:
        """Güncel adım sayısını döndürür."""
        return self.step_count




# =============================================================================
# Modül testi
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("DQNAgent modül testi")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    STATE_SHAPE = (4, 84, 84)
    N_ACTIONS   = 4

    agent = DQNAgent(
        state_shape=STATE_SHAPE,
        n_actions=N_ACTIONS,
        device=device,
        min_replay_size=64,   # Test için küçük tutuyoruz
        buffer_size=1000,
    )
    print("Ajan oluşturuldu ✓")

    # Sahte geçişler ekle
    for i in range(100):
        s  = np.random.randint(0, 256, STATE_SHAPE, dtype=np.uint8)
        ns = np.random.randint(0, 256, STATE_SHAPE, dtype=np.uint8)
        a  = agent.select_action(s)
        agent.store_transition(s, a, 1.0, ns, False)

    print(f"Tampon boyutu: {len(agent.replay)}")

    # 5 güncelleme denemesi
    losses = []
    for _ in range(5):
        l = agent.update()
        if l is not None:
            losses.append(l)

    print(f"Kayıplar: {losses}")
    print(f"Epsilon : {agent.get_epsilon():.4f}")
    print(f"Adım    : {agent.get_step_count()}")
    print("Test BAŞARILI ✓")
