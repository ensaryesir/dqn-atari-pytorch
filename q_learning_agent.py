# =============================================================================
# q_learning_agent.py
# -----------------------------------------------------------------------------
# Karşılaştırma amaçlı geleneksel tabular Q-Öğrenme ajanı.
# Atari'nin ham piksel uzayı tabular yöntemle işlenemeyeceğinden CartPole-v1
# ortamında çalışır. Durum gözlemleri ayrıklaştırılarak Q tablosunda saklanır.
# DQN ile kıyaslamalı performans grafiklerinin üretiminde kullanılır.
# =============================================================================

import numpy as np
import gymnasium as gym
from collections import defaultdict
from typing import List, Tuple, Optional

# ---------------------------------------------------------------------------
# Hiperparametreler
# ---------------------------------------------------------------------------
Q_ALPHA           = 0.1     # Öğrenme oranı
Q_GAMMA           = 0.99    # İndirim çarpanı
Q_EPSILON_START   = 1.0     # Başlangıç keşif oranı
Q_EPSILON_END     = 0.01    # Minimum keşif oranı
Q_EPSILON_DECAY_EP= 500     # Kaç bölümde epsilon azalır
Q_TRAIN_EPISODES  = 500     # Toplam eğitim bölümü sayısı

# CartPole gözlem ayrıklaştırma bin sayısı
CART_BINS = 6


class TabularQAgent:
    """Epsilon-açgözlü politikayla çalışan tabular Q-Öğrenme ajanı.

    CartPole-v1 ortamında çalışır; sürekli gözlemler sabit sayıda
    bölmeye (bin) eşlenerek ayrık durum uzayı oluşturulur.

    Parameters
    ----------
    n_actions : int
        Eylem uzayının boyutu.
    obs_low : np.ndarray
        Gözlem uzayının alt sınırları.
    obs_high : np.ndarray
        Gözlem uzayının üst sınırları.
    n_bins : int
        Her boyut için ayrıklaştırma bölme sayısı.
    alpha : float
        Öğrenme oranı.
    gamma : float
        İndirim çarpanı.
    epsilon_start : float
        Başlangıç keşif oranı.
    epsilon_end : float
        Minimum keşif oranı.
    epsilon_decay_episodes : int
        Epsilon'un minimum değere ulaşacağı bölüm sayısı.

    Attributes
    ----------
    q_table : defaultdict
        Durum-eylem çiftlerine karşılık gelen Q-değerleri.
    episode_count : int
        Tamamlanan bölüm sayısı.
    epsilon : float
        Güncel keşif oranı.
    """

    def __init__(
        self,
        n_actions: int,
        obs_low: np.ndarray,
        obs_high: np.ndarray,
        n_bins: int               = CART_BINS,
        alpha: float              = Q_ALPHA,
        gamma: float              = Q_GAMMA,
        epsilon_start: float      = Q_EPSILON_START,
        epsilon_end: float        = Q_EPSILON_END,
        epsilon_decay_episodes: int = Q_EPSILON_DECAY_EP,
    ) -> None:

        self.n_actions   = n_actions
        self.alpha       = alpha
        self.gamma       = gamma
        self.epsilon     = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_episodes
        self.n_bins      = n_bins
        self.episode_count = 0

        # Sonsuz üst/alt sınırları kırp (CartPole için gerekli)
        clipped_low  = np.clip(obs_low,  -10.0, 10.0)
        clipped_high = np.clip(obs_high, -10.0, 10.0)

        # Her boyut için bölme kenarları
        self.bins = [
            np.linspace(clipped_low[i], clipped_high[i], n_bins + 1)[1:-1]
            for i in range(len(obs_low))
        ]

        # Q tablosu: state_key → np.zeros(n_actions)
        self.q_table: defaultdict = defaultdict(
            lambda: np.zeros(n_actions, dtype=np.float64)
        )

    # ------------------------------------------------------------------
    # Durum ayrıklaştırma
    # ------------------------------------------------------------------

    def _discretize(self, obs: np.ndarray) -> Tuple:
        """Sürekli gözlemi ayrık bölme indekslerine dönüştürür.

        Parameters
        ----------
        obs : np.ndarray
            Sürekli gözlem vektörü.

        Returns
        -------
        tuple
            Her boyut için bölme indeksinden oluşan demet (hashable).
        """
        return tuple(
            int(np.digitize(obs[i], self.bins[i]))
            for i in range(len(obs))
        )

    # ------------------------------------------------------------------
    # Eylem seçimi
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray) -> int:
        """Epsilon-açgözlü politikayla eylem seçer.

        Parameters
        ----------
        obs : np.ndarray
            Mevcut sürekli gözlem.

        Returns
        -------
        int
            Seçilen eylem indeksi.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        state_key = self._discretize(obs)
        return int(np.argmax(self.q_table[state_key]))

    # ------------------------------------------------------------------
    # Q tablosu güncelleme
    # ------------------------------------------------------------------

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Bellman denklemine göre Q tablosunu günceller.

        Q(s,a) ← Q(s,a) + α·[r + γ·max_{a'}Q(s',a') - Q(s,a)]

        Parameters
        ----------
        obs : np.ndarray
            Mevcut durum gözlemi.
        action : int
            Uygulanan eylem.
        reward : float
            Alınan ödül.
        next_obs : np.ndarray
            Sonraki durum gözlemi.
        done : bool
            Bölüm bitti mi?
        """
        s  = self._discretize(obs)
        s_ = self._discretize(next_obs)

        q_current = self.q_table[s][action]
        q_next    = 0.0 if done else np.max(self.q_table[s_])
        td_target = reward + self.gamma * q_next
        td_error  = td_target - q_current

        self.q_table[s][action] += self.alpha * td_error

    # ------------------------------------------------------------------
    # Epsilon güncelleme
    # ------------------------------------------------------------------

    def decay_epsilon(self) -> None:
        """Bölüm sonunda epsilon'u doğrusal olarak azaltır."""
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        self.episode_count += 1

    def get_epsilon(self) -> float:
        """Güncel epsilon değerini döndürür."""
        return self.epsilon


# =============================================================================
# Eğitim fonksiyonu
# =============================================================================

def train_tabular_q(
    env_id: str   = "CartPole-v1",
    episodes: int = Q_TRAIN_EPISODES,
    seed: int     = 42,
    n_bins: int   = CART_BINS,
) -> Tuple[List[float], TabularQAgent]:
    """Tabular Q-Öğrenme ajanını CartPole-v1'de eğitir.

    Parameters
    ----------
    env_id : str
        Gymnasium ortam kimliği.
    episodes : int
        Toplam eğitim bölümü sayısı.
    seed : int
        Rastgele tohum.
    n_bins : int
        Ayrıklaştırma bölme sayısı.

    Returns
    -------
    episode_rewards : list of float
        Her bölüme ait toplam ödül.
    agent : TabularQAgent
        Eğitilmiş ajan nesnesi.
    """
    np.random.seed(seed)

    try:
        env = gym.make(env_id)
    except Exception as exc:
        raise RuntimeError(f"Ortam oluşturulamadı: {env_id!r} — {exc}") from exc

    obs_space = env.observation_space
    agent = TabularQAgent(
        n_actions   = env.action_space.n,
        obs_low     = obs_space.low,
        obs_high    = obs_space.high,
        n_bins      = n_bins,
        epsilon_decay_episodes=episodes,
    )

    episode_rewards: List[float] = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward

        agent.decay_epsilon()
        episode_rewards.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(
                f"  [Q-Öğrenme] Bölüm {ep+1:4d}/{episodes}  "
                f"ε={agent.get_epsilon():.3f}  "
                f"Ort. ödül (son 100): {avg:.1f}"
            )

    env.close()
    return episode_rewards, agent


# =============================================================================
# Modül testi
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("TabularQAgent modül testi (CartPole-v1, 50 bölüm)")
    print("=" * 50)

    rewards, agent = train_tabular_q(episodes=50, seed=42)
    print(f"İlk 10 bölüm ödülleri : {rewards[:10]}")
    print(f"Son  10 bölüm ödülleri: {rewards[-10:]}")
    print(f"Ortalama ödül         : {np.mean(rewards):.2f}")
    print(f"Q tablosu büyüklüğü   : {len(agent.q_table)} durum")
    print("Test BAŞARILI ✓")
