# =============================================================================
# replay_buffer.py
# -----------------------------------------------------------------------------
# DQN Deneyim Yeniden Oynatma tamponu (Experience Replay Buffer).
# Geçişleri (state, action, reward, next_state, done) dörtlüsü olarak saklar.
# State'ler uint8 numpy array olarak tutularak bellek verimliliği sağlanır.
# Rastgele mini-batch örneklemesi ile bağımsız ve özdeş dağılım (i.i.d.) garantisi.
# =============================================================================

import random
import numpy as np
from collections import deque
from typing import Tuple, List

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
DEFAULT_BUFFER_SIZE = 1_000_000
DEFAULT_BATCH_SIZE  = 32


class ReplayBuffer:
    """Sabit kapasiteli deneyim yeniden oynatma tamponu.

    Geçişleri (s, a, r, s', done) biçiminde saklar.
    State ve next_state'ler bellek tasarrufu için uint8 numpy dizisi olarak
    depolanır; örnekleme sırasında float32'ye dönüştürme çağıranın sorumluluğundadır.

    Parameters
    ----------
    capacity : int
        Tamponun tutabileceği maksimum geçiş sayısı.
    """

    def __init__(self, capacity: int = DEFAULT_BUFFER_SIZE) -> None:
        self.buffer: deque = deque(maxlen=capacity)
        self.capacity = capacity

    # ------------------------------------------------------------------
    # Temel işlemler
    # ------------------------------------------------------------------

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Tampona yeni bir geçiş ekler.

        Parameters
        ----------
        state : np.ndarray
            Mevcut durum (uint8, shape [4, 84, 84]).
        action : int
            Seçilen eylem indeksi.
        reward : float
            Alınan ödül (zaten kırpılmış olabilir).
        next_state : np.ndarray
            Sonraki durum (uint8, shape [4, 84, 84]).
        done : bool
            Bölümün bitip bitmediği.
        """
        # uint8 olarak sakla — float32'ye kıyasla ~4x bellek tasarrufu
        state_u8      = np.array(state,      dtype=np.uint8)
        next_state_u8 = np.array(next_state, dtype=np.uint8)
        self.buffer.append((state_u8, action, reward, next_state_u8, done))

    def sample(self, batch_size: int = DEFAULT_BATCH_SIZE) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Tampondan rastgele bir mini-batch örnekler.

        Parameters
        ----------
        batch_size : int
            Örneklenecek geçiş sayısı.

        Returns
        -------
        states : np.ndarray,  float32, shape (B, 4, 84, 84)
        actions : np.ndarray, int64,   shape (B,)
        rewards : np.ndarray, float32, shape (B,)
        next_states : np.ndarray, float32, shape (B, 4, 84, 84)
        dones : np.ndarray, float32, shape (B,)

        Raises
        ------
        ValueError
            Tampon boyutu batch_size'dan küçükse.
        """
        if len(self) < batch_size:
            raise ValueError(
                f"Tampon boyutu ({len(self)}) < batch_size ({batch_size})"
            )

        batch: List = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Normalize etme burada yapılmıyor; ajan içinde yapılır
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        """Tamponda mevcut geçiş sayısını döndürür."""
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        """Tamponun örnekleme için hazır olup olmadığını kontrol eder."""
        return len(self) >= min_size


# =============================================================================
# Modül testi
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("ReplayBuffer modül testi")
    print("=" * 50)

    buf = ReplayBuffer(capacity=100)
    assert len(buf) == 0

    # Sahte geçişler ekle
    for i in range(50):
        s  = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
        ns = np.random.randint(0, 256, (4, 84, 84), dtype=np.uint8)
        buf.push(s, action=i % 4, reward=float(i), next_state=ns, done=(i % 10 == 0))

    print(f"Tampon boyutu : {len(buf)} (beklenen: 50)")
    assert len(buf) == 50

    states, actions, rewards, next_states, dones = buf.sample(32)
    print(f"States şekli  : {states.shape}")
    print(f"Actions şekli : {actions.shape}")
    assert states.shape      == (32, 4, 84, 84)
    assert actions.shape     == (32,)
    assert rewards.shape     == (32,)
    assert next_states.shape == (32, 4, 84, 84)
    assert dones.shape       == (32,)
    print("Test BAŞARILI ✓")
