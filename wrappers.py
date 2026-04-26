# =============================================================================
# wrappers.py
# -----------------------------------------------------------------------------
# Atari ortamları için ön işleme sarmalayıcıları (wrappers).
# DeepMind DQN makalesindeki standart ön işleme adımlarını uygular:
#   NoopReset, MaxAndSkip, EpisodicLife, FireReset, WarpFrame,
#   ClipReward ve FrameStack.
# gymnasium ile tam uyumlu gymnasium.Wrapper alt sınıfları kullanılır.
# =============================================================================

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import Optional, Tuple, Any
import cv2

# gymnasium>=1.0 ile ALE ortamlarını kaydet (Atari ROM'larını aktif et)
try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass  # ALE kurulu değilse wrapper modülü yine de import edilebilsin

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
NOOP_ACTION       = 0
FIRE_ACTION       = 1
DEFAULT_NOOP_MAX  = 30
DEFAULT_SKIP      = 4
DEFAULT_FRAME_W   = 84
DEFAULT_FRAME_H   = 84
DEFAULT_FRAME_K   = 4     # Frame stack derinliği


# =============================================================================
# 1. NoopResetEnv
# =============================================================================

class NoopResetEnv(gym.Wrapper):
    """Bölüm başında rastgele sayıda NOOP eylemi gerçekleştirir.

    Başlangıç durumunu çeşitlendirerek ajan aşırı uyum (overfitting)
    yapmadan genelleşmeyi öğrenir.

    Parameters
    ----------
    env : gym.Env
        Sarmalanacak ortam.
    noop_max : int
        Maksimum NOOP sayısı (1 ile noop_max arasında rastgele seçilir).
    """

    def __init__(self, env: gym.Env, noop_max: int = DEFAULT_NOOP_MAX) -> None:
        super().__init__(env)
        self.noop_max    = noop_max
        self.noop_action = NOOP_ACTION
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


# =============================================================================
# 2. MaxAndSkipEnv
# =============================================================================

class MaxAndSkipEnv(gym.Wrapper):
    """Her skip karede bir eylem uygular; son 2 kareyi max-pool eder.

    Atari ALE'nin titreşim (flickering) sorununu çözmek için son iki kare
    piksel bazında maksimum alınır.

    Parameters
    ----------
    env : gym.Env
        Sarmalanacak ortam.
    skip : int
        Aynı eylemin uygulanacağı kare sayısı.
    """

    def __init__(self, env: gym.Env, skip: int = DEFAULT_SKIP) -> None:
        super().__init__(env)
        self._skip        = skip
        self._obs_buffer  = deque(maxlen=2)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if terminated or truncated:
                break
        # Son 2 karenin piksel maksimumu
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.clear()
        self._obs_buffer.append(obs)
        return obs, info


# =============================================================================
# 3. EpisodicLifeEnv
# =============================================================================

class EpisodicLifeEnv(gym.Wrapper):
    """Hayat kaybını bölüm sonu (done) olarak işaretler.

    Gerçek bölüm sonu yalnızca tüm hayatlar bittiğinde tetiklenir;
    bu sayede ajan her hayatı ayrı bir bölüm olarak öğrenir.

    Parameters
    ----------
    env : gym.Env
        Sarmalanacak ortam.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives        = 0
        self.was_real_done = True

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # Hayat kaybı → bu adım için done = True
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # Hayat kaybından sonra ortamı sıfırla ama gerçekten yeniden başlatma
            obs, _, _, _, info = self.env.step(NOOP_ACTION)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


# =============================================================================
# 4. FireResetEnv
# =============================================================================

class FireResetEnv(gym.Wrapper):
    """Sıfırlama sonrasında FIRE eylemiyle oyunu başlatır.

    Breakout gibi oyunlarda FIRE eylemi olmadan oyun başlamaz.

    Parameters
    ----------
    env : gym.Env
        Sarmalanacak ortam.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[FIRE_ACTION] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(FIRE_ACTION)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)  # RIGHT
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


# =============================================================================
# 5. WarpFrame
# =============================================================================

class WarpFrame(gym.ObservationWrapper):
    """Gözlemi gri tonlamaya çevirir ve 84×84'e yeniden boyutlandırır.

    Parameters
    ----------
    env : gym.Env
        Sarmalanacak ortam.
    width : int
        Hedef genişlik (piksel).
    height : int
        Hedef yükseklik (piksel).
    """

    def __init__(
        self,
        env: gym.Env,
        width: int  = DEFAULT_FRAME_W,
        height: int = DEFAULT_FRAME_H,
    ) -> None:
        super().__init__(env)
        self._width  = width
        self._height = height
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(height, width, 1),
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame,
            (self._width, self._height),
            interpolation=cv2.INTER_AREA,
        )
        return frame[:, :, np.newaxis]   # (H, W, 1)


# =============================================================================
# 6. ClipRewardEnv
# =============================================================================

class ClipRewardEnv(gym.RewardWrapper):
    """Ödülü {-1, 0, +1} kümesine kırpar (sign fonksiyonu).

    Farklı oyunların farklı ödül ölçeklerini normalize ederek
    karşılaştırılabilir öğrenme sinyali sağlar.

    Parameters
    ----------
    env : gym.Env
        Sarmalanacak ortam.
    """

    def reward(self, reward: float) -> float:
        return np.sign(reward)


# =============================================================================
# 7. FrameStack
# =============================================================================

class FrameStack(gym.Wrapper):
    """Son k kareyi kanallara yığarak (4, 84, 84) gözlem üretir.

    Parameters
    ----------
    env : gym.Env
        Sarmalanacak ortam.
    k : int
        Yığılacak kare sayısı.
    """

    def __init__(self, env: gym.Env, k: int = DEFAULT_FRAME_K) -> None:
        super().__init__(env)
        self._k      = k
        self._frames = deque(maxlen=k)

        low  = np.repeat(env.observation_space.low,  k, axis=-1)
        high = np.repeat(env.observation_space.high, k, axis=-1)

        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(k, env.observation_space.shape[0], env.observation_space.shape[1]),
            dtype=np.uint8,
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Kare yığınını (k, H, W) şeklinde döndürür."""
        frames = np.concatenate(list(self._frames), axis=-1)  # (H, W, k)
        return frames.transpose(2, 0, 1)                       # (k, H, W)


# =============================================================================
# Ana sarmalayıcı fabrikası
# =============================================================================

def make_atari_env(
    env_id: str,
    seed: int = 42,
    noop_max: int  = DEFAULT_NOOP_MAX,
    skip: int      = DEFAULT_SKIP,
    frame_w: int   = DEFAULT_FRAME_W,
    frame_h: int   = DEFAULT_FRAME_H,
    frame_k: int   = DEFAULT_FRAME_K,
) -> gym.Env:
    """Standart Atari ön işleme zincirini uygulayarak ortam oluşturur.

    Uygulama sırası (DeepMind DQN makalesi):
      NoopReset → MaxAndSkip → EpisodicLife → FireReset →
      WarpFrame → ClipReward → FrameStack

    Parameters
    ----------
    env_id : str
        Gymnasium ortam kimliği (örn. "BreakoutNoFrameskip-v4").
    seed : int
        Tekrarlanabilirlik için rastgele tohum.
    noop_max : int
        Maksimum başlangıç NOOP sayısı.
    skip : int
        Eylem tekrar sayısı.
    frame_w, frame_h : int
        Hedef çerçeve boyutları.
    frame_k : int
        Yığılacak çerçeve sayısı.

    Returns
    -------
    gym.Env
        Sarmalanmış Atari ortamı.
    """
    try:
        env = gym.make(env_id, render_mode=None)
    except Exception as exc:
        raise RuntimeError(f"Ortam oluşturulamadı: {env_id!r} — {exc}") from exc

    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=skip)
    env = EpisodicLifeEnv(env)

    # Yalnızca FIRE eylemi olan oyunlarda FireResetEnv uygula
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env = WarpFrame(env, width=frame_w, height=frame_h)
    env = ClipRewardEnv(env)
    env = FrameStack(env, k=frame_k)

    return env


# =============================================================================
# Modül testi
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("wrappers.py modül testi")
    print("=" * 50)

    ENV_ID = "BreakoutNoFrameskip-v4"
    try:
        env = make_atari_env(ENV_ID, seed=42)
        obs, info = env.reset()
        print(f"Ortam           : {ENV_ID}")
        print(f"Gözlem şekli    : {obs.shape}  (beklenen: (4, 84, 84))")
        print(f"Eylem uzayı     : {env.action_space.n}")
        assert obs.shape == (4, 84, 84), f"Beklenmeyen şekil: {obs.shape}"

        obs2, reward, term, trunc, info = env.step(env.action_space.sample())
        print(f"Adım sonrası şekil  : {obs2.shape}")
        print(f"Ödül (kırpılmış)    : {reward}")
        env.close()
        print("Test BAŞARILI ✓")
    except Exception as e:
        print(f"Test başarısız: {e}")
        print("Atari ROM'ları kurulmamış olabilir. Şu komutu çalıştırın:")
        print("  pip install gymnasium[atari] ale-py")
        print("  python -m ale_py.roms")
