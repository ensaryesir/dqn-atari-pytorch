# =============================================================================
# main.py
# -----------------------------------------------------------------------------
# DQN eğitim döngüsü, değerlendirme, checkpoint kaydı ve grafik üretimi.
# argparse ile tüm hiperparametreler komut satırından özelleştirilebilir.
# Tek komutla çalıştırılabilir:
#   python main.py --total_steps 2000000 --device cuda
#   python main.py --test_run   (hızlı pipeline doğrulama, ~1000 adım)
# =============================================================================

import os
import sys
import time
import random
import argparse
import numpy as np
import torch

# Yerel modüller
from wrappers        import make_atari_env
from dqn_agent       import DQNAgent
from q_learning_agent import train_tabular_q
from plot_results    import generate_all_plots

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
DEFAULT_ENV          = "BreakoutNoFrameskip-v4"
DEFAULT_TOTAL_STEPS  = 2_000_000
DEFAULT_EVAL_FREQ    = 10_000
DEFAULT_EVAL_EPS     = 5
DEFAULT_SAVE_DIR     = "./results"
DEFAULT_SEED         = 42
CHECKPOINT_FREQ      = 100_000   # Kaç adımda bir model kaydedilir
TEST_RUN_STEPS       = 1_000     # --test_run için adım sayısı
LOG_PRINT_FREQ       = 10_000    # Konsolda kaç adımda bir yazdırılır


# =============================================================================
# Argparse
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    """Komut satırı argümanlarını tanımlar."""
    parser = argparse.ArgumentParser(
        description="DQN Atari Eğitim Döngüsü",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env",          type=str,   default=DEFAULT_ENV)
    parser.add_argument("--total_steps",  type=int,   default=DEFAULT_TOTAL_STEPS)
    parser.add_argument("--eval_freq",    type=int,   default=DEFAULT_EVAL_FREQ)
    parser.add_argument("--eval_eps",     type=int,   default=DEFAULT_EVAL_EPS)
    parser.add_argument("--save_dir",     type=str,   default=DEFAULT_SAVE_DIR)
    parser.add_argument("--device",       type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",         type=int,   default=DEFAULT_SEED)
    parser.add_argument("--test_run",     action="store_true",
                        help="Hızlı test modu: 1000 adım, grafik üretir, çıkar.")
    # Hiperparametre geçersiz kılma
    parser.add_argument("--alpha",              type=float, default=None)
    parser.add_argument("--gamma",              type=float, default=None)
    parser.add_argument("--batch_size",         type=int,   default=None)
    parser.add_argument("--buffer_size",        type=int,   default=None)
    parser.add_argument("--target_update_freq", type=int,   default=None)
    parser.add_argument("--epsilon_start",      type=float, default=None)
    parser.add_argument("--epsilon_end",        type=float, default=None)
    parser.add_argument("--epsilon_decay",      type=int,   default=None)
    parser.add_argument("--min_replay_size",    type=int,   default=None)
    return parser


# =============================================================================
# Seed
# =============================================================================

def set_global_seed(seed: int) -> None:
    """numpy, random ve torch için global rastgele tohumu ayarlar."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# =============================================================================
# Değerlendirme
# =============================================================================

def evaluate(
    agent: DQNAgent,
    env_id: str,
    n_episodes: int,
    seed: int,
    device: torch.device,
) -> float:
    """Ajanı açgözlü politikayla (ε=0.05) n_episodes bölüm boyunca değerlendirir.

    Parameters
    ----------
    agent : DQNAgent
        Değerlendirilecek ajan.
    env_id : str
        Ortam kimliği.
    n_episodes : int
        Değerlendirme bölümü sayısı.
    seed : int
        Başlangıç tohumu.
    device : torch.device
        Hesaplama cihazı.

    Returns
    -------
    float
        n_episodes bölümü boyunca ortalama toplam ödül.
    """
    try:
        eval_env = make_atari_env(env_id, seed=seed + 9999)
    except Exception as exc:
        print(f"[HATA] Değerlendirme ortamı oluşturulamadı: {exc}")
        return 0.0

    # Değerlendirme sırasında epsilon'u geçici olarak küçük tut
    original_epsilon = agent.epsilon
    agent.epsilon    = 0.05

    total_rewards = []
    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            ep_reward += reward
        total_rewards.append(ep_reward)

    eval_env.close()
    agent.epsilon = original_epsilon
    return float(np.mean(total_rewards))


# =============================================================================
# Kayıt yardımcıları
# =============================================================================

def setup_directories(save_dir: str) -> None:
    """Gerekli dizinleri oluşturur."""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)


def save_checkpoint(agent: DQNAgent, save_dir: str, step: int) -> None:
    """Online ağ ağırlıklarını kaydeder.

    Parameters
    ----------
    agent : DQNAgent
        Kaydedilecek ajan.
    save_dir : str
        Kayıt dizini.
    step : int
        Mevcut adım sayısı.
    """
    ckpt_dir  = os.path.join(save_dir, "checkpoints")
    ckpt_path = os.path.join(ckpt_dir, f"step_{step:07d}.pt")
    try:
        torch.save(agent.online_net.state_dict(), ckpt_path)
        print(f"  [Checkpoint] Kaydedildi: {ckpt_path}")
    except Exception as exc:
        print(f"  [UYARI] Checkpoint kaydı başarısız: {exc}")


def log_to_file(
    log_path: str,
    step: int,
    epsilon: float,
    score: float,
    avg_loss: float,
) -> None:
    """training_log.txt dosyasına bir satır yazar.

    Parameters
    ----------
    log_path : str
        Log dosyasının tam yolu.
    step : int
        Mevcut adım.
    epsilon : float
        Güncel epsilon.
    score : float
        Değerlendirme skoru.
    avg_loss : float
        Ortalama Huber kaybı.
    """
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{step}\t{epsilon:.4f}\t{score:.4f}\t{avg_loss:.6f}\n")
    except Exception as exc:
        print(f"  [UYARI] Log yazma hatası: {exc}")


# =============================================================================
# Ana eğitim döngüsü
# =============================================================================

def train(args: argparse.Namespace) -> None:
    """DQN eğitim döngüsünü çalıştırır.

    Parameters
    ----------
    args : argparse.Namespace
        Komut satırı argümanları.
    """
    set_global_seed(args.seed)
    setup_directories(args.save_dir)

    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"  DQN Atari Eğitimi")
    print(f"  Ortam      : {args.env}")
    print(f"  Cihaz      : {device}")
    print(f"  Toplam adım: {args.total_steps:,}")
    print(f"  Seed       : {args.seed}")
    print(f"{'='*60}\n")

    # ---- Ortam ----
    try:
        env = make_atari_env(args.env, seed=args.seed)
    except Exception as exc:
        print(f"[HATA] Ortam oluşturulamadı: {exc}")
        sys.exit(1)

    n_actions   = env.action_space.n
    state_shape = env.observation_space.shape

    # ---- Ajan ----
    agent_kwargs = {}
    if args.alpha             is not None: agent_kwargs["alpha"]             = args.alpha
    if args.gamma             is not None: agent_kwargs["gamma"]             = args.gamma
    if args.batch_size        is not None: agent_kwargs["batch_size"]        = args.batch_size
    if args.buffer_size       is not None: agent_kwargs["buffer_size"]       = args.buffer_size
    if args.target_update_freq is not None: agent_kwargs["target_update_freq"] = args.target_update_freq
    if args.epsilon_start     is not None: agent_kwargs["epsilon_start"]     = args.epsilon_start
    if args.epsilon_end       is not None: agent_kwargs["epsilon_end"]       = args.epsilon_end
    if args.epsilon_decay     is not None: agent_kwargs["epsilon_decay_steps"] = args.epsilon_decay
    if args.min_replay_size   is not None: agent_kwargs["min_replay_size"]   = args.min_replay_size

    agent = DQNAgent(state_shape=state_shape, n_actions=n_actions, device=device, **agent_kwargs)

    # ---- Log dosyası ----
    log_path = os.path.join(args.save_dir, "training_log.txt")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("step\tepsilon\tscore\tavg_loss\n")
    except Exception as exc:
        print(f"[UYARI] Log dosyası açılamadı: {exc}")

    # ---- Kayıt listeleri ----
    score_records: list = []   # [(step, avg_score)]
    loss_records: list  = []   # [(step, loss)]
    step_losses: list   = []   # Bir eval penceresi içindeki kayıplar

    # ---- Eğitim döngüsü ----
    obs, _ = env.reset(seed=args.seed)
    total_steps = args.total_steps
    start_time  = time.time()

    print(f"Eğitim başlıyor... (toplam {total_steps:,} adım)\n")

    step = 0
    while step < total_steps:
        # 1. Eylem seç
        action = agent.select_action(obs)

        # 2. Ortamda adım at
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 3. Geçişi depola
        agent.store_transition(obs, action, reward, next_obs, done)

        # 4. Güncelle
        loss = agent.update()
        if loss is not None:
            step_losses.append(loss)
            loss_records.append((step, loss))

        # 5. Bölüm bittiyse yeniden başlat
        if done:
            obs, _ = env.reset(seed=args.seed + step)
        else:
            obs = next_obs

        # 6. Değerlendirme
        if (step + 1) % args.eval_freq == 0:
            avg_score = evaluate(
                agent, args.env, args.eval_eps, args.seed, device
            )
            avg_loss  = float(np.mean(step_losses)) if step_losses else 0.0
            step_losses = []

            score_records.append((step + 1, avg_score))

            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / (elapsed + 1e-8)
            eta_sec = (total_steps - step - 1) / (steps_per_sec + 1e-8)

            print(
                f"Adım {step+1:7,}/{total_steps:,}  "
                f"ε={agent.get_epsilon():.3f}  "
                f"Skor={avg_score:.2f}  "
                f"Kayıp={avg_loss:.4f}  "
                f"Hız={steps_per_sec:.0f}adım/s  "
                f"ETA={eta_sec/60:.1f}dk"
            )
            log_to_file(log_path, step + 1, agent.get_epsilon(), avg_score, avg_loss)

        # 7. Checkpoint
        if (step + 1) % CHECKPOINT_FREQ == 0:
            save_checkpoint(agent, args.save_dir, step + 1)

        step += 1

    env.close()
    print("\nEğitim tamamlandı!")

    # ---- Numpy kayıtları ----
    _save_numpy_records(score_records, loss_records, args.save_dir)

    # ---- Q-Öğrenme karşılaştırması ----
    print("\nQ-Öğrenme karşılaştırma eğitimi başlıyor...")
    q_rewards, _ = train_tabular_q(
        env_id="CartPole-v1",
        episodes=500,
        seed=args.seed,
    )
    q_save_path = os.path.join(args.save_dir, "q_scores.npy")
    try:
        np.save(q_save_path, np.array(q_rewards, dtype=np.float32))
        print(f"Q-Öğrenme skorları kaydedildi: {q_save_path}")
    except Exception as exc:
        print(f"[UYARI] Q-Öğrenme skorları kaydedilemedi: {exc}")

    # ---- Grafik üretimi ----
    print("\nGrafikler üretiliyor...")
    generate_all_plots(
        results_dir=args.save_dir,
        figures_dir=os.path.join(args.save_dir, "figures"),
    )

    print("\nTüm işlemler tamamlandı ✓")
    print(f"Sonuçlar: {os.path.abspath(args.save_dir)}")


# =============================================================================
# Hızlı test modu (--test_run)
# =============================================================================

def test_run(args: argparse.Namespace) -> None:
    """1000 adımlık hızlı pipeline doğrulama modu.

    Ortamın kurulumunu, ajan oluşturmayı, replay tamponunu, güncellemeyi
    ve grafik üretimini birkaç dakikada doğrular.

    Parameters
    ----------
    args : argparse.Namespace
        Komut satırı argümanları.
    """
    print("\n[TEST MODU] Hızlı pipeline doğrulaması başlıyor...\n")

    set_global_seed(args.seed)
    setup_directories(args.save_dir)
    device = torch.device(args.device)

    # Ortam
    try:
        env = make_atari_env(args.env, seed=args.seed)
        print(f"  Ortam kurulumu    : OK ({args.env})")
    except Exception as exc:
        print(f"  [HATA] Ortam kurulamadı: {exc}")
        sys.exit(1)

    n_actions   = env.action_space.n
    state_shape = env.observation_space.shape

    # Ajan (küçük tampon ve min doluluk)
    agent = DQNAgent(
        state_shape      = state_shape,
        n_actions        = n_actions,
        device           = device,
        buffer_size      = 2_000,
        min_replay_size  = 500,
    )
    print(f"  Ajan oluşturma    : OK (n_actions={n_actions})")

    obs, _ = env.reset(seed=args.seed)
    score_records = []
    loss_records  = []

    for step in range(TEST_RUN_STEPS):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.store_transition(obs, action, reward, next_obs, done)
        loss = agent.update()
        if loss is not None:
            loss_records.append((step, loss))
        if done:
            obs, _ = env.reset(seed=args.seed)
        else:
            obs = next_obs

        if (step + 1) % 200 == 0:
            score_records.append((step + 1, float(np.random.uniform(0, 5))))
            print(f"  Adım {step+1}/{TEST_RUN_STEPS} — ε={agent.get_epsilon():.3f}")

    env.close()
    print(f"  Eğitim döngüsü    : OK ({TEST_RUN_STEPS} adım)")

    # Numpy kayıtları
    _save_numpy_records(score_records, loss_records, args.save_dir)
    print("  Numpy kaydı       : OK")

    # Q-Öğrenme (kısa)
    q_rewards, _ = train_tabular_q(env_id="CartPole-v1", episodes=20, seed=args.seed)
    np.save(os.path.join(args.save_dir, "q_scores.npy"), np.array(q_rewards, dtype=np.float32))
    print("  Q-Öğrenme         : OK")

    # Grafikler
    generate_all_plots(
        results_dir=args.save_dir,
        figures_dir=os.path.join(args.save_dir, "figures"),
    )
    print("  Grafik üretimi    : OK")

    print("\n[TEST MODU] Pipeline doğrulama başarıyla tamamlandı ✓")


# =============================================================================
# Yardımcı: numpy kaydetme
# =============================================================================

def _save_numpy_records(
    score_records: list,
    loss_records: list,
    save_dir: str,
) -> None:
    """Score ve loss kayıtlarını numpy dosyaları olarak kaydeder."""
    if score_records:
        scores_arr = np.array(score_records, dtype=np.float32)
        try:
            np.save(os.path.join(save_dir, "dqn_scores.npy"), scores_arr)
            print(f"  dqn_scores.npy kaydedildi ({len(scores_arr)} nokta)")
        except Exception as exc:
            print(f"  [UYARI] dqn_scores.npy kaydı başarısız: {exc}")

    if loss_records:
        losses_arr = np.array(loss_records, dtype=np.float32)
        try:
            np.save(os.path.join(save_dir, "dqn_losses.npy"), losses_arr)
            print(f"  dqn_losses.npy kaydedildi ({len(losses_arr)} nokta)")
        except Exception as exc:
            print(f"  [UYARI] dqn_losses.npy kaydı başarısız: {exc}")


# =============================================================================
# Giriş noktası
# =============================================================================

if __name__ == "__main__":
    parser = build_arg_parser()
    args   = parser.parse_args()

    if args.test_run:
        test_run(args)
    else:
        train(args)
