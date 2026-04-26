# =============================================================================
# plot_results.py
# -----------------------------------------------------------------------------
# Eğitim sonuçlarını görselleştiren grafik üretim modülü.
# Üç ayrı grafik üretir:
#   Grafik 1 — DQN ve Q-Öğrenme karşılaştırmalı bölüm skoru eğrisi
#   Grafik 2 — DQN eğitim kayıp eğrisi (Huber)
#   Grafik 3 — Epsilon (ε) keşif oranı azalma eğrisi
# Her grafik hem .png (300 dpi) hem .pdf olarak kaydedilir.
# =============================================================================

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # GUI gerektirmeyen arka uç
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Optional

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
DPI              = 300
FIGURE_SIZE      = (10, 5)
SMOOTH_WIN_SCORE = 50    # Bölüm skoru için hareketli ortalama penceresi
SMOOTH_WIN_LOSS  = 100   # Kayıp eğrisi için hareketli ortalama penceresi

# Renk paleti
COLOR_DQN      = "#2196F3"   # Mavi
COLOR_Q_LEARN  = "#F44336"   # Kırmızı
COLOR_FILL     = "#90CAF9"   # Açık mavi (güven bantları için)
COLOR_LOSS     = "#7C4DFF"   # Mor
COLOR_EPSILON  = "#FF6F00"   # Turuncu

# Font
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
    "legend.fontsize"  : 10,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})


# =============================================================================
# Yardımcı fonksiyon: hareketli ortalama
# =============================================================================

def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Basit hareketli ortalama hesaplar.

    Parameters
    ----------
    data : np.ndarray
        Ham veri dizisi.
    window : int
        Pencerenin boyutu.

    Returns
    -------
    np.ndarray
        Pürüzsüzleştirilmiş dizi (aynı uzunlukta).
    """
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    # 'same' modu için sınır etkilerini azaltmak adına 'valid' kullanılır ve pad edilir
    smoothed = np.convolve(data, kernel, mode="valid")
    # Başa ham verilerle doldur
    pad = np.full(window - 1, data[:window - 1].mean())
    return np.concatenate([pad, smoothed])


def _save_figure(fig: plt.Figure, path_no_ext: str) -> None:
    """Figürü hem PNG hem PDF olarak kaydeder.

    Parameters
    ----------
    fig : plt.Figure
        Kaydedilecek figür.
    path_no_ext : str
        Uzantısız dosya yolu.
    """
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    for ext in (".png", ".pdf"):
        out_path = path_no_ext + ext
        fig.savefig(out_path, dpi=DPI if ext == ".png" else None, bbox_inches="tight")
        print(f"  Kaydedildi : {out_path}")


# =============================================================================
# Grafik 1 — Bölüm Başına Ortalama Skor (DQN vs Q-Öğrenme)
# =============================================================================

def plot_score_comparison(
    dqn_steps: np.ndarray,
    dqn_scores: np.ndarray,
    q_episodes: np.ndarray,
    q_scores: np.ndarray,
    save_dir: str = "./results/figures",
    smooth_window: int = SMOOTH_WIN_SCORE,
) -> None:
    """DQN ve Tabular Q-Öğrenme bölüm skorlarını karşılaştırmalı çizer.

    Parameters
    ----------
    dqn_steps : np.ndarray
        DQN değerlendirme adımları (X ekseni).
    dqn_scores : np.ndarray
        DQN ortalama bölüm skorları.
    q_episodes : np.ndarray
        Q-Öğrenme bölüm numaraları.
    q_scores : np.ndarray
        Q-Öğrenme bölüm skorları.
    save_dir : str
        Grafiğin kaydedileceği dizin.
    smooth_window : int
        Hareketli ortalama pencere boyutu.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # DQN eğrisi
    dqn_smooth = moving_average(dqn_scores, smooth_window)
    dqn_x_norm = np.linspace(0, 1, len(dqn_steps))

    ax.plot(dqn_x_norm, dqn_scores, alpha=0.2, color=COLOR_DQN, linewidth=0.8)
    ax.plot(dqn_x_norm, dqn_smooth,
            color=COLOR_DQN, linewidth=2.0, label="DQN (Önerilen)")

    # Q-Öğrenme eğrisi (normalize edilmiş X)
    q_smooth = moving_average(q_scores, smooth_window)
    q_x_norm = np.linspace(0, 1, len(q_episodes))

    ax.plot(q_x_norm, q_scores, alpha=0.2, color=COLOR_Q_LEARN, linewidth=0.8)
    ax.plot(q_x_norm, q_smooth,
            color=COLOR_Q_LEARN, linewidth=2.0, linestyle="--",
            label="Q-Öğrenme (Temel)")

    ax.set_xlabel("Normalize Edilmiş Eğitim İlerlemesi")
    ax.set_ylabel("Ortalama Bölüm Skoru")
    ax.set_title("Şekil 1. Eğitim Boyunca Ortalama Bölüm Skoru")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    plt.tight_layout()
    _save_figure(fig, os.path.join(save_dir, "figure1_score_comparison"))
    plt.close(fig)


# =============================================================================
# Grafik 2 — DQN Eğitim Kayıp Eğrisi
# =============================================================================

def plot_loss_curve(
    steps: np.ndarray,
    losses: np.ndarray,
    save_dir: str = "./results/figures",
    smooth_window: int = SMOOTH_WIN_LOSS,
) -> None:
    """DQN eğitim sırasında hesaplanan Huber kayıp eğrisini çizer.

    Parameters
    ----------
    steps : np.ndarray
        Güncelleme adımları.
    losses : np.ndarray
        Adım başına Huber kaybı değerleri.
    save_dir : str
        Grafiğin kaydedileceği dizin.
    smooth_window : int
        Hareketli ortalama pencere boyutu.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    smoothed = moving_average(losses, smooth_window)
    ax.plot(steps, losses,   alpha=0.15, color=COLOR_LOSS, linewidth=0.5)
    ax.plot(steps, smoothed, color=COLOR_LOSS, linewidth=2.0)

    ax.set_xlabel("Güncelleme Adımı")
    ax.set_ylabel("Huber Kaybı")
    ax.set_title("Şekil 2. DQN Eğitim Kayıp Eğrisi")
    ax.grid(True, linestyle=":", alpha=0.5)

    # Logaritmik Y ekseni (kayıp geniş aralıkta değişebilir)
    if losses.max() / (losses.min() + 1e-8) > 100:
        ax.set_yscale("log")
        ax.set_ylabel("Huber Kaybı (log)")

    plt.tight_layout()
    _save_figure(fig, os.path.join(save_dir, "figure2_loss_curve"))
    plt.close(fig)


# =============================================================================
# Grafik 3 — Epsilon Azalma Eğrisi
# =============================================================================

def plot_epsilon_decay(
    steps: np.ndarray,
    epsilons: np.ndarray,
    save_dir: str = "./results/figures",
) -> None:
    """Eğitim boyunca epsilon keşif oranının azalışını çizer.

    Parameters
    ----------
    steps : np.ndarray
        Adım sayıları.
    epsilons : np.ndarray
        Her adıma karşılık gelen epsilon değerleri.
    save_dir : str
        Grafiğin kaydedileceği dizin.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.plot(steps, epsilons, color=COLOR_EPSILON, linewidth=2.0)
    ax.fill_between(steps, epsilons, alpha=0.15, color=COLOR_EPSILON)

    ax.set_xlabel("Adım")
    ax.set_ylabel("ε (Epsilon) Değeri")
    ax.set_title("Şekil 3. ε-Açgözlü Politika: Keşif Oranı Azalması")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    _save_figure(fig, os.path.join(save_dir, "figure3_epsilon_decay"))
    plt.close(fig)


# =============================================================================
# Ana grafik üretim fonksiyonu (main.py tarafından çağrılır)
# =============================================================================

def generate_all_plots(
    results_dir: str = "./results",
    figures_dir: str = "./results/figures",
) -> None:
    """Kaydedilmiş numpy dizilerini okuyarak tüm grafikleri üretir.

    Parameters
    ----------
    results_dir : str
        DQN eğitim çıktılarının bulunduğu dizin.
    figures_dir : str
        Grafiklerin kaydedileceği dizin.
    """
    os.makedirs(figures_dir, exist_ok=True)

    # ---- DQN verileri ----
    dqn_scores_path  = os.path.join(results_dir, "dqn_scores.npy")
    dqn_losses_path  = os.path.join(results_dir, "dqn_losses.npy")
    q_scores_path    = os.path.join(results_dir, "q_scores.npy")

    try:
        dqn_data = np.load(dqn_scores_path)   # shape (N, 2): [step, score]
        dqn_steps   = dqn_data[:, 0]
        dqn_scores  = dqn_data[:, 1]
    except FileNotFoundError:
        print(f"[UYARI] DQN skor dosyası bulunamadı: {dqn_scores_path}")
        return

    try:
        loss_data = np.load(dqn_losses_path)   # shape (M, 2): [step, loss]
        loss_steps  = loss_data[:, 0]
        loss_values = loss_data[:, 1]
    except FileNotFoundError:
        print(f"[UYARI] Kayıp dosyası bulunamadı: {dqn_losses_path}")
        loss_steps  = np.array([0, 1])
        loss_values = np.array([0.0, 0.0])

    # ---- Q-Öğrenme verileri ----
    try:
        q_scores = np.load(q_scores_path)
        q_episodes = np.arange(len(q_scores))
    except FileNotFoundError:
        print(f"[UYARI] Q-Öğrenme skor dosyası bulunamadı: {q_scores_path}")
        q_scores   = np.zeros_like(dqn_scores)
        q_episodes = np.arange(len(q_scores))

    # ---- Epsilon eğrisi (deterministik, hesaplanabilir) ----
    from dqn_agent import EPSILON_START, EPSILON_END, EPSILON_DECAY_STEPS
    eps_steps   = np.arange(0, int(dqn_steps[-1]) + 1, 10_000)
    eps_values  = np.maximum(
        EPSILON_END,
        EPSILON_START - (EPSILON_START - EPSILON_END) / EPSILON_DECAY_STEPS * eps_steps,
    )

    print("\nGrafikler üretiliyor...")
    plot_score_comparison(dqn_steps, dqn_scores, q_episodes, q_scores, save_dir=figures_dir)
    plot_loss_curve(loss_steps, loss_values, save_dir=figures_dir)
    plot_epsilon_decay(eps_steps, eps_values, save_dir=figures_dir)
    print("Tüm grafikler başarıyla kaydedildi ✓")


# =============================================================================
# Modül testi
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("plot_results.py modül testi (sentetik veri)")
    print("=" * 50)

    N_STEPS = 200
    FIGURES = "./results/figures"
    os.makedirs(FIGURES, exist_ok=True)

    # --- Sentetik DQN verileri ---
    dqn_steps  = np.linspace(0, 2_000_000, N_STEPS)
    dqn_scores = np.cumsum(np.random.randn(N_STEPS) * 0.5 + 0.05)
    dqn_scores = np.clip(dqn_scores, 0, 10)

    # --- Sentetik kayıp ---
    loss_steps  = np.linspace(0, 2_000_000, N_STEPS * 10)
    loss_values = np.abs(np.random.randn(N_STEPS * 10) * 0.3 + 0.5 * np.exp(-loss_steps / 5e5))

    # --- Sentetik Q-Öğrenme verileri ---
    q_episodes = np.arange(500)
    q_scores   = np.cumsum(np.random.randn(500) * 1.5 + 0.1)
    q_scores   = np.clip(q_scores, -5, 200)

    # --- Epsilon ---
    from dqn_agent import EPSILON_START, EPSILON_END, EPSILON_DECAY_STEPS
    eps_steps  = np.arange(0, 2_000_001, 10_000)
    eps_values = np.maximum(
        EPSILON_END,
        EPSILON_START - (EPSILON_START - EPSILON_END) / EPSILON_DECAY_STEPS * eps_steps,
    )

    print("Grafik 1 üretiliyor...")
    plot_score_comparison(dqn_steps, dqn_scores, q_episodes, q_scores, save_dir=FIGURES)
    print("Grafik 2 üretiliyor...")
    plot_loss_curve(loss_steps, loss_values, save_dir=FIGURES)
    print("Grafik 3 üretiliyor...")
    plot_epsilon_decay(eps_steps, eps_values, save_dir=FIGURES)
    print("\nTest BAŞARILI ✓")
    print(f"Grafikler '{FIGURES}' dizinine kaydedildi.")
