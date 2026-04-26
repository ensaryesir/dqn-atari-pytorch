README.md
# DQN Atari — PyTorch

Deep Q-Network ile Atari 2600 oyunlarını oynayan bir pekiştirmeli öğrenme ajanı.

## Kurulum

```bash
pip install -r requirements.txt
# Atari ROM'larını yükle
pip install "gymnasium[atari,accept-rom-license]"
python -m ale_py.roms   # ROM'ları doğrula
```

## Kullanım

### Hızlı test (pipeline doğrulama)
```bash
python main.py --test_run
```

### Tam eğitim (GPU)
```bash
python main.py --total_steps 2000000 --device cuda
```

### Tam eğitim (CPU)
```bash
python main.py --total_steps 2000000 --device cpu
```

### Tek modülü test etme
```bash
python model.py
python replay_buffer.py
python wrappers.py
python dqn_agent.py
python q_learning_agent.py
python plot_results.py
```

## Proje Yapısı

| Dosya | Açıklama |
|---|---|
| `main.py` | Eğitim döngüsü, argparse, kayıt |
| `model.py` | CNN mimarisi (DeepMind DQN) |
| `dqn_agent.py` | DQN Ajanı (Experience Replay + Target Network) |
| `replay_buffer.py` | Deneyim Yeniden Oynatma Tamponu |
| `wrappers.py` | Atari ön işleme wrapper'ları |
| `q_learning_agent.py` | Tabular Q-Öğrenme karşılaştırması |
| `plot_results.py` | Grafik üretimi (3 şekil) |

## Hiperparametreler (DeepMind DQN 2015)

| Parametre | Değer |
|---|---|
| Öğrenme oranı (α) | 0.00025 |
| İndirim (γ) | 0.99 |
| Batch boyutu | 32 |
| Replay boyutu | 1,000,000 |
| Target güncelleme (C) | 10,000 adım |
| ε başlangıç → bitiş | 1.0 → 0.1 |
| ε azalma adımı | 1,000,000 |
| Min replay doluluk | 50,000 |

## Çıktılar

```
results/
├── dqn_scores.npy          # (N, 2) adım & skor
├── dqn_losses.npy          # (M, 2) adım & kayıp
├── q_scores.npy            # (500,) Q-Öğrenme bölüm skorları
├── training_log.txt        # adım | ε | skor | kayıp
├── checkpoints/            # Her 100k adımda .pt dosyaları
└── figures/
    ├── figure1_score_comparison.png / .pdf
    ├── figure2_loss_curve.png / .pdf
    └── figure3_epsilon_decay.png / .pdf
```
