# =============================================================================
# model.py
# -----------------------------------------------------------------------------
# DQN için evrişimli sinir ağı (CNN) mimarisini tanımlar.
# Girdi: (batch, 4, 84, 84) boyutunda 4 ardışık gri tonlamalı Atari karesi.
# Çıktı: Her eylem için Q-değeri içeren (batch, n_actions) tensörü.
# Ağırlıklar Xavier uniform ile başlatılır.
# =============================================================================

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
CONV1_FILTERS = 32
CONV1_KERNEL  = 8
CONV1_STRIDE  = 4

CONV2_FILTERS = 64
CONV2_KERNEL  = 4
CONV2_STRIDE  = 2

CONV3_FILTERS = 64
CONV3_KERNEL  = 3
CONV3_STRIDE  = 1

FC1_UNITS     = 512


class DQNModel(nn.Module):
    """Atari oyunları için DeepMind DQN CNN mimarisi.

    Parameters
    ----------
    n_actions : int
        Ortamın eylem uzayının boyutu (çıkış nöronu sayısı).

    Attributes
    ----------
    conv_layers : nn.Sequential
        3 evrişimli katmandan oluşan özellik çıkarıcı blok.
    fc_layers : nn.Sequential
        2 tam bağlantılı katmandan oluşan Q-değeri tahmincisi.
    """

    def __init__(self, n_actions: int) -> None:
        super(DQNModel, self).__init__()

        self.n_actions = n_actions

        # Evrişimli blok: (batch, 4, 84, 84) → (batch, 64, 7, 7)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, CONV1_FILTERS, kernel_size=CONV1_KERNEL, stride=CONV1_STRIDE),
            nn.ReLU(),
            nn.Conv2d(CONV1_FILTERS, CONV2_FILTERS, kernel_size=CONV2_KERNEL, stride=CONV2_STRIDE),
            nn.ReLU(),
            nn.Conv2d(CONV2_FILTERS, CONV3_FILTERS, kernel_size=CONV3_KERNEL, stride=CONV3_STRIDE),
            nn.ReLU(),
        )

        # Tam bağlantılı boyutunu dinamik hesapla
        conv_out_size = self._get_conv_output_size()

        # Tam bağlantılı blok
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, FC1_UNITS),
            nn.ReLU(),
            nn.Linear(FC1_UNITS, n_actions),
        )

        # Ağırlık başlatma
        self._initialize_weights()

    # ------------------------------------------------------------------
    # Yardımcı metodlar
    # ------------------------------------------------------------------

    def _get_conv_output_size(self) -> int:
        """Evrişim bloğunun çıkış boyutunu dummy ileri geçişle hesaplar."""
        dummy = torch.zeros(1, 4, 84, 84)
        out = self.conv_layers(dummy)
        return int(out.numel())

    def _initialize_weights(self) -> None:
        """Tüm Conv ve Linear katmanlara Xavier uniform başlatma uygular."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # İleri geçiş
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ağın ileri geçişini gerçekleştirir.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, 4, 84, 84), [0, 1] aralığında normalize edilmiş.

        Returns
        -------
        torch.Tensor
            Shape (batch, n_actions), her eylem için Q-değerleri.
        """
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)  # Flatten
        return self.fc_layers(features)


# =============================================================================
# Modül testi
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("DQNModel modül testi")
    print("=" * 50)

    n_actions = 4
    model = DQNModel(n_actions=n_actions)
    print(f"Model oluşturuldu: n_actions={n_actions}")

    dummy_input = torch.zeros(8, 4, 84, 84)
    output = model(dummy_input)
    print(f"Girdi şekli : {dummy_input.shape}")
    print(f"Çıktı şekli : {output.shape}")
    assert output.shape == (8, n_actions), "Çıktı boyutu hatalı!"
    print("Test BAŞARILI ✓")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Toplam parametre sayısı: {total_params:,}")
