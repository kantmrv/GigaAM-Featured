import torch
import torchaudio
from torch import Tensor, nn


class SpecScaler(nn.Module):
    """
    Module that applies logarithmic scaling to spectrogram values.
    This module clamps the input values within a certain range and then applies a natural logarithm.
    """

    def forward(self, x: Tensor) -> Tensor:
        return torch.log(x.clamp_(1e-9, 1e9))


class FeatureExtractor(nn.Module):
    """
    Module for extracting Log-mel spectrogram features from raw audio signals.
    This module uses Torchaudio's MelSpectrogram transform to extract features
    and applies logarithmic scaling.
    """

    def __init__(
        self,
        sample_rate: int,
        features: int,
        hop_length: int | None = None,
        win_length: int | None = None,
        n_fft: int | None = None,
        center: bool = True,
        **_kwargs: object,
    ):
        super().__init__()
        self.hop_length = hop_length if hop_length is not None else sample_rate // 100
        self.win_length = win_length if win_length is not None else sample_rate // 40
        self.n_fft = n_fft if n_fft is not None else sample_rate // 40
        self.center = center
        self.featurizer = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_mels=features,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_fft=self.n_fft,
                center=self.center,
            ),
            SpecScaler(),
        )

    def out_len(self, input_lengths: Tensor) -> Tensor:
        """
        Calculates the output length after the feature extraction process.
        """
        if self.center:
            return input_lengths.div(self.hop_length, rounding_mode="floor").add(1).long()
        else:
            return (input_lengths - self.win_length).div(self.hop_length, rounding_mode="floor").add(1).long()

    def forward(self, input_signal: Tensor, length: Tensor) -> tuple[Tensor, Tensor]:
        """
        Extract Log-mel spectrogram features from the input audio signal.
        """
        return self.featurizer(input_signal), self.out_len(length)
