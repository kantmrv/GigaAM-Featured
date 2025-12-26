import warnings
from subprocess import CalledProcessError, run

import numpy as np
import torch
import torchaudio
from torch import Tensor

SAMPLE_RATE = 16000


def _load_audio_torchaudio(audio_path: str) -> Tensor:
    """Load audio using torchaudio backend."""
    waveform, file_sample_rate = torchaudio.load(audio_path)

    if file_sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(file_sample_rate, SAMPLE_RATE)
        waveform = resampler(waveform)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform[0]

    return waveform


def _load_audio_ffmpeg(audio_path: str) -> Tensor:
    """Load audio using ffmpeg backend (supports more formats)."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        audio_path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(SAMPLE_RATE),
        "-",
    ]
    audio = run(cmd, capture_output=True, check=True).stdout

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return torch.frombuffer(audio, dtype=torch.int16).float() / 32768.0


def load_audio(audio_path: str) -> Tensor:
    """
    Load an audio file and resample it to SAMPLE_RATE (16kHz).
    """
    try:
        return _load_audio_torchaudio(audio_path)
    except (RuntimeError, OSError):
        pass

    try:
        return _load_audio_ffmpeg(audio_path)
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {audio_path}") from e


def check_tensor(audio_tensor: Tensor, sample_rate: int = SAMPLE_RATE) -> Tensor:
    """
    Validate and normalize an audio tensor for processing.
    """
    if audio_tensor.ndim != 1:
        raise ValueError(f"Expected 1D tensor, got {audio_tensor.ndim}D tensor")
    if audio_tensor.dtype not in (torch.float32, torch.float64, torch.int16):
        raise ValueError(f"Unsupported dtype {audio_tensor.dtype}. Expected float32, float64, or int16 tensor")
    if audio_tensor.dtype == torch.int16:
        audio_tensor = audio_tensor.float() / 32768.0
    if not torch.isfinite(audio_tensor).all():
        raise ValueError("Audio tensor contains NaN or Inf values")
    if sample_rate != SAMPLE_RATE:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, SAMPLE_RATE)

    return audio_tensor


def format_time(seconds: float) -> str:
    """
    Format time in seconds to HH:MM:SS:mm format.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    full_seconds = int(seconds)
    milliseconds = int((seconds - full_seconds) * 100)

    if hours > 0:
        return f"{hours:02}:{minutes:02}:{full_seconds:02}:{milliseconds:02}"
    return f"{minutes:02}:{full_seconds:02}:{milliseconds:02}"


class AudioDataset(torch.utils.data.Dataset[Tensor]):
    """
    Dataset for creating batched audio inputs from mixed input types.

    Supports file paths, numpy arrays, and PyTorch tensors as inputs.
    """

    def __init__(self, lst: list[str | np.ndarray[tuple[int], np.dtype[np.floating[np.typing.NBitBase]]] | Tensor]):
        """
        Initialize the dataset.
        """
        if len(lst) == 0:
            raise ValueError("AudioDataset cannot be initialized with an empty list")
        assert isinstance(lst[0], (str, np.ndarray, Tensor)), f"Unexpected dtype: {type(lst[0])}"
        self.lst = lst

    def __len__(self) -> int:
        return len(self.lst)

    def __getitem__(self, idx: int) -> Tensor:
        item = self.lst[idx]
        if isinstance(item, str):
            wav_tns = load_audio(item)
        elif isinstance(item, np.ndarray):
            wav_tns = torch.from_numpy(item)
        elif isinstance(item, Tensor):
            wav_tns = item
        else:
            raise RuntimeError(f"Unexpected sample type: {type(item)} at idx={idx}")
        return wav_tns

    @staticmethod
    def collate(wavs: list[Tensor]) -> tuple[Tensor, Tensor]:
        """
        Collate function for batching variable-length audio tensors.

        Pads all tensors to the maximum length in the batch.
        """
        lengths = torch.tensor([len(wav) for wav in wavs])
        max_len = int(lengths.max().item())
        wav_tns = torch.zeros(len(wavs), max_len, dtype=wavs[0].dtype)
        for idx, wav in enumerate(wavs):
            wav_tns[idx, : wav.shape[-1]] = wav.squeeze()
        return wav_tns, lengths
