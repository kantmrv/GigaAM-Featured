"""Tests for module structure and backward compatibility."""

import logging

import pytest
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_audio_utils_module_exports():
    """Test that gigaam.audio_utils module exports correct functions."""
    from gigaam.audio_utils import SAMPLE_RATE, check_tensor, load_audio

    assert SAMPLE_RATE == 16000
    assert callable(load_audio)
    assert callable(check_tensor)


def test_preprocess_exports():
    """Test that gigaam.preprocess exports FeatureExtractor and SpecScaler."""
    from gigaam.preprocess import FeatureExtractor, SpecScaler

    assert FeatureExtractor is not None
    assert SpecScaler is not None


def test_audio_utils_exports_consistency():
    """Test that audio_utils exports are consistent and correct."""
    from gigaam.audio_utils import SAMPLE_RATE, check_tensor, load_audio

    assert SAMPLE_RATE == 16000
    assert callable(load_audio)
    assert callable(check_tensor)


def test_attention_module_exports():
    """Test that gigaam.attention module exports correct functions."""
    from gigaam.attention import apply_masked_flash_attn, apply_rotary_pos_emb, rtt_half

    assert callable(rtt_half)
    assert callable(apply_rotary_pos_emb)
    assert callable(apply_masked_flash_attn)


def test_audio_utils_module_additional_exports():
    """Test that gigaam.audio_utils module exports AudioDataset and format_time."""
    from gigaam.audio_utils import AudioDataset, format_time

    assert callable(format_time)
    assert AudioDataset is not None


def test_gigaam_public_api():
    """Test that gigaam public API is accessible."""
    import gigaam

    assert callable(gigaam.load_model)
    assert callable(gigaam.load_audio)
    assert callable(gigaam.format_time)
    assert gigaam.GigaAM is not None
    assert gigaam.GigaAMASR is not None
    assert gigaam.GigaAMEmo is not None


def test_rtt_half_correctness():
    """Test rtt_half function produces correct output."""
    from gigaam.attention import rtt_half

    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = rtt_half(x)

    expected = torch.tensor([-3.0, -4.0, 1.0, 2.0])
    assert torch.allclose(result, expected), f"Expected {expected}, got {result}"


def test_apply_rotary_pos_emb_shapes():
    """Test apply_rotary_pos_emb maintains correct shapes."""
    from gigaam.attention import apply_rotary_pos_emb

    batch, seq_len, heads, dim = 2, 10, 4, 8
    q = torch.randn(seq_len, batch, heads, dim)
    k = torch.randn(seq_len, batch, heads, dim)
    cos = torch.randn(seq_len, 1, 1, dim)
    sin = torch.randn(seq_len, 1, 1, dim)

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)

    assert q_rot.shape == q.shape, f"q shape mismatch: {q_rot.shape} != {q.shape}"
    assert k_rot.shape == k.shape, f"k shape mismatch: {k_rot.shape} != {k.shape}"


def test_format_time():
    """Test format_time function produces correct output."""
    from gigaam.audio_utils import format_time

    assert format_time(0.0) == "00:00:00"
    assert format_time(61.5) == "01:01:50"
    assert format_time(3661.25) == "01:01:01:25"


def test_audio_dataset_empty_list():
    """Test AudioDataset raises error for empty list."""
    from gigaam.audio_utils import AudioDataset

    with pytest.raises(ValueError, match="empty list"):
        AudioDataset([])


def test_audio_dataset_with_tensors():
    """Test AudioDataset works with tensor inputs."""
    from gigaam.audio_utils import AudioDataset

    tensors = [torch.randn(16000), torch.randn(8000), torch.randn(24000)]
    dataset = AudioDataset(tensors)

    assert len(dataset) == 3
    assert dataset[0].shape == (16000,)
    assert dataset[1].shape == (8000,)


def test_audio_dataset_collate():
    """Test AudioDataset collate function pads correctly."""
    from gigaam.audio_utils import AudioDataset

    tensors = [torch.randn(16000), torch.randn(8000)]
    dataset = AudioDataset(tensors)

    wavs = [dataset[i] for i in range(len(dataset))]
    padded, lengths = dataset.collate(wavs)

    assert padded.shape == (2, 16000)
    assert lengths.tolist() == [16000, 8000]


def test_feature_extractor_init():
    """Test FeatureExtractor initialization with different parameters."""
    from gigaam.preprocess import FeatureExtractor

    extractor = FeatureExtractor(sample_rate=16000, features=80)
    assert extractor.hop_length == 160
    assert extractor.win_length == 400
    assert extractor.n_fft == 400
    assert extractor.center is True

    extractor_custom = FeatureExtractor(
        sample_rate=16000,
        features=80,
        hop_length=128,
        win_length=512,
        n_fft=512,
        center=False,
    )
    assert extractor_custom.hop_length == 128
    assert extractor_custom.win_length == 512
    assert extractor_custom.n_fft == 512
    assert extractor_custom.center is False


def test_check_tensor_resampling():
    """Test check_tensor resamples audio correctly."""
    from gigaam.audio_utils import check_tensor

    audio_8k = torch.randn(8000)
    resampled = check_tensor(audio_8k, sample_rate=8000)

    expected_length = 16000
    assert resampled.shape[0] == expected_length, f"Expected {expected_length}, got {resampled.shape[0]}"


def test_check_tensor_int16_conversion():
    """Test check_tensor converts int16 to float correctly."""
    from gigaam.audio_utils import check_tensor

    audio_int16 = torch.randint(-32768, 32767, (16000,), dtype=torch.int16)
    converted = check_tensor(audio_int16)

    assert converted.dtype == torch.float32
    assert converted.max() <= 1.0
    assert converted.min() >= -1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
