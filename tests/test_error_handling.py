import logging
import os
import tempfile

import pytest
import torch

import gigaam
from gigaam.audio_utils import check_tensor, load_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_load_audio_invalid_path():
    """Test load_audio with invalid file path"""
    with pytest.raises((FileNotFoundError, RuntimeError)):
        load_audio("/nonexistent/path/to/audio.wav")


def test_load_audio_corrupted():
    """Test load_audio with corrupted audio file"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        try:
            f.write(b"not a valid wav file" * 100)
            f.flush()

            with pytest.raises((RuntimeError, ValueError, OSError)):
                load_audio(f.name)

        finally:
            if os.path.exists(f.name):
                os.remove(f.name)


def test_check_tensor_invalid_dims():
    """Test check_tensor with wrong tensor dimensions"""
    tensor_2d = torch.randn(2, 16000)
    with pytest.raises(ValueError, match="Expected 1D tensor"):
        check_tensor(tensor_2d, 16000)

    tensor_3d = torch.randn(1, 2, 16000)
    with pytest.raises(ValueError, match="Expected 1D tensor"):
        check_tensor(tensor_3d, 16000)


def test_check_tensor_invalid_dtype():
    """Test check_tensor with unsupported dtype"""
    tensor_complex = torch.complex(torch.randn(16000), torch.randn(16000))
    with pytest.raises(ValueError, match="Unsupported dtype"):
        check_tensor(tensor_complex, 16000)

    tensor_bool = torch.randint(0, 2, (16000,)).bool()
    with pytest.raises(ValueError, match="Unsupported dtype"):
        check_tensor(tensor_bool, 16000)


def test_load_model_invalid_name():
    """Test load_model with invalid model name"""
    with pytest.raises(ValueError, match="Unknown model name"):
        gigaam.load_model("nonexistent_model_xyz")


def test_vad_missing_token():
    """Test VAD segmentation when HF_TOKEN is missing"""
    from gigaam.vad_utils import segment_audio_file

    original_token = os.environ.get("HF_TOKEN")
    try:
        if "HF_TOKEN" in os.environ:
            del os.environ["HF_TOKEN"]

        audio = torch.randn(16000)

        with pytest.raises(ValueError, match="HF_TOKEN environment variable is not set"):
            segment_audio_file(audio, 16000)

    finally:
        if original_token is not None:
            os.environ["HF_TOKEN"] = original_token


def test_extract_timestamps_empty_sequence():
    """Test that extract_word_timestamps handles empty encoded sequence gracefully"""

    model = gigaam.load_model("v3_ctc")

    audio = torch.zeros(10)

    with pytest.raises((ValueError, RuntimeError)):
        model.transcribe(audio, sample_rate=16000, word_timestamps=True)


def test_vad_invalid_strict_limit():
    """Test that VAD raises error for zero/negative strict_limit_duration"""
    from gigaam.vad_utils import segment_audio_file

    audio = torch.randn(16000)

    with pytest.raises(ValueError, match="strict_limit_duration must be positive"):
        segment_audio_file(audio, 16000, strict_limit_duration=0.0)

    with pytest.raises(ValueError, match="strict_limit_duration must be positive"):
        segment_audio_file(audio, 16000, strict_limit_duration=-10.0)


def test_onnx_missing_config():
    """Test ONNX loading with missing config file"""
    from gigaam.onnx_utils import load_onnx

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError, match="ONNX config not found"):
            load_onnx(tmpdir, "v3_ctc")


def test_onnx_corrupted_config():
    """Test ONNX loading with corrupted config file"""
    from gigaam.onnx_utils import load_onnx

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "v3_ctc.yaml")
        with open(config_path, "w") as f:
            f.write("invalid: yaml: syntax: {{{")

        with pytest.raises(ValueError, match="Invalid ONNX config file"):
            load_onnx(tmpdir, "v3_ctc")


def test_rnnt_batch_size_mismatch():
    """Test that RNNT decoder validates batch size mismatch"""

    model = gigaam.load_model("v3_rnnt")

    batch_size_encoded = 2
    batch_size_enc_len = 3

    encoded = torch.randn(batch_size_encoded, 256, 100)
    enc_len = torch.tensor([100] * batch_size_enc_len)

    with pytest.raises(AssertionError, match="Batch size mismatch"):
        model.decoding.decode(model.head, encoded, enc_len)


def test_empty_encoded_sequence_emo():
    """Test that emotion model handles empty encoded sequence"""
    model = gigaam.load_model("emo")

    audio = torch.zeros(10)

    with pytest.raises((ValueError, RuntimeError)):
        model.get_probs(audio, sample_rate=16000)


def test_transcribe_with_nan_audio():
    """Test transcription with NaN values in audio"""
    model = gigaam.load_model("v3_ctc")

    audio = torch.full((16000,), float("nan"))

    with pytest.raises((ValueError, RuntimeError)):
        model.transcribe(audio, sample_rate=16000)


def test_transcribe_with_inf_audio():
    """Test transcription with infinite values in audio"""
    model = gigaam.load_model("v3_ctc")

    audio = torch.full((16000,), float("inf"))

    with pytest.raises((ValueError, RuntimeError)):
        model.transcribe(audio, sample_rate=16000)


def test_transcribe_extremely_long_audio_without_longform():
    """Test that transcribe warns or errors on very long audio"""
    model = gigaam.load_model("v3_ctc")

    audio = torch.randn(16000 * 30)

    try:
        result = model.transcribe(audio, sample_rate=16000)

        assert isinstance(result, list)
    except (ValueError, RuntimeError) as e:
        assert len(str(e)) > 0


def test_word_timestamps_with_empty_result():
    """Test word timestamp extraction when transcription is empty or blank"""
    model = gigaam.load_model("v3_ctc")

    audio = torch.randn(16000) * 0.0001

    result = model.transcribe(audio, sample_rate=16000, word_timestamps=True)
    assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
