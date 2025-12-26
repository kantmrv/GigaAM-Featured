import logging
import tempfile

import numpy as np
import pytest
import soundfile as sf
import torch
from test_utils import download_short_audio

import gigaam
from gigaam.audio_utils import SAMPLE_RATE, load_audio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def test_audio():
    """Provide test audio file for all tests"""
    return download_short_audio()


@pytest.mark.parametrize("revision", ["v3_ctc", "v3_rnnt"])
@pytest.mark.partial
def test_tensor_input_shortform(revision, test_audio):
    """Test transcribe_featured with tensor input for short audio"""
    model = gigaam.load_model(revision)

    wav_tensor = load_audio(test_audio)

    results_tensor = model.transcribe(wav_tensor, word_timestamps=False)

    results_file = model.transcribe(test_audio, word_timestamps=False)

    assert len(results_tensor) == len(results_file)
    assert results_tensor[0].text == results_file[0].text

    logger.info(f"{revision}: Tensor input shortform test passed")


@pytest.mark.parametrize("revision", ["v3_ctc", "v3_rnnt"])
@pytest.mark.partial
def test_tensor_input_with_word_timestamps(revision, test_audio):
    """Test tensor input with word timestamps enabled"""
    model = gigaam.load_model(revision)

    wav_tensor = load_audio(test_audio)

    results = model.transcribe(wav_tensor, word_timestamps=True)

    assert isinstance(results, list)
    assert len(results) == 1

    segment = results[0]
    assert hasattr(segment, "text")
    assert segment.words is not None
    assert isinstance(segment.words, list)
    assert len(segment.words) > 0

    for word_segment in segment.words:
        assert hasattr(word_segment, "text")
        assert hasattr(word_segment, "start")
        assert hasattr(word_segment, "end")
        assert isinstance(word_segment.text, str)
        assert isinstance(word_segment.start, float)
        assert isinstance(word_segment.end, float)

    logger.info(f"{revision}: Tensor input with word timestamps test passed")


@pytest.mark.parametrize("revision", ["v3_ctc"])
@pytest.mark.partial
def test_tensor_dimension_validation(revision):
    """Test that 1D tensor requirement is enforced"""
    model = gigaam.load_model(revision)

    wav_2d = torch.randn(2, 16000)
    with pytest.raises(ValueError, match="Expected 1D tensor, got 2D tensor"):
        model.transcribe(wav_2d, word_timestamps=False)

    wav_3d = torch.randn(1, 2, 16000)
    with pytest.raises(ValueError, match="Expected 1D tensor, got 3D tensor"):
        model.transcribe(wav_3d, word_timestamps=False)

    wav_0d = torch.tensor(0.5)
    with pytest.raises(ValueError, match="Expected 1D tensor, got 0D tensor"):
        model.transcribe(wav_0d, word_timestamps=False)

    logger.info(f"{revision}: Tensor dimension validation test passed")


@pytest.mark.parametrize("revision", ["v3_ctc"])
@pytest.mark.partial
def test_tensor_dtype_validation(revision):
    """Test that appropriate dtypes are accepted"""
    model = gigaam.load_model(revision)

    wav_float32 = torch.randn(16000, dtype=torch.float32)
    results = model.transcribe(wav_float32, word_timestamps=False)
    assert isinstance(results, list)

    wav_float64 = torch.randn(16000, dtype=torch.float64)
    results = model.transcribe(wav_float64, word_timestamps=False)
    assert isinstance(results, list)

    wav_int16 = torch.randint(-32768, 32767, (16000,), dtype=torch.int16)
    results = model.transcribe(wav_int16, word_timestamps=False)
    assert isinstance(results, list)

    wav_int32 = torch.randint(-32768, 32767, (16000,), dtype=torch.int32)
    with pytest.raises(ValueError, match="Expected float32, float64, or int16 tensor"):
        model.transcribe(wav_int32, word_timestamps=False)

    logger.info(f"{revision}: Tensor dtype validation test passed")


@pytest.mark.parametrize("revision", ["v3_ctc"])
@pytest.mark.partial
def test_int16_conversion(revision, test_audio):
    """Test that int16 tensors are properly converted to float"""
    model = gigaam.load_model(revision)

    wav_float = load_audio(test_audio)
    wav_int16 = (wav_float * 32768.0).to(torch.int16)

    results_int16 = model.transcribe(wav_int16, word_timestamps=False)

    results_float = model.transcribe(wav_float, word_timestamps=False)

    assert results_int16[0].text == results_float[0].text

    logger.info(f"{revision}: Int16 conversion test passed")


@pytest.mark.parametrize("revision", ["v3_ctc", "v3_rnnt"])
@pytest.mark.partial
def test_tensor_equivalence_with_file_path(revision, test_audio):
    """Test that tensor input produces same results as file path input"""
    model = gigaam.load_model(revision)

    wav_tensor = load_audio(test_audio)

    results_tensor = model.transcribe(wav_tensor, word_timestamps=False)
    results_file = model.transcribe(test_audio, word_timestamps=False)

    assert results_tensor[0].text == results_file[0].text

    logger.info(f"{revision}: Tensor equivalence with file path test passed")


@pytest.mark.parametrize("revision", ["v3_ctc"])
@pytest.mark.partial
def test_tensor_input_longform_validation(revision):
    """Test tensor validation in transcribe_longform_featured"""
    model = gigaam.load_model(revision)

    duration = 60.0
    sr = SAMPLE_RATE
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.1 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sr)
        tmp_path = tmp.name

    try:
        wav_tensor = torch.from_numpy(audio)

        wav_2d = wav_tensor.unsqueeze(0)
        with pytest.raises(ValueError, match="Expected 1D tensor, got 2D tensor"):
            model.transcribe_longform(wav_2d, word_timestamps=False)

        wav_int32 = wav_tensor.to(torch.int32)
        with pytest.raises(ValueError, match="Expected float32, float64, or int16 tensor"):
            model.transcribe_longform(wav_int32, word_timestamps=False)

        logger.info(f"{revision}: Longform tensor validation test passed")

    finally:
        import os

        os.remove(tmp_path)


@pytest.mark.parametrize("revision", ["v3_ssl", "v2_ssl"])
@pytest.mark.partial
def test_embed_audio_tensor_input(revision, test_audio):
    """Test embed_audio with tensor input for SSL models"""
    model = gigaam.load_model(revision)

    wav_tensor = load_audio(test_audio)

    embedding_tensor, length_tensor = model.embed_audio(wav_tensor)

    embedding_file, length_file = model.embed_audio(test_audio)

    assert torch.allclose(embedding_tensor, embedding_file, atol=1e-5)
    assert torch.equal(length_tensor, length_file)

    logger.info(f"{revision}: embed_audio tensor input test passed")


@pytest.mark.parametrize("revision", ["emo"])
@pytest.mark.partial
def test_get_probs_tensor_input(revision, test_audio):
    """Test get_probs with tensor input for Emo models"""
    model = gigaam.load_model(revision)

    wav_tensor = load_audio(test_audio)

    probs_tensor = model.get_probs(wav_tensor)

    probs_file = model.get_probs(test_audio)

    assert probs_tensor.keys() == probs_file.keys()
    for emotion in probs_tensor:
        assert abs(probs_tensor[emotion] - probs_file[emotion]) < 1e-5

    logger.info(f"{revision}: get_probs tensor input test passed")


@pytest.mark.parametrize("revision", ["v3_ctc"])
@pytest.mark.partial
def test_tensor_wrong_device(revision, test_audio):
    """Test handling of tensors on different device than model"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for device mismatch test")

    model_cpu = gigaam.load_model(revision, device="cpu")

    wav_tensor = load_audio(test_audio)

    results_cpu = model_cpu.transcribe(wav_tensor, word_timestamps=False)
    assert isinstance(results_cpu, list)

    model_cuda = gigaam.load_model(revision, device="cuda")

    results_mixed = model_cuda.transcribe(wav_tensor, word_timestamps=False)
    assert isinstance(results_mixed, list)

    logger.info(f"{revision}: Tensor device handling test passed")


@pytest.mark.parametrize("revision", ["v3_ctc"])
@pytest.mark.partial
def test_tensor_requires_grad(revision, test_audio):
    """Test handling of tensors with requires_grad=True"""
    model = gigaam.load_model(revision)

    wav_tensor = load_audio(test_audio)
    wav_tensor.requires_grad_(True)

    results = model.transcribe(wav_tensor, word_timestamps=False)
    assert isinstance(results, list)

    logger.info(f"{revision}: Tensor requires_grad test passed")


@pytest.mark.parametrize("revision", ["v3_ssl"])
@pytest.mark.partial
def test_embed_audio_different_sample_rates(revision):
    """Test embed_audio with different sample rates requiring resampling"""
    model = gigaam.load_model(revision)

    duration = 2.0
    sample_rates = [8000, 22050, 44100, 48000]

    for sr in sample_rates:
        t = np.linspace(0, duration, int(sr * duration))
        audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        wav_tensor = torch.from_numpy(audio)

        embedding, length = model.embed_audio(wav_tensor, sample_rate=sr)
        assert embedding is not None
        assert length is not None

    logger.info(f"{revision}: embed_audio sample rate resampling test passed")


@pytest.mark.parametrize("revision", ["emo"])
@pytest.mark.partial
def test_get_probs_different_sample_rates(revision):
    """Test get_probs with different sample rates requiring resampling"""
    model = gigaam.load_model(revision)

    duration = 2.0
    sample_rates = [8000, 22050, 44100, 48000]

    for sr in sample_rates:
        t = np.linspace(0, duration, int(sr * duration))
        audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        wav_tensor = torch.from_numpy(audio)

        probs = model.get_probs(wav_tensor, sample_rate=sr)
        assert isinstance(probs, dict)
        assert len(probs) > 0

    logger.info(f"{revision}: get_probs sample rate resampling test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "partial"])
