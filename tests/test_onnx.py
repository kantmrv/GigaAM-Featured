import logging
import os
import shutil

import numpy as np
import pytest
from test_utils import download_short_audio

import gigaam
from gigaam.onnx_utils import infer_onnx, load_onnx
from gigaam.types import TranscribedSegment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def test_audio():
    """Provide test audio file for all tests"""
    return download_short_audio()


@pytest.mark.parametrize("revision", ["emo", "v2_ssl", "v3_ctc", "v3_e2e_rnnt"])
def test_onnx_converting(revision, test_audio):
    """Test specific model revision loads and processes audio (partial models enabled)"""
    onnx_dir = "test_onnx_tmp"
    try:
        model = gigaam.load_model(revision, fp16_encoder=False)
        model.to_onnx(dir_path=onnx_dir)
        sessions, model_cfg = load_onnx(onnx_dir, revision)
        result = infer_onnx(test_audio, model_cfg, sessions)

        if "ssl" in revision:
            orig_embed, orig_len = model.embed_audio(test_audio)
            orig_embed = orig_embed.detach().cpu().numpy()
            # result is now tuple (encoded, encoded_len)
            onnx_embed, onnx_len = result
            diff = np.abs(orig_embed - onnx_embed).max()
            assert diff < 0.01, f"{revision}: ONNX embed failed with diff {diff}"
            assert orig_embed.shape == onnx_embed.shape, f"{revision}: Shape mismatch"

        elif "emo" in revision:
            orig_probs = model.get_probs(test_audio)
            # result is now dict directly
            pred_probs = result
            assert all(abs(orig_probs[em] - pred_probs[em]) < 1e-3 for em in orig_probs), (
                f"{revision}: ONNX emotions probs failed: {pred_probs}"
            )

        else:
            orig_text = model.transcribe(test_audio)[0].text
            assert orig_text == result, f"{revision}: ONNX transcribe failed: {result}"
    finally:
        shutil.rmtree(onnx_dir, ignore_errors=True)


@pytest.mark.parametrize("revision", ["v3_ctc", "v3_e2e_rnnt"])
def test_onnx_word_timestamps(revision, test_audio):
    """Test word timestamp extraction with ONNX models"""
    onnx_dir = "test_onnx_tmp"
    try:
        model = gigaam.load_model(revision, fp16_encoder=False)
        model.to_onnx(dir_path=onnx_dir)
        sessions, model_cfg = load_onnx(onnx_dir, revision)

        # Test with word_timestamps=True
        result = infer_onnx(test_audio, model_cfg, sessions, word_timestamps=True)

        # result should be TranscribedSegment with "text" and "words"
        assert isinstance(result, TranscribedSegment), f"{revision}: Expected TranscribedSegment result"
        assert hasattr(result, "text"), f"{revision}: Missing 'text' attribute"
        assert result.words is not None, f"{revision}: Missing 'words'"
        assert isinstance(result.words, list), f"{revision}: 'words' should be list"

        # Verify each word has required fields
        for word in result.words:
            assert hasattr(word, "text"), f"{revision}: Word missing 'text' attribute"
            assert hasattr(word, "start"), f"{revision}: Word missing 'start' attribute"
            assert hasattr(word, "end"), f"{revision}: Word missing 'end' attribute"
            assert word.start <= word.end, f"{revision}: Invalid word timestamps"

        logger.info(f"{revision} word timestamps: {result}")
    finally:
        shutil.rmtree(onnx_dir, ignore_errors=True)


@pytest.mark.longform
@pytest.mark.skipif(not os.environ.get("HF_TOKEN"), reason="HF_TOKEN not set")
@pytest.mark.parametrize("revision", ["v3_ctc", "v3_e2e_rnnt"])
def test_onnx_longform(revision, test_audio):
    """Test longform transcription with ONNX models (requires HF_TOKEN)"""
    pytest.importorskip("pyannote.audio", reason="longform extra not installed")
    from gigaam.onnx_utils import infer_onnx_longform

    onnx_dir = "test_onnx_tmp"
    try:
        model = gigaam.load_model(revision, fp16_encoder=False)
        model.to_onnx(dir_path=onnx_dir)
        sessions, model_cfg = load_onnx(onnx_dir, revision)

        # Test longform transcription
        result = infer_onnx_longform(test_audio, model_cfg, sessions)

        # result should be list of TranscribedSegment
        assert isinstance(result, list), f"{revision}: Expected list result"
        assert len(result) > 0, f"{revision}: No segments returned"

        # Verify each segment has required fields
        for segment in result:
            assert hasattr(segment, "text"), f"{revision}: Segment missing 'text' attribute"
            assert hasattr(segment, "start"), f"{revision}: Segment missing 'start' attribute"
            assert hasattr(segment, "end"), f"{revision}: Segment missing 'end' attribute"
            assert segment.start <= segment.end, f"{revision}: Invalid segment timestamps"

        # Test with word timestamps
        result_with_words = infer_onnx_longform(test_audio, model_cfg, sessions, word_timestamps=True)

        for segment in result_with_words:
            assert segment.words is not None, f"{revision}: Segment missing 'words' with word_timestamps=True"
            # Verify each word has required fields
            for word in segment.words:
                assert hasattr(word, "text"), f"{revision}: Word missing 'text' attribute"
                assert hasattr(word, "start"), f"{revision}: Word missing 'start' attribute"
                assert hasattr(word, "end"), f"{revision}: Word missing 'end' attribute"
                assert word.start <= word.end, f"{revision}: Invalid word timestamps"

        logger.info(f"{revision} longform result: {result}")
    finally:
        shutil.rmtree(onnx_dir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "partial"])
