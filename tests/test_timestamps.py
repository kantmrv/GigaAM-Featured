import logging

import pytest
from test_utils import download_short_audio

import gigaam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def test_audio():
    """Provide test audio file for all tests"""
    return download_short_audio()


@pytest.mark.parametrize("revision", ["v3_ctc", "v3_e2e_ctc"])
@pytest.mark.partial
def test_word_timestamps_shortform_ctc(revision, test_audio):
    """Test word-level timestamps for CTC models on short audio"""
    model = gigaam.load_model(revision)

    results = model.transcribe(test_audio, word_timestamps=True)

    assert isinstance(results, list)
    assert len(results) == 1

    result = results[0]
    assert hasattr(result, "text")
    assert hasattr(result, "words")
    assert isinstance(result.text, str)
    assert isinstance(result.words, list)
    assert len(result.text) > 0
    assert len(result.words) > 0

    for word_segment in result.words:
        assert hasattr(word_segment, "text")
        assert hasattr(word_segment, "start")
        assert hasattr(word_segment, "end")
        assert isinstance(word_segment.text, str)
        assert isinstance(word_segment.start, float)
        assert isinstance(word_segment.end, float)
        assert word_segment.start >= 0.0
        assert word_segment.end > word_segment.start
        assert len(word_segment.text) > 0

    words = result.words
    for i in range(len(words) - 1):
        assert words[i].end <= words[i + 1].start + 0.1

    logger.info(f"{revision}: Word timestamps test passed ({len(words)} words)")


@pytest.mark.parametrize("revision", ["v3_rnnt", "v3_e2e_rnnt"])
@pytest.mark.partial
def test_word_timestamps_shortform_rnnt(revision, test_audio):
    """Test word-level timestamps for RNNT models on short audio"""
    model = gigaam.load_model(revision)

    results = model.transcribe(test_audio, word_timestamps=True)

    assert isinstance(results, list)
    assert len(results) == 1

    result = results[0]
    assert hasattr(result, "text")
    assert hasattr(result, "words")
    assert isinstance(result.text, str)
    assert isinstance(result.words, list)
    assert len(result.text) > 0
    assert len(result.words) > 0

    for word_segment in result.words:
        assert hasattr(word_segment, "text")
        assert hasattr(word_segment, "start")
        assert hasattr(word_segment, "end")
        assert isinstance(word_segment.text, str)
        assert isinstance(word_segment.start, float)
        assert isinstance(word_segment.end, float)
        assert word_segment.start >= 0.0
        assert word_segment.end > word_segment.start
        assert len(word_segment.text) > 0

    words = result.words
    for i in range(len(words) - 1):
        assert words[i].end <= words[i + 1].start + 0.1

    logger.info(f"{revision}: Word timestamps test passed ({len(words)} words)")


@pytest.mark.parametrize("revision", ["v3_ctc", "v3_rnnt"])
@pytest.mark.partial
def test_word_timestamps_vs_no_timestamps(revision, test_audio):
    """Test that word_timestamps=False returns different format"""
    model = gigaam.load_model(revision)

    results_no_ts = model.transcribe(test_audio, word_timestamps=False)
    assert isinstance(results_no_ts, list)
    assert len(results_no_ts) == 1
    assert hasattr(results_no_ts[0], "text")
    assert results_no_ts[0].start is None
    assert results_no_ts[0].end is None
    assert results_no_ts[0].words is None

    results_with_ts = model.transcribe(test_audio, word_timestamps=True)
    assert isinstance(results_with_ts, list)
    assert len(results_with_ts) == 1
    assert hasattr(results_with_ts[0], "text")
    assert results_with_ts[0].words is not None
    assert len(results_with_ts[0].words) > 1

    text_with_ts = results_with_ts[0].text
    text_no_ts = results_no_ts[0].text

    text_with_ts_normalized = text_with_ts.lower().replace(" ", "")
    text_no_ts_normalized = text_no_ts.lower().replace(" ", "")

    set_with_ts = set(text_with_ts_normalized)
    set_no_ts = set(text_no_ts_normalized)
    similarity = len(set_with_ts & set_no_ts) / len(set_with_ts | set_no_ts) if set_with_ts or set_no_ts else 0.0
    assert similarity > 0.9

    logger.info(f"{revision}: Timestamps vs no timestamps test passed")


@pytest.mark.parametrize("revision", ["v3_ctc", "v3_rnnt"])
@pytest.mark.partial
def test_frame_shift_precision(revision, test_audio):
    """Test that frame shift calculation preserves floating point precision"""
    model = gigaam.load_model(revision)

    results = model.transcribe(test_audio, word_timestamps=True)

    assert len(results) == 1
    words = results[0].words

    has_decimal_precision = False
    for word_segment in words:
        start_decimal = word_segment.start % 1
        end_decimal = word_segment.end % 1
        if start_decimal > 0.01 or end_decimal > 0.01:
            has_decimal_precision = True
            break

    assert has_decimal_precision, "Timestamps should have sub-second precision"

    logger.info(f"{revision}: Frame shift precision test passed")


@pytest.mark.parametrize("revision", ["v3_e2e_ctc", "v3_e2e_rnnt"])
@pytest.mark.partial
def test_word_timestamps_different_sample_rates(revision, test_audio):
    """Test word timestamps with default sample rate"""
    model = gigaam.load_model(revision)

    results = model.transcribe(test_audio, word_timestamps=True)

    assert isinstance(results, list)
    assert len(results) == 1
    words = results[0].words
    assert len(words) > 0

    for word_segment in words:
        assert 0.0 <= word_segment.start <= 60.0
        assert 0.0 <= word_segment.end <= 60.0
        assert word_segment.end > word_segment.start

    logger.info(f"{revision}: Sample rate test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "partial"])
