import logging
import os

import pytest
from test_utils import download_long_audio

pytest.importorskip("pyannote.audio", reason="longform extra not installed")

if not os.environ.get("HF_TOKEN"):
    pytest.skip("HF_TOKEN not set", allow_module_level=True)

import gigaam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize("revision", ["v3_ctc", "v3_e2e_rnnt"])
def test_longform_word_timestamps_structure(revision):
    """Test that longform transcription with word_timestamps returns correct structure"""
    model = gigaam.load_model(revision)
    results = model.transcribe_longform(download_long_audio(), word_timestamps=True)

    assert isinstance(results, list), "Should return list of segments"
    assert len(results) > 0, "Should have at least one segment"

    for segment in results:
        assert hasattr(segment, "text"), "Missing text attribute"
        assert hasattr(segment, "start"), "Missing start attribute"
        assert hasattr(segment, "end"), "Missing end attribute"

        assert segment.words is not None, "Missing words when word_timestamps=True"
        assert isinstance(segment.words, list), "Words should be a list"

        for word in segment.words:
            assert hasattr(word, "text"), "Word missing text attribute"
            assert hasattr(word, "start"), "Word missing start attribute"
            assert hasattr(word, "end"), "Word missing end attribute"
            assert isinstance(word.start, (int, float)), "Word start should be numeric"
            assert isinstance(word.end, (int, float)), "Word end should be numeric"
            assert word.start >= 0, "Word start should be non-negative"
            assert word.end > word.start, "Word end should be after start"


@pytest.mark.parametrize("revision", ["v3_ctc", "v3_e2e_rnnt"])
def test_word_timestamps_offset(revision):
    """Test that word timestamps are correctly offset by segment start time"""
    model = gigaam.load_model(revision)
    results = model.transcribe_longform(download_long_audio(), word_timestamps=True)

    for segment in results:
        segment_start = segment.start
        segment_end = segment.end

        if segment.words is not None and len(segment.words) > 0:
            first_word_start = segment.words[0].start
            assert first_word_start >= segment_start, (
                f"First word starts before segment: {first_word_start} < {segment_start}"
            )

            last_word_end = segment.words[-1].end
            assert last_word_end <= segment_end, f"Last word ends after segment: {last_word_end} > {segment_end}"


@pytest.mark.parametrize("revision", ["v3_ctc", "v3_e2e_rnnt"])
def test_word_timestamps_monotonic(revision):
    """Test that word timestamps are monotonically increasing within segments"""
    model = gigaam.load_model(revision)
    results = model.transcribe_longform(download_long_audio(), word_timestamps=True)

    for i, segment in enumerate(results):
        if segment.words is None or len(segment.words) == 0:
            continue

        prev_end = segment.words[0].start
        for j, word in enumerate(segment.words):
            assert word.start >= prev_end, (
                f"Segment {i}, word {j}: timestamps not monotonic - "
                f"word starts at {word.start} but previous ended at {prev_end}"
            )
            assert word.end > word.start, (
                f"Segment {i}, word {j}: invalid timestamps - end {word.end} <= start {word.start}"
            )
            prev_end = word.end


@pytest.mark.parametrize("revision", ["v3_ctc"])
def test_word_timestamps_coverage(revision):
    """Test that words reasonably cover the segment duration"""
    model = gigaam.load_model(revision)
    results = model.transcribe_longform(download_long_audio(), word_timestamps=True)

    for i, segment in enumerate(results):
        if segment.words is None or len(segment.words) == 0:
            logger.warning(f"Segment {i} has no words")
            continue

        segment_duration = segment.end - segment.start

        words_start = segment.words[0].start
        words_end = segment.words[-1].end
        words_duration = words_end - words_start

        coverage_ratio = words_duration / segment_duration if segment_duration > 0 else 0
        assert coverage_ratio > 0.1, f"Segment {i}: words cover only {coverage_ratio:.2%} of segment duration"


@pytest.mark.parametrize("revision", ["v3_ctc", "v3_e2e_rnnt"])
def test_longform_without_word_timestamps(revision):
    """Test that longform transcription works without word_timestamps"""
    model = gigaam.load_model(revision)
    results = model.transcribe_longform(download_long_audio(), word_timestamps=False)

    assert isinstance(results, list), "Should return list of segments"
    assert len(results) > 0, "Should have at least one segment"

    for segment in results:
        assert hasattr(segment, "text"), "Missing text attribute"
        assert hasattr(segment, "start"), "Missing start attribute"
        assert hasattr(segment, "end"), "Missing end attribute"
        assert segment.words is None, "Should not have words when word_timestamps=False"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
