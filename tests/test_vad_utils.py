import logging
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf
import torch

from gigaam.vad_utils import segment_audio_file

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_segment_invalid_strict_limit():
    """Test that segmentation raises error for invalid strict_limit_duration"""
    audio = np.random.randn(16000).astype(np.float32)

    with pytest.raises(ValueError, match="strict_limit_duration must be positive"):
        segment_audio_file(torch.from_numpy(audio), sample_rate=16000, strict_limit_duration=0.0)

    with pytest.raises(ValueError, match="strict_limit_duration must be positive"):
        segment_audio_file(torch.from_numpy(audio), sample_rate=16000, strict_limit_duration=-5.0)


def test_segment_empty_audio():
    """Test segmentation with silent audio (all zeros)"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        try:
            audio = np.zeros(16000 * 5, dtype=np.float32)
            sf.write(f.name, audio, 16000)

            segments, boundaries = segment_audio_file(f.name, sample_rate=16000)

            assert isinstance(segments, list), "Should return list"
            assert isinstance(boundaries, list), "Should return boundaries list"
            assert len(segments) >= 1, "Should have at least one segment"
            assert len(boundaries) >= 1, "Should have at least one boundary"
            assert len(segments) == len(boundaries), "Segments and boundaries count mismatch"

        finally:
            if os.path.exists(f.name):
                os.remove(f.name)


def test_segment_short_audio():
    """Test segmentation with very short audio"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        try:
            audio = np.random.randn(8000).astype(np.float32) * 0.1
            sf.write(f.name, audio, 16000)

            segments, boundaries = segment_audio_file(f.name, sample_rate=16000)

            assert isinstance(segments, list), "Should return list"
            assert isinstance(boundaries, list), "Should return boundaries"

        finally:
            if os.path.exists(f.name):
                os.remove(f.name)


def test_segment_custom_parameters():
    """Test segmentation with custom VAD parameters"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        try:
            duration = 30.0
            t = np.linspace(0, duration, int(16000 * duration))
            audio = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
            sf.write(f.name, audio, 16000)

            segments, boundaries = segment_audio_file(
                f.name,
                sample_rate=16000,
                max_duration=10.0,
                min_duration=5.0,
                strict_limit_duration=15.0,
            )

            assert isinstance(segments, list), "Should return list"
            assert isinstance(boundaries, list), "Should return boundaries"

        finally:
            if os.path.exists(f.name):
                os.remove(f.name)


def test_pipeline_caching():
    """Test that VAD pipeline is cached correctly across calls"""
    import gigaam.vad_utils
    from gigaam.vad_utils import get_pipeline

    gigaam.vad_utils._PIPELINES.clear()

    device = torch.device("cpu")
    checkpoint = "pyannote/segmentation-3.0"

    pipeline1 = get_pipeline(device, checkpoint)
    assert pipeline1 is not None, "Pipeline should be created"

    pipeline2 = get_pipeline(device, checkpoint)
    assert pipeline2 is pipeline1, "Pipeline should be cached"


def test_strict_limit_subdivision():
    """Test that segments exceeding strict_limit_duration are subdivided"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        try:
            duration = 40.0
            t = np.linspace(0, duration, int(16000 * duration))
            audio = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
            sf.write(f.name, audio, 16000)

            segments, boundaries = segment_audio_file(
                f.name,
                sample_rate=16000,
                max_duration=50.0,
                strict_limit_duration=20.0,
            )

            for i, (start, end) in enumerate(boundaries):
                segment_duration = end - start
                assert segment_duration <= 20.05, (
                    f"Segment {i} exceeds strict_limit_duration: {segment_duration:.2f}s > 20s"
                )

        finally:
            if os.path.exists(f.name):
                os.remove(f.name)


def test_segment_tensor_input():
    """Test segmentation with tensor input vs file path"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        try:
            duration = 10.0
            t = np.linspace(0, duration, int(16000 * duration))
            audio_np = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
            sf.write(f.name, audio_np, 16000)

            segments_file, boundaries_file = segment_audio_file(f.name, sample_rate=16000)

            audio_tensor = torch.from_numpy(audio_np)
            segments_tensor, boundaries_tensor = segment_audio_file(audio_tensor, sample_rate=16000)

            assert len(segments_file) == len(segments_tensor), "Different number of segments for file vs tensor input"
            assert len(boundaries_file) == len(boundaries_tensor), (
                "Different number of boundaries for file vs tensor input"
            )

        finally:
            if os.path.exists(f.name):
                os.remove(f.name)


def test_segment_boundary_validation():
    """Test that segment boundaries are valid"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        try:
            duration = 30.0
            t = np.linspace(0, duration, int(16000 * duration))
            audio = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
            sf.write(f.name, audio, 16000)

            segments, boundaries = segment_audio_file(f.name, sample_rate=16000)

            for i, (start, end) in enumerate(boundaries):
                assert start >= 0, f"Segment {i} has negative start: {start}"
                assert end > start, f"Segment {i} has end <= start: {start}, {end}"
                assert end <= duration + 0.1, f"Segment {i} exceeds audio duration: {end} > {duration}"

            assert len(segments) == len(boundaries), "Segments and boundaries count mismatch"

        finally:
            if os.path.exists(f.name):
                os.remove(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
