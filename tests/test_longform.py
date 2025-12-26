import logging
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf
from scipy import signal
from test_utils import download_long_audio

pytest.importorskip("pyannote.audio", reason="longform extra not installed")

if not os.environ.get("HF_TOKEN"):
    pytest.skip("HF_TOKEN not set", allow_module_level=True)

import gigaam

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_predictions = {
    "v3_e2e_rnnt": [
        {
            "text": "Вечерня отошла давно, Но в кельях тихо и темно; Уже и сам игумен строгий Свои молитвы прекратил И кости ветхие склонил, Перекрестясь на одр убогий. Кругом и сон, и тишина; Но церкви дверь отворена.",  # noqa: E501
            "start": 0.0,
            "end": 16.80471875,
        },
        {
            "text": "Трепещет луч лампады, И тускло озаряет он И тёмную живопись икон, и возглащённые оклады. И раздаётся в тишине То тяжкий вздох, то шёпот важный, И мрачно дремлет в тишине старинный свод.",  # noqa: E501
            "start": 17.074718750000002,
            "end": 32.549093750000004,
        },
        {
            "text": "Глухой и влажный Стоят за клиросом чернец и грешник, Неподвижны оба. И шёпот их — Как глаз из гроба, И грешник бледен, как мертвец — Монах. Несчастный! Полно, перестань!",  # noqa: E501
            "start": 32.95409375,
            "end": 49.305968750000005,
        },
        {
            "text": "Ужасна исповедь злодея, Заплачена тобою дань Тому, Кто в злобе пламенея Лукавого грешника блюдёт И к вечной гибели ведёт. Смирись, опомнись. Время, время. Раскаянье, покров",  # noqa: E501
            "start": 49.81221875,
            "end": 65.65784375,
        },
        {
            "text": "Я разрешу тебя, грехов сложи мучительное бремя.",
            "start": 65.94471875,
            "end": 70.88909375,
        },
    ],
    "v3_ctc": [
        {
            "text": "вечерня отошла давно но в кельях тихо и темно уже и сам игумен строгий свои молитвы прекратил и кости ветхие склонил перекрестясь на одр убогий кругом и сон и тишина но церкви дверь отворена",  # noqa: E501
            "start": 0.0,
            "end": 16.80471875,
        },
        {
            "text": "трепещет луч лампады и тускло озаряет он и темную живопись икон и позлащенные оклады и раздается в тишине то тяжкий вздох то шепот важный и мрачно дремлет в вашине старинный свод",  # noqa: E501
            "start": 17.074718750000002,
            "end": 32.549093750000004,
        },
        {
            "text": "глухой и влажный стоят за клиросом чернец и грешник неподвижны оба и шепот их как глаз из гроба и грешник бледен как мертвец монах несчастный полно перестань",  # noqa: E501
            "start": 32.95409375,
            "end": 49.305968750000005,
        },
        {
            "text": "ужасна исповедь злодея заплачена тобою дань тому кто в злобе пламенея лукаво грешника блюдет и к вечной гибели ведет смирись опомнись время время раскаянье покров",  # noqa: E501
            "start": 49.81221875,
            "end": 65.65784375,
        },
        {
            "text": "я разрешу тебя грехов сложи мучительное бремя",
            "start": 65.94471875,
            "end": 70.88909375,
        },
    ],
}


def generate_long_audio(duration=60.0, sample_rate=16000, include_silence=True):
    """Generate long test audio with speech-like segments and silence"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t, dtype=np.float32)
    segment_durations = list(np.random.uniform(0.2, 5, size=100))
    current_time = 0.0

    for i, seg_duration in enumerate(segment_durations):
        if current_time + seg_duration > duration:
            break
        seg_t = np.linspace(0, seg_duration, int(sample_rate * seg_duration))
        freq1, freq2, freq3 = 100 + i * 20, 200 + i * 30, 300 + i * 40
        segment = (
            0.4 * np.sin(2 * np.pi * freq1 * seg_t)
            + 0.3 * np.sin(2 * np.pi * freq2 * seg_t)
            + 0.2 * np.sin(2 * np.pi * freq3 * seg_t)
            + 0.1 * np.random.normal(0, 0.2, len(seg_t))
        )
        envelope = signal.windows.tukey(len(segment), alpha=0.1)
        segment = segment * envelope
        start_idx, end_idx = (
            int(current_time * sample_rate),
            int(current_time * sample_rate) + len(segment),
        )
        audio[start_idx:end_idx] = segment
        if include_silence and i < len(segment_durations) - 1:
            current_time += seg_duration + np.random.uniform(0.1, 0.5)
        else:
            current_time += seg_duration
    return audio


def validate_segmentation_boundaries(boundaries: list[tuple[float, float]], audio_duration: float):
    """Validate segmentation boundaries meet requirements"""
    issues = []
    total_duration = 0.0

    for i, (start, end) in enumerate(boundaries):
        duration = end - start
        if duration < 0.2:
            issues.append(f"Segment {i} too short: {duration:.2f}s")
        if duration > 30.0:
            issues.append(f"Segment {i} too long: {duration:.2f}s")
        if start >= end:
            issues.append(f"Segment {i} invalid boundaries: {start:.2f}-{end:.2f}")
        total_duration += duration

    if boundaries and boundaries[-1][1] > audio_duration:
        issues.append(f"Last segment exceeds audio: {boundaries[-1][1]:.2f} > {audio_duration:.2f}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "total_segments": len(boundaries),
    }


@pytest.mark.parametrize("duration", [30.0, 60.0, 120.0])
def test_segmentation_functionality(duration):
    """Test audio segmentation with different durations"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        try:
            audio = generate_long_audio(duration=duration)
            sf.write(f.name, audio, 16000)

            from gigaam.vad_utils import segment_audio_file

            segments, boundaries = segment_audio_file(f.name, sample_rate=16000)

            validation = validate_segmentation_boundaries(boundaries, duration)
            assert validation["valid"], f"Boundary validation failed: {validation['issues']}"
            assert len(segments) == len(boundaries), "Segments and boundaries count mismatch"

            logger.info(f"Segmentation: {len(segments)} segments for {duration}s audio")

        finally:
            if os.path.exists(f.name):
                os.remove(f.name)


@pytest.mark.parametrize("revision", ["v3_ctc", "v3_e2e_rnnt"])
def test_transcribe_longform(revision):
    """Test longform transcription for different models"""
    model = gigaam.load_model(revision)
    results = model.transcribe_longform(download_long_audio())
    ref = _predictions[revision]

    assert isinstance(results, list), "Should return list of segments"
    assert len(results) == len(ref), "Distinct results len from reference"

    for segment, ref_segment in zip(results, ref, strict=True):
        assert hasattr(segment, "text"), "Missing text attribute"
        assert hasattr(segment, "start"), "Missing start attribute"
        assert hasattr(segment, "end"), "Missing end attribute"
        start, end = segment.start, segment.end
        ref_start, ref_end = ref_segment["start"], ref_segment["end"]
        assert abs(start - ref_start) < 0.1 and abs(end - ref_end) < 0.1, (
            f"Segments are not close {start, end} and {ref_start, ref_end}"
        )
        assert segment.text == ref_segment["text"], f"Different transcription: {segment.text} and {ref_segment['text']}"


@pytest.mark.parametrize("revision", ["v3_ctc"])
def test_longform_consistency(revision):
    """Test that multiple runs produce consistent results"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        try:
            audio = generate_long_audio(duration=30.0)
            sf.write(f.name, audio, 16000)

            model = gigaam.load_model(revision)
            results1 = model.transcribe_longform(f.name)
            results2 = model.transcribe_longform(f.name)

            assert len(results1) == len(results2), "Inconsistent segment count"
            for seg1, seg2 in zip(results1, results2, strict=True):
                assert seg1.start == seg2.start, "Inconsistent start boundaries"
                assert seg1.end == seg2.end, "Inconsistent end boundaries"

        finally:
            if os.path.exists(f.name):
                os.remove(f.name)


def test_segmentation_edge_cases():
    """Test segmentation with edge cases"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        try:
            audio = generate_long_audio(duration=0.5)
            sf.write(f.name, audio, 16000)

            from gigaam.vad_utils import segment_audio_file

            segments, boundaries = segment_audio_file(f.name, sample_rate=16000)

            assert isinstance(segments, list), "Should return list even for short audio"

        finally:
            if os.path.exists(f.name):
                os.remove(f.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
