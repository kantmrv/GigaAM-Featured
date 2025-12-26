import math
import os
import threading

import torch
from pyannote.audio import Model, Pipeline
from pyannote.audio.core.task import Problem, Resolution, Specifications
from pyannote.audio.pipelines import VoiceActivityDetection
from torch.torch_version import TorchVersion

from .audio_utils import SAMPLE_RATE, check_tensor, load_audio

# Cache pipelines by (checkpoint, device_str) tuple for efficiency
_PIPELINES: dict[tuple[str, str], Pipeline] = {}
_PIPELINES_LOCK = threading.Lock()


def get_pipeline(device: torch.device, checkpoint: str = "pyannote/segmentation-3.0") -> Pipeline:
    """
    Retrieves a PyAnnote voice activity detection pipeline for the specified device.

    The pipeline is cached per (checkpoint, device) tuple and reused across subsequent calls.
    This avoids redundant .to(device) calls when the same checkpoint is used on the same device.
    Thread-safe for concurrent access.

    It requires the Hugging Face API token to be set in the HF_TOKEN environment variable.
    """
    cache_key = (checkpoint, str(device))

    with _PIPELINES_LOCK:
        if cache_key in _PIPELINES:
            return _PIPELINES[cache_key]

        # Load pipeline inside lock to prevent duplicate loading
        try:
            hf_token = os.environ["HF_TOKEN"]
        except KeyError as e:
            raise ValueError("HF_TOKEN environment variable is not set") from e

        with torch.serialization.safe_globals(
            [
                TorchVersion,
                Problem,
                Specifications,
                Resolution,
            ]
        ):
            model = Model.from_pretrained(checkpoint, token=hf_token)
        pipeline = VoiceActivityDetection(segmentation=model)  # type: ignore[arg-type]
        pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})
        pipeline = pipeline.to(device)

        _PIPELINES[cache_key] = pipeline
        return pipeline


def segment_audio_file(
    wav: str | torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    max_duration: float = 22.0,
    min_duration: float = 15.0,
    strict_limit_duration: float = 30.0,
    new_chunk_threshold: float = 0.2,
    device: torch.device = torch.device("cpu"),
    checkpoint: str = "pyannote/segmentation-3.0",
) -> tuple[list[torch.Tensor], list[tuple[float, float]]]:
    """
    Segments an audio waveform into smaller chunks based on speech activity.
    """
    if strict_limit_duration <= 0:
        raise ValueError(f"strict_limit_duration must be positive, got {strict_limit_duration}")

    if isinstance(wav, str):
        audio = load_audio(wav)
    else:
        audio = check_tensor(wav, sample_rate)

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    _, num_samples = audio.shape
    file_duration = num_samples / SAMPLE_RATE

    pipeline = get_pipeline(device, checkpoint)

    try:
        sad_segments = pipeline({"waveform": audio, "sample_rate": SAMPLE_RATE})
    except TypeError:
        sad_segments = pipeline({"waveform": audio.to(device), "sample_rate": SAMPLE_RATE})

    segments: list[torch.Tensor] = []
    boundaries: list[tuple[float, float]] = []

    def _extract_slice(start_sec: float, end_sec: float):
        start_idx = math.floor(start_sec * SAMPLE_RATE)
        end_idx = min(math.floor(end_sec * SAMPLE_RATE), num_samples)
        length = end_idx - start_idx

        if length <= 0:
            return

        chunk = audio.narrow(-1, start_idx, length)

        if chunk.dim() > 1 and chunk.shape[0] == 1:
            chunk = chunk.squeeze(0)

        segments.append(chunk)
        boundaries.append((start_sec, end_sec))

    def _add_segment_strict(start_t: float, end_t: float):
        dur = end_t - start_t
        if dur > strict_limit_duration:
            num_splits = math.ceil(dur / strict_limit_duration)
            split_dur = dur / num_splits

            for i in range(num_splits):
                s = start_t + (i * split_dur)
                e = min(start_t + ((i + 1) * split_dur), end_t)
                _extract_slice(s, e)
        else:
            _extract_slice(start_t, end_t)

    timeline = sad_segments.get_timeline().support()
    if len(timeline) == 0:
        _add_segment_strict(0.0, file_duration)
        return segments, boundaries

    curr_start = 0.0
    curr_end = 0.0
    curr_duration = 0.0

    for segment in timeline:
        start = max(0.0, segment.start)
        end = min(file_duration, segment.end)

        has_gap = (start - curr_end) > new_chunk_threshold

        would_exceed_max = curr_duration + (end - curr_end) > max_duration
        reached_min_at_gap = has_gap and curr_duration >= min_duration

        if curr_duration > new_chunk_threshold and (would_exceed_max or reached_min_at_gap):
            _add_segment_strict(curr_start, curr_end)
            curr_start = start

        curr_end = end
        curr_duration = curr_end - curr_start

    if curr_duration > new_chunk_threshold:
        _add_segment_strict(curr_start, curr_end)

    return segments, boundaries
