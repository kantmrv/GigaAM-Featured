from dataclasses import dataclass


@dataclass(slots=True)
class WordTimestamp:
    """Single word with timing information."""

    text: str
    start: float
    end: float


@dataclass(slots=True)
class TranscribedSegment:
    """Transcription result for a segment of audio."""

    text: str
    start: float | None = None
    end: float | None = None
    words: list[WordTimestamp] | None = None
