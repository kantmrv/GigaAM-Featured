from collections.abc import Sequence

import torch
from torch import Tensor

from .decoder import CTCHead, RNNTHead
from .decoding import Tokenizer
from .types import WordTimestamp


def decode_with_alignment_ctc(
    head: CTCHead, encoded_seq: Tensor, seq_len: int, blank_id: int
) -> tuple[list[int], list[int]]:
    """
    Greedy CTC decoding that also keeps the encoder frame index for every token.
    """
    log_probs = head(encoder_output=encoded_seq)
    labels = log_probs.argmax(dim=-1)
    frames = labels[0, :seq_len].cpu().tolist()

    hyp, token_frames = [], []
    prev = blank_id
    for t, label in enumerate(frames):
        if label != blank_id and (label != prev or prev == blank_id):
            hyp.append(int(label))
            token_frames.append(t)
        prev = label
    return hyp, token_frames


def decode_with_alignment_rnnt(
    head: RNNTHead, encoded_seq: Tensor, seq_len: int, blank_id: int, max_symbols: int
) -> tuple[list[int], list[int]]:
    """
    Greedy RNNT decoding that also keeps the encoder frame index for every token.
    """
    hyp, token_frames = [], []
    dec_state = None
    last_label = None

    for t in range(seq_len):
        encoder_step = encoded_seq[t, :, :].unsqueeze(1)
        emitted = 0
        not_blank = True

        while not_blank and emitted < max_symbols:
            decoder_step, hidden = head.decoder.predict(last_label, dec_state)
            joint_logp = head.joint.joint(encoder_step, decoder_step)[0, 0, 0, :]
            k = int(torch.argmax(joint_logp).item())
            if k == blank_id:
                not_blank = False
                continue
            hyp.append(k)
            token_frames.append(t)
            dec_state = hidden
            last_label = torch.tensor([[k]], dtype=torch.long, device=encoded_seq.device)
            emitted += 1
    return hyp, token_frames


def token_to_str(tokenizer: Tokenizer, token_id: int) -> str:
    """
    Convert a token ID to its string representation.

    Uses character vocabulary for charwise tokenizers or SentencePiece
    IdToPiece for subword tokenizers.
    """
    if tokenizer.charwise:
        return tokenizer.vocab[token_id]
    return tokenizer.model.IdToPiece(token_id)


def chars_to_words(chars: Sequence[str], frames: Sequence[int], frame_shift: float) -> list[WordTimestamp]:
    """
    Collapse a sequence of character (or subword) tokens with frame indices into
    contiguous word segments, emitting absolute start/end times in seconds.
    """
    words: list[WordTimestamp] = []
    current_chars: list[str] = []
    current_frames: list[int] = []

    def commit():
        if not current_chars or not current_frames:
            return
        text = "".join(current_chars).strip()
        if not text:
            current_chars.clear()
            current_frames.clear()
            return
        start = current_frames[0] * frame_shift
        end = (current_frames[-1] + 1) * frame_shift
        words.append(WordTimestamp(text=text, start=start, end=end))
        current_chars.clear()
        current_frames.clear()

    for char, frame in zip(chars, frames, strict=True):
        if char.startswith("▁"):
            commit()
            char = char[1:]
        elif char == " ":
            commit()
            continue
        current_chars.append(char)
        current_frames.append(frame)

    commit()
    return words


def format_word_timestamps(words: list[WordTimestamp], offset: float = 0.0) -> list[WordTimestamp]:
    """
    Format word timestamps by applying offset and rounding.
    """
    return [
        WordTimestamp(
            text=word.text,
            start=round(word.start + offset, 3),
            end=round(word.end + offset, 3),
        )
        for word in words
    ]
