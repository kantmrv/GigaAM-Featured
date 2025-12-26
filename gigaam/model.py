import hydra
import omegaconf
import torch
from torch import Tensor, nn

from .audio_utils import SAMPLE_RATE, check_tensor, load_audio
from .onnx_utils import onnx_converter
from .types import TranscribedSegment, WordTimestamp

LONGFORM_THRESHOLD = 25 * SAMPLE_RATE


class GigaAM(nn.Module):
    """
    Giga Acoustic Model: Self-Supervised Model for Speech Tasks
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__()
        self.cfg = cfg
        self.preprocessor = hydra.utils.instantiate(self.cfg.preprocessor)
        self.encoder = hydra.utils.instantiate(self.cfg.encoder)
        self._cached_device: torch.device | None = None
        self._cached_dtype: torch.dtype | None = None

    def forward(self, features: Tensor, feature_lengths: Tensor) -> tuple[Tensor, Tensor]:
        """
        Perform forward pass through the preprocessor and encoder.
        """
        features, feature_lengths = self.preprocessor(features, feature_lengths)
        if self._device.type == "cpu":
            return self.encoder(features, feature_lengths)
        with torch.autocast(device_type=self._device.type, dtype=torch.float16):
            return self.encoder(features, feature_lengths)

    @property
    def _device(self) -> torch.device:
        if self._cached_device is None:
            self._cached_device = next(self.parameters()).device
        return self._cached_device

    @property
    def _dtype(self) -> torch.dtype:
        if self._cached_dtype is None:
            self._cached_dtype = next(self.parameters()).dtype
        return self._cached_dtype

    def to(self, *args, **kwargs):
        """Override to invalidate cache when device changes"""
        result = super().to(*args, **kwargs)
        self._cached_device = None
        self._cached_dtype = None
        return result

    def prepare_wav(self, wav: str | Tensor, sample_rate: int = SAMPLE_RATE) -> tuple[Tensor, Tensor]:
        """
        Prepare an audio file for processing by loading it onto
        the correct device and converting its format.
        """
        if isinstance(wav, str):
            wav = load_audio(wav)
        else:
            wav = check_tensor(wav, sample_rate)

        wav = wav.to(self._device).to(self._dtype).unsqueeze(0)
        length = torch.full([1], wav.shape[-1], device=self._device)

        return wav, length

    def embed_audio(self, wav: str | Tensor, sample_rate: int = SAMPLE_RATE) -> tuple[Tensor, Tensor]:
        """
        Extract audio representations using the GigaAM model.
        """
        wav, length = self.prepare_wav(wav, sample_rate)
        encoded, encoded_len = self.forward(wav, length)
        return encoded, encoded_len

    def to_onnx(self, dir_path: str = ".") -> None:
        """
        Export onnx model encoder to the specified dir.
        """
        self._to_onnx(dir_path)
        omegaconf.OmegaConf.save(self.cfg, f"{dir_path}/{self.cfg.model_name}.yaml")

    def _to_onnx(self, dir_path: str = ".") -> None:
        """
        Export onnx model encoder to the specified dir.
        """
        onnx_converter(
            model_name=f"{self.cfg.model_name}_encoder",
            out_dir=dir_path,
            module=self.encoder,
            dynamic_axes=self.encoder.dynamic_axes(),
        )


class GigaAMASR(GigaAM):
    """
    Giga Acoustic Model for Speech Recognition
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.head = hydra.utils.instantiate(self.cfg.head)
        self.decoding = hydra.utils.instantiate(self.cfg.decoding)

    def forward_for_export(self, features: Tensor, feature_lengths: Tensor) -> Tensor:
        """
        Encoder-decoder forward to save model entirely in onnx format.
        """
        return self.head(self.encoder(features, feature_lengths)[0])

    def _to_onnx(self, dir_path: str = ".") -> None:
        """
        Export onnx ASR model.
        `ctc`:  exported entirely in encoder-decoder format.
        `rnnt`: exported in encoder/decoder/joint parts separately.
        """
        if "ctc" in self.cfg.model_name:
            saved_forward = self.forward
            self.forward = self.forward_for_export  # type: ignore[assignment, method-assign]
            try:
                onnx_converter(
                    model_name=self.cfg.model_name,
                    out_dir=dir_path,
                    module=self,
                    inputs=self.encoder.input_example(),
                    input_names=["features", "feature_lengths"],
                    output_names=["log_probs"],
                    dynamic_axes={
                        "features": {0: "batch_size", 2: "seq_len"},
                        "feature_lengths": {0: "batch_size"},
                        "log_probs": {0: "batch_size", 1: "seq_len"},
                    },
                )
            finally:
                self.forward = saved_forward  # type: ignore[assignment, method-assign]
        else:
            super()._to_onnx(dir_path)
            onnx_converter(
                model_name=f"{self.cfg.model_name}_decoder",
                out_dir=dir_path,
                module=self.head.decoder,
            )
            onnx_converter(
                model_name=f"{self.cfg.model_name}_joint",
                out_dir=dir_path,
                module=self.head.joint,
            )

    def extract_word_timestamps(
        self,
        wav: Tensor,
        length: Tensor,
        sample_rate: int = SAMPLE_RATE,
        encoded: Tensor | None = None,
        encoded_len: Tensor | None = None,
    ) -> list[WordTimestamp]:
        """
        Run the model on a single waveform chunk and return word-level time spans.
        """
        from .timestamps_utils import (
            chars_to_words,
            decode_with_alignment_ctc,
            decode_with_alignment_rnnt,
            token_to_str,
        )

        if encoded is None or encoded_len is None:
            encoded, encoded_len = self.forward(wav, length)
        seq_len = int(encoded_len[0].item())
        if seq_len == 0:
            raise ValueError("Empty encoded sequence - cannot extract timestamps")
        frame_shift = length[0].item() / sample_rate / seq_len

        if hasattr(self.head, "decoder"):
            encoded_rnnt = encoded.transpose(1, 2)
            seq = encoded_rnnt[0, :, :].unsqueeze(1)
            max_symbols = getattr(self.decoding, "max_symbols", 10)
            token_ids, token_frames = decode_with_alignment_rnnt(
                self.head, seq, seq_len, self.decoding.blank_id, max_symbols
            )
        else:
            token_ids, token_frames = decode_with_alignment_ctc(self.head, encoded, seq_len, self.decoding.blank_id)

        chars = [token_to_str(self.decoding.tokenizer, idx) for idx in token_ids]
        return chars_to_words(chars, token_frames, frame_shift)

    @torch.inference_mode()
    def transcribe(
        self,
        wav: str | Tensor,
        sample_rate: int = SAMPLE_RATE,
        word_timestamps: bool = False,
    ) -> list[TranscribedSegment]:
        """
        Transcribe a short audio file or tensor into text.
        """
        wav, length = self.prepare_wav(wav, sample_rate)

        if length.item() > LONGFORM_THRESHOLD:
            raise ValueError("Too long wav file, use 'transcribe_longform' method.")

        encoded, encoded_len = self.forward(wav, length)
        transcription = self.decoding.decode(self.head, encoded, encoded_len)[0]

        words = None
        if word_timestamps:
            from .timestamps_utils import format_word_timestamps

            words = format_word_timestamps(self.extract_word_timestamps(wav, length, SAMPLE_RATE, encoded, encoded_len))

        return [TranscribedSegment(text=transcription, words=words)]

    @torch.inference_mode()
    def transcribe_longform(
        self,
        wav: str | Tensor,
        sample_rate: int = SAMPLE_RATE,
        word_timestamps: bool = False,
        checkpoint: str = "pyannote/segmentation-3.0",
        **kwargs,
    ) -> list[TranscribedSegment]:
        """
        Transcribe a long audio file or tensor by splitting into segments.
        """
        from .vad_utils import segment_audio_file

        segments, boundaries = segment_audio_file(
            wav, sample_rate, device=self._device, checkpoint=checkpoint, **kwargs
        )

        transcribed_segments: list[TranscribedSegment] = []
        for segment, segment_boundaries in zip(segments, boundaries, strict=True):
            segment_wav, length = self.prepare_wav(segment, SAMPLE_RATE)
            encoded, encoded_len = self.forward(segment_wav, length)
            transcription = self.decoding.decode(self.head, encoded, encoded_len)[0]

            words = None
            if word_timestamps:
                from .timestamps_utils import format_word_timestamps

                words = format_word_timestamps(
                    self.extract_word_timestamps(segment_wav, length, SAMPLE_RATE, encoded, encoded_len),
                    offset=segment_boundaries[0],
                )

            transcribed_segments.append(
                TranscribedSegment(
                    text=transcription,
                    start=round(segment_boundaries[0], 3),
                    end=round(segment_boundaries[1], 3),
                    words=words,
                )
            )

        return transcribed_segments


class GigaAMEmo(GigaAM):
    """
    Giga Acoustic Model for Emotion Recognition
    """

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)
        self.head = hydra.utils.instantiate(self.cfg.head)
        self.id2name = cfg.id2name

    def get_probs(
        self,
        wav: str | Tensor,
        sample_rate: int = SAMPLE_RATE,
    ) -> dict[str, float]:
        """
        Calculate probabilities for each emotion class based on the provided audio file.
        """
        wav, length = self.prepare_wav(wav, sample_rate)
        encoded, _ = self.forward(wav, length)
        if encoded.shape[-1] == 0:
            raise ValueError("Empty encoded sequence")
        encoded_pooled = nn.functional.avg_pool1d(encoded, kernel_size=encoded.shape[-1]).squeeze(-1)

        logits = self.head(encoded_pooled)[0]
        probs = nn.functional.softmax(logits, dim=-1).detach().tolist()

        return {self.id2name[i]: probs[i] for i in range(len(self.id2name))}

    def forward_for_export(self, features: Tensor, feature_lengths: Tensor) -> Tensor:
        """
        Encoder-decoder forward to save model entirely in onnx format.
        """
        encoded, _ = self.encoder(features, feature_lengths)
        enc_pooled = encoded.mean(dim=-1)
        return nn.functional.softmax(self.head(enc_pooled), dim=-1)

    def _to_onnx(self, dir_path: str = ".") -> None:
        """
        Export onnx Emo model.
        """
        saved_forward = self.forward
        self.forward = self.forward_for_export  # type: ignore[assignment, method-assign]
        try:
            onnx_converter(
                model_name=self.cfg.model_name,
                out_dir=dir_path,
                module=self,
                inputs=self.encoder.input_example(),
                input_names=["features", "feature_lengths"],
                output_names=["probs"],
                dynamic_axes={
                    "features": {0: "batch_size", 2: "seq_len"},
                    "feature_lengths": {0: "batch_size"},
                    "probs": {0: "batch_size", 1: "seq_len"},
                },
            )
        finally:
            self.forward = saved_forward  # type: ignore[assignment, method-assign]
