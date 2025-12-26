import warnings
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import onnxruntime as rt
import torch
from torch import Tensor
from torch.jit import TracerWarning  # type: ignore[attr-defined]

from .audio_utils import SAMPLE_RATE, check_tensor, load_audio
from .decoding import Tokenizer
from .preprocess import FeatureExtractor
from .timestamps_utils import chars_to_words, token_to_str
from .types import TranscribedSegment, WordTimestamp

warnings.simplefilter("ignore", category=UserWarning)


DTYPE = np.float32
MAX_LETTERS_PER_FRAME = 3


def onnx_converter(
    model_name: str,
    module: torch.nn.Module,
    out_dir: str,
    inputs: tuple[Tensor, ...] | None = None,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict[str, list[int]] | dict[str, dict[int, str]] | None = None,
    opset_version: int = 17,
) -> None:
    """
    Export a PyTorch module to ONNX format.
    """
    if inputs is None:
        inputs = module.input_example()  # type: ignore[operator]
    if input_names is None:
        input_names = module.input_names()  # type: ignore[operator]
    if output_names is None:
        output_names = module.output_names()  # type: ignore[operator]

    Path(out_dir).mkdir(exist_ok=True, parents=True)
    out_path = str(Path(out_dir) / f"{model_name}.onnx")
    saved_dtype = next(module.parameters()).dtype
    with warnings.catch_warnings(), torch.no_grad():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=TracerWarning)
        torch.onnx.export(
            module.to(torch.float32),
            inputs,  # type: ignore[arg-type]
            out_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
        )
    print(f"Successfully ported onnx {model_name} to {out_path}.")
    module.to(saved_dtype)


def infer_onnx(
    wav: str | torch.Tensor,
    model_cfg: omegaconf.DictConfig,
    sessions: list[rt.InferenceSession],
    sample_rate: int = SAMPLE_RATE,
    preprocessor: FeatureExtractor | None = None,
    tokenizer: Tokenizer | None = None,
    word_timestamps: bool = False,
) -> tuple[np.ndarray, np.ndarray] | dict[str, float] | str | TranscribedSegment:
    """
    Run ONNX inference on audio input.
    """
    model_name = model_cfg.model_name

    if preprocessor is None:
        preprocessor = hydra.utils.instantiate(model_cfg.preprocessor)
    if tokenizer is None and ("ctc" in model_name or "rnnt" in model_name):
        tokenizer = hydra.utils.instantiate(model_cfg.decoding).tokenizer

    if isinstance(wav, str):
        input_signal = load_audio(wav)
    else:
        input_signal = check_tensor(wav, sample_rate)

    original_audio_len = input_signal.shape[-1]

    input_signal = preprocessor(  # type: ignore[misc]
        input_signal.unsqueeze(0), torch.tensor([original_audio_len])
    )[0].numpy()

    enc_sess = sessions[0]
    enc_inputs = {
        node.name: data
        for (node, data) in zip(
            enc_sess.get_inputs(), [input_signal.astype(DTYPE), [input_signal.shape[-1]]], strict=True
        )
    }
    enc_features = enc_sess.run([node.name for node in enc_sess.get_outputs()], enc_inputs)[0]

    if "ssl" in model_name:
        encoded_len = np.array([enc_features.shape[-1]], dtype=np.int64)  # type: ignore[union-attr]
        return (enc_features, encoded_len)  # type: ignore[return-value]

    if "emo" in model_name:
        id2name = model_cfg.id2name
        probs = enc_features[0]  # type: ignore[index]
        return {id2name[i]: float(probs[i]) for i in range(len(id2name))}  # type: ignore[return-value]

    blank_idx = len(tokenizer)  # type: ignore[arg-type]
    token_ids = []
    token_frames: list[int] = []
    prev_token = blank_idx

    if "ctc" in model_name:
        frames_list = enc_features.argmax(-1).squeeze().tolist()  # type: ignore[attr-defined, union-attr]
        for t, tok in enumerate(frames_list):
            if (tok != prev_token or prev_token == blank_idx) and tok != blank_idx:
                token_ids.append(tok)
                token_frames.append(t)
            prev_token = tok
    else:
        pred_states = [
            np.zeros(shape=(1, 1, model_cfg.head.decoder.pred_hidden), dtype=DTYPE),
            np.zeros(shape=(1, 1, model_cfg.head.decoder.pred_hidden), dtype=DTYPE),
        ]
        pred_sess, joint_sess = sessions[1:]
        for j in range(enc_features.shape[-1]):  # type: ignore[attr-defined, union-attr]
            emitted_letters = 0
            while emitted_letters < MAX_LETTERS_PER_FRAME:
                pred_inputs = {
                    node.name: data
                    for (node, data) in zip(
                        pred_sess.get_inputs(),
                        [np.array([[prev_token]])] + pred_states,  # type: ignore[operator]
                        strict=True,
                    )
                }
                pred_outputs = pred_sess.run([node.name for node in pred_sess.get_outputs()], pred_inputs)

                joint_inputs = {
                    node.name: data
                    for node, data in zip(
                        joint_sess.get_inputs(),
                        [enc_features[:, :, [j]], pred_outputs[0].swapaxes(1, 2)],  # type: ignore[index, union-attr, operator]
                        strict=True,
                    )
                }
                log_probs = joint_sess.run([node.name for node in joint_sess.get_outputs()], joint_inputs)
                token = log_probs[0].argmax(-1)[0][0]  # type: ignore[attr-defined, union-attr]

                if token != blank_idx:
                    prev_token = int(token)
                    pred_states = pred_outputs[1:]
                    token_ids.append(int(token))
                    token_frames.append(j)
                    emitted_letters += 1
                else:
                    break

    if word_timestamps:
        audio_duration = original_audio_len / sample_rate
        frame_shift = audio_duration / enc_features.shape[-1]  # type: ignore[attr-defined, union-attr]

        chars = [token_to_str(tokenizer, idx) for idx in token_ids]  # type: ignore[arg-type]
        words = chars_to_words(chars, token_frames, frame_shift)
        words_rounded = [WordTimestamp(text=w.text, start=round(w.start, 3), end=round(w.end, 3)) for w in words]

        return TranscribedSegment(text=tokenizer.decode(token_ids), words=words_rounded)  # type: ignore[union-attr]

    return tokenizer.decode(token_ids)  # type: ignore[union-attr]


def load_onnx(
    onnx_dir: str,
    model_version: str,
    provider: str | None = None,
) -> tuple[list[rt.InferenceSession], omegaconf.DictConfig | omegaconf.ListConfig]:
    """Load ONNX sessions for the given versions and cpu / cuda provider"""
    if provider is None and "CUDAExecutionProvider" in rt.get_available_providers():
        provider = "CUDAExecutionProvider"
    elif provider is None:
        provider = "CPUExecutionProvider"

    opts = rt.SessionOptions()
    opts.intra_op_num_threads = 16
    opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    opts.log_severity_level = 3

    config_path = f"{onnx_dir}/{model_version}.yaml"
    try:
        model_cfg = omegaconf.OmegaConf.load(config_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"ONNX config not found: {config_path}") from e
    except Exception as e:
        raise ValueError(f"Invalid ONNX config file: {e}") from e

    if "rnnt" not in model_version and "ssl" not in model_version:
        model_path = f"{onnx_dir}/{model_version}.onnx"
        sessions = [rt.InferenceSession(model_path, providers=[provider], sess_options=opts)]
    elif "ssl" in model_version:
        pth = f"{onnx_dir}/{model_version}"
        enc_sess = rt.InferenceSession(f"{pth}_encoder.onnx", providers=[provider], sess_options=opts)
        sessions = [enc_sess]
    else:
        pth = f"{onnx_dir}/{model_version}"
        enc_sess = rt.InferenceSession(f"{pth}_encoder.onnx", providers=[provider], sess_options=opts)
        pred_sess = rt.InferenceSession(f"{pth}_decoder.onnx", providers=[provider], sess_options=opts)
        joint_sess = rt.InferenceSession(f"{pth}_joint.onnx", providers=[provider], sess_options=opts)
        sessions = [enc_sess, pred_sess, joint_sess]

    return sessions, model_cfg


def infer_onnx_longform(
    wav: str | torch.Tensor,
    model_cfg: omegaconf.DictConfig,
    sessions: list[rt.InferenceSession],
    sample_rate: int = SAMPLE_RATE,
    word_timestamps: bool = False,
    checkpoint: str = "pyannote/segmentation-3.0",
    device: torch.device | None = None,
    **kwargs,
) -> list[TranscribedSegment]:
    """
    Transcribe long audio using ONNX model with VAD-based segmentation.
    """
    from .vad_utils import segment_audio_file

    if device is None:
        device = torch.device("cpu")

    segments, boundaries = segment_audio_file(wav, sample_rate, device=device, checkpoint=checkpoint, **kwargs)

    preprocessor = hydra.utils.instantiate(model_cfg.preprocessor)
    tokenizer = None
    if "ctc" in model_cfg.model_name or "rnnt" in model_cfg.model_name:
        tokenizer = hydra.utils.instantiate(model_cfg.decoding).tokenizer

    transcribed_segments: list[TranscribedSegment] = []

    for segment, segment_boundaries in zip(segments, boundaries, strict=True):
        result = infer_onnx(
            segment,
            model_cfg,
            sessions,
            sample_rate=sample_rate,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            word_timestamps=word_timestamps,
        )

        if word_timestamps:
            assert isinstance(result, TranscribedSegment)
            adjusted_words = [
                WordTimestamp(
                    text=w.text,
                    start=round(w.start + segment_boundaries[0], 3),
                    end=round(w.end + segment_boundaries[0], 3),
                )
                for w in result.words or []
            ]
            transcribed_segments.append(
                TranscribedSegment(
                    text=result.text,
                    start=round(segment_boundaries[0], 3),
                    end=round(segment_boundaries[1], 3),
                    words=adjusted_words,
                )
            )
        else:
            assert isinstance(result, str)
            transcribed_segments.append(
                TranscribedSegment(
                    text=result,
                    start=round(segment_boundaries[0], 3),
                    end=round(segment_boundaries[1], 3),
                )
            )

    return transcribed_segments
