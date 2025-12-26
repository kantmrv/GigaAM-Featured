# GigaAM: the family of open-source acoustic models for speech processing

<div align="center" style="line-height: 1;">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2506.01192-b31b1b.svg)](https://arxiv.org/abs/2506.01192)

</div>

<hr>
    
> [!Note]
> This repository is a featured version of the original [GigaAM repository](https://github.com/salute-developers/GigaAM.git) with enhanced features:
> - **Tensor Input Support**: Use PyTorch tensors directly as input audio with automatic resampling via `sample_rate` parameter
> - **Word-Level Timestamps**: Get word-level timestamps with the `word_timestamps` parameter in `transcribe` and `transcribe_longform` methods
> - **Flexible VAD Checkpoints**: Configure PyAnnote segmentation checkpoints for long-form transcription via `checkpoint` parameter

---

## Setup

### Requirements
- Python ≥ 3.10
- [ffmpeg](https://ffmpeg.org/) installed and added to your system's PATH

### Install the GigaAM Package

```bash
# Clone the repository
git clone https://github.com/salute-developers/GigaAM.git
cd GigaAM

# Install the package requirements
uv venv && uv sync

# (optionally) Verify the installation:
uv sync --extra tests
uv run pytest -v tests/test_loading.py -m partial  # or `-m full` to test all models
```

---

## GigaAM overview

GigaAM is a [Conformer](https://arxiv.org/pdf/2005.08100.pdf)-based foundational model (220-240M parameters) pre-trained on diverse Russian speech data. It serves as the backbone for the entire GigaAM family, enabling state-of-the-art fine-tuned performance in speech recognition and emotion recognition. More information about GigaAM-v1 can be found in our [post on Habr](https://habr.com/ru/companies/sberdevices/articles/805569). We fine-tuned the GigaAM encoder for ASR using [CTC](https://www.cs.toronto.edu/~graves/icml_2006.pdf) and [RNNT](https://arxiv.org/abs/1211.3711) decoders. GigaAM family includes three lines of models

| | Pretrain Method | Pretrain (hours) | ASR (hours) | Available Versions |
| :--- | :--- | :--- | :--- | :---: |
| **v1** | [Wav2vec 2.0](https://arxiv.org/abs/2006.11477) | 50,000 | 2,000 | `v1_ssl`, `emo`, `v1_ctc`, `v1_rnnt` |
| **v2** | [HuBERT–CTC](https://arxiv.org/abs/2506.01192) | 50,000 | 2,000 | `v2_ssl`, `v2_ctc`, `v2_rnnt` |
| **v3** | HuBERT–CTC | 700,000 | 4,000 | `v3_ssl`, `v3_ctc`, `v3_rnnt`, `v3_e2e_ctc`, `v3_e2e_rnnt` |

Where `v3_e2e_ctc` and `v3_e2e_rnnt` support punctuation and text normalization.

---

## Usage

### Model inference

**Note:** ASR with `.transcribe` function is applicable for audio **only up to 25 seconds**. To enable `.transcribe_longform` install the additional [pyannote.audio](https://github.com/pyannote/pyannote-audio) dependencies

<details>
<summary>Longform setup instruction</summary>

* Generate [Hugging Face API token](https://huggingface.co/docs/hub/security-tokens)
* Accept the conditions to access [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) files and content

```bash
uv sync --extra longform
# optionally run longform testing
uv sync --all-extras
HF_TOKEN=<your hf token> uv run pytest -v tests/test_longform.py
```
</details>

<br>


```python
import gigaam
import urllib.request

# Download sample audio files
short_audio_url = "https://n-ws-q0bez.s3pd12.sbercloud.ru/b-ws-q0bez-jpv/GigaAM/audio/example_0.wav"
long_audio_url = "https://n-ws-q0bez.s3pd12.sbercloud.ru/b-ws-q0bez-jpv/GigaAM/audio/example_3.wav"

urllib.request.urlretrieve(short_audio_url, "example_short.wav")
urllib.request.urlretrieve(long_audio_url, "example_long.wav")

audio_path = "example_short.wav"
long_audio_path = "example_long.wav"

# Audio embeddings
model_name = "v3_ssl"       # Options: `v1_ssl`, `v2_ssl`, `v3_ssl`
model = gigaam.load_model(model_name)
embedding, _ = model.embed_audio(audio_path)
print(embedding)

# ASR
model_name = "v3_e2e_rnnt"  # Options: any model version with suffix `_ctc` or `_rnnt`
model = gigaam.load_model(model_name)
results = model.transcribe(audio_path)
print(results[0].text)

# ASR with tensor input and optional word timestamps
import torch
import torchaudio

# Load audio as tensor
wav_tensor, sr = torchaudio.load(audio_path)
wav_tensor = wav_tensor.squeeze(0)  # Convert to 1D tensor

# Transcribe with automatic resampling and word-level timestamps
results = model.transcribe(wav_tensor, sample_rate=sr, word_timestamps=True)
for segment in results:
    print(f"Full text: {segment.text}")
    if segment.words:
        for word in segment.words:
            print(f"  [{word.start:.3f} - {word.end:.3f}s]: {word.text}")

# Long-form ASR with tensor input
import os
os.environ["HF_TOKEN"] = "<your HF_TOKEN>"  # with read access to "pyannote/segmentation-3.0"

# With file path
utterances = model.transcribe_longform(long_audio_path)
for utt in utterances:
   transcription = utt.text
   start, end = utt.start, utt.end
   print(f"[{gigaam.format_time(start)} - {gigaam.format_time(end)}]: {transcription}")

# With tensor input and word-level timestamps
long_wav_tensor, long_sr = torchaudio.load(long_audio_path)
long_wav_tensor = long_wav_tensor.squeeze(0)

results = model.transcribe_longform(
    long_wav_tensor,
    sample_rate=long_sr,
    word_timestamps=True
)
for segment in results:
    print(f"[{segment.start:.3f} - {segment.end:.3f}s]: {segment.text}")
    if segment.words:
        for word in segment.words:
            print(f"  [{word.start:.3f} - {word.end:.3f}s]: {word.text}")

# Emotion recognition
model = gigaam.load_model("emo")
emotion2prob = model.get_probs(audio_path)
print(", ".join([f"{emotion}: {prob:.3f}" for emotion, prob in emotion2prob.items()]))
```

### ONNX Export and Inference

> **Note:** GPU support can be enabled with `pip install onnxruntime-gpu==1.23.*` if applicable.

1. Export the model to ONNX using the `model.to_onnx` method:
   ```python
   onnx_dir = "onnx"
   model_version = "v3_ctc"  # Options: any version

   model = gigaam.load_model(model_version)
   model.to_onnx(dir_path=onnx_dir)
   ```

2. Run ONNX inference:
   ```python
   from gigaam.onnx_utils import load_onnx, infer_onnx

   sessions, model_cfg = load_onnx(onnx_dir, model_version)
   result = infer_onnx(audio_path, model_cfg, sessions)
   print(result)  # string for ctc / rnnt, np.ndarray for ssl / emo
   ```
