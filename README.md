## Whisper Tools: Real-time Transcriber and Subtitle Generator

Small toolkit built around OpenAI Whisper for:

- Real-time transcription of system audio or microphone input
- Offline transcription of audio files with `.txt` and `.srt` outputs
- Extracting audio from video using ffmpeg

### Contents

- `realtime_transcriber.py`: Real-time transcription from system audio (macOS virtual device) or mic
- `main.py`: Transcribe an audio file to plain text and SRT subtitles
- `extract.py`: Extract audio from a video file using ffmpeg

## Requirements

- Python 3.10+
- OS packages:
  - ffmpeg (required for `extract.py`)
  - PortAudio (used by `sounddevice`; typically bundled on macOS; installable via package manager elsewhere)
  - For system audio capture on macOS: a virtual audio device such as [BlackHole](https://github.com/ExistentialAudio/BlackHole) or Soundflower

Python dependencies (see `requirements.txt`):

- openai-whisper
- torch
- sounddevice
- numpy
- librosa
- scikit-learn
- colorama

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- ffmpeg is required for `extract.py`. On macOS (Homebrew): `brew install ffmpeg`. On Linux: install via your distro. On Windows: install ffmpeg and add it to PATH.
- Apple Silicon (M1/M2/M3): Whisper can run with Metal (MPS) acceleration via PyTorch. This script auto-selects `mps` if available. If MPS is unavailable, it will fall back to CPU.

## Usage

### 1) Real-time transcription (system audio or mic)

`realtime_transcriber.py` streams audio into Whisper, prints a live hypothesis, and replaces it with a refined line as more context arrives.

Basic (auto device selection, prefer system audio on macOS if a virtual device is detected):

```bash
python realtime_transcriber.py
```

Use the microphone instead of system audio:

```bash
python realtime_transcriber.py --use-mic
```

Choose model and device:

```bash
python realtime_transcriber.py --model small --device mps
```

Tune chunking and refinement:

```bash
python realtime_transcriber.py \
  --chunk-duration 3.0 \
  --sample-rate 16000 \
  --refinement-chunks 3
```

Options:

- `--model {tiny,base,small,medium,large,large-v2,large-v3}`: Whisper model (default: `base`)
- `--device {cpu,cuda,mps}`: Force device (default: auto-select; prefers MPS on macOS, then CUDA, then CPU)
- `--chunk-duration FLOAT`: Per-inference audio window in seconds (default: `3.0`)
- `--sample-rate INT`: Input sample rate (default: `16000`)
- `--refinement-chunks INT`: Rolling window of recent chunks to re-transcribe for refinement (default: `3`)
- `--use-mic`: Use microphone instead of system audio

Behavior details:

- The script prints a single "(live)" line immediately from the current chunk, then re-transcribes a rolling window of recent chunks to refine the text. If the refined text modifies prior words, the printed line is prefixed with `Refined:`.
- FP16 is enabled automatically on `cuda` and `mps` for speed.
- A simple volume gate ignores near-silent chunks (amplitude threshold ~0.01).
- Stop with Ctrl+C.

macOS system audio capture:

- Install a virtual audio device such as [BlackHole](https://github.com/ExistentialAudio/BlackHole) and set it as the system output (or use an aggregate device). The script looks for device names containing "blackhole", "soundflower", "loopback", or "virtual" and will use it automatically if found.
- If no virtual device is found, the script falls back to the default input device (mic). Use `--use-mic` to explicitly choose the microphone.

### 2) Transcribe an audio file to text and SRT

`main.py` loads an audio file, runs Whisper once, and writes both `.txt` and `.srt` files. SRT timestamps are formatted as `HH:MM:SS,mmm` with hours always present.

Examples:

```bash
python main.py path/to/audio.wav
```

Specify options:

```bash
python main.py path/to/audio.wav \
  --model small \
  --language en \
  --device cpu \
  --out-base ./outputs/my_transcript
```

Options:

- `audio`: Path to input audio file
- `--model {tiny,base,small,medium,large,large-v2,large-v3}` (default: `small`)
- `--language CODE`: e.g., `en`, `vi`; auto-detect if omitted
- `--device {cpu,cuda}`: Auto if omitted
- `--out-base PATH`: Output base path without extension; default is alongside the input

Outputs:

- `<out-base>.txt`: Full transcript text
- `<out-base>.srt`: SubRip subtitles with segment boundaries from Whisper

### 3) Extract audio from a video file

`extract.py` is a thin wrapper around ffmpeg to extract audio suitable for Whisper.

Examples:

```bash
python extract.py path/to/video.mov
```

Specify options:

```bash
python extract.py path/to/video.mov \
  --output ./audio.wav \
  --ext wav \
  --sample-rate 16000 \
  --channels 1 \
  --codec pcm_s16le \
  --overwrite
```

Options:

- `input`: Path to input video file
- `--output PATH`: Optional explicit output path; default is `<input-stem>.<ext>`
- `--ext {wav,flac,mp3,m4a}`: Output format (default: `wav`)
- `--sample-rate INT`: Default `16000`
- `--channels {1,2}`: Default `1`
- `--codec STR`: Default `pcm_s16le` for WAV
- `--overwrite`: Allow replacing existing output

## Tips and Troubleshooting

- No virtual audio device found (macOS): Install [BlackHole](https://github.com/ExistentialAudio/BlackHole) or Soundflower. After installation, set it as your output (or create an aggregate device with your normal output).
- Microphone or audio permissions (macOS): Grant Terminal or your IDE microphone access in System Settings → Privacy & Security → Microphone.
- `sounddevice` errors: Ensure PortAudio is installed/available. On macOS, it is typically present; on Linux, install `portaudio` via your package manager.
- ffmpeg not found: Install ffmpeg and ensure it is on your PATH. macOS (Homebrew): `brew install ffmpeg`.
- Performance:
  - Use smaller Whisper models (`tiny`/`base`/`small`) for low-latency real-time use
  - Prefer GPU (`cuda`) or Apple Silicon (`mps`) where available
  - Reduce `--chunk-duration` and/or `--refinement-chunks` to lower latency

## Acknowledgments

- Built on top of OpenAI Whisper

# Whisper Tools

Utilities for transcribing audio and video using OpenAI Whisper, including:

- Real-time transcription from system audio or microphone with optional speaker detection (`realtime_transcriber.py`).
- Batch transcription from an audio file to `.srt` subtitles and `.txt` transcript (`main.py`).
- Audio extraction from video via `ffmpeg` (`extract.py`).

## Requirements

- Python 3.10+
- `ffmpeg` (required for `extract.py`; recommended for media handling)
- For real-time capture:
  - macOS: A virtual audio device (e.g., BlackHole, Soundflower, or Loopback) if you want to capture system audio instead of microphone
  - PortAudio (backend for `sounddevice`)

Install system dependencies (examples):

```bash
# macOS (Homebrew)
brew install ffmpeg portaudio
# Optional for system-audio capture
brew install blackhole-2ch  # or install Soundflower/Loopback
```

## Python setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` includes: `openai-whisper`, `torch`, `numpy`, `sounddevice`, `colorama`, `scikit-learn`, `librosa`.

Note: Torch wheel selection depends on your platform. On Apple Silicon, the tools auto-select Metal (MPS) if available. On NVIDIA GPUs, CUDA is used if available.

## Scripts

### 1) Real-time Transcription (`realtime_transcriber.py`)
Real-time transcription of audio with optional lightweight speaker detection (unsupervised clustering over MFCC features). The script:

- Captures audio from a virtual system device (e.g., BlackHole) by default, or from the microphone with `--use-mic`.
- Streams audio into chunks (default 3s), prints a live line first, then a refined line after aggregating a short history of chunks.
- Detects speakers using MFCC features + `KMeans` over a sliding window, color-coding each inferred speaker.
- Auto-selects device: MPS (Apple Silicon) > CUDA > CPU. Uses FP16 on CUDA/MPS.

Usage examples:

```bash
# System audio (default) with auto device and base model
python realtime_transcriber.py

# Use microphone as input
python realtime_transcriber.py --use-mic

# Choose model, device, chunking, and speaker detection
python realtime_transcriber.py \
  --model small \
  --device mps \
  --chunk-duration 2.5 \
  --sample-rate 16000 \
  --refinement-chunks 3 \
  --max-speakers 4

# Disable speaker detection
python realtime_transcriber.py --disable-speaker-detection
```

Key options:

- `--model`: `tiny|base|small|medium|large|large-v2|large-v3` (default: `base`)
- `--device`: `cpu|cuda|mps` (default: auto)
- `--chunk-duration`: seconds per chunk (default: 3.0)
- `--sample-rate`: input sample rate (default: 16000)
- `--refinement-chunks`: number of recent chunks used to refine text (default: 3)
- `--use-mic`: use microphone input instead of system audio
- `--disable-speaker-detection`: single speaker mode
- `--max-speakers`: maximum distinct speakers to colorize (default: 4)

Notes:

- Speaker detection is heuristic and unsupervised; it may fluctuate early on and stabilizes as more audio is observed.
- For system audio on macOS, ensure a virtual audio device is installed and selected (script auto-picks devices with names like “BlackHole”, “Soundflower”, or “Loopback”).

### 2) File Transcription to SRT/TXT (`main.py`)
Transcribes an audio file and writes both a plain-text transcript and an `.srt` subtitle file.

```bash
python main.py /path/to/audio.wav \
  --model small \
  --language en \
  --device cuda        # or cpu
```

Outputs:

- `/path/to/audio.txt` — full transcript
- `/path/to/audio.srt` — subtitle file (timestamps formatted HH:MM:SS,mmm)

### 3) Extract Audio from Video (`extract.py`)
Extracts audio from a video using `ffmpeg`, with configurable format, sample rate, channels, and codec.

```bash
python extract.py /path/to/video.mov \
  --output /path/to/output.wav \
  --ext wav \
  --sample-rate 16000 \
  --channels 1 \
  --codec pcm_s16le \
  --overwrite
```

If `--output` is omitted, the output path defaults to the input stem with the chosen extension.

## Troubleshooting

- `ffmpeg not found`: Install `ffmpeg` and ensure it’s on your PATH.
- `sounddevice`/PortAudio errors: Install PortAudio (`brew install portaudio`) and reinstall `sounddevice` if needed.
- No system audio captured on macOS: Install and select a virtual audio device like BlackHole. The script auto-selects if detected.
- Poor real-time accuracy: Try a larger model (`small`/`medium`), increase `--refinement-chunks`, or ensure clean audio.
- High CPU/GPU usage: Reduce model size, increase `--chunk-duration`, or run on GPU/MPS.

## License

This repository is provided as-is, without warranty. Whisper’s license and any third-party licenses apply to their respective packages.

