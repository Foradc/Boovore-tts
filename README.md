---
title: Boovore — Multi-Engine TTS Studio
emoji: 🎙
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🎙 Boovore — Multi-Engine TTS Studio

**Boovore** is a self-hosted, GPU-accelerated Text-to-Speech studio with 6 best-in-class engines and a built-in audiobook generator. Run it on any CUDA machine (tested on RTX 3090) via a clean, dark-mode web UI.

> **Name**: Boovore = *Book* + *Devour* — built to devour books in audio.

![Boovore UI](screenshot.png)

---

## ✨ Engines

| Engine | Quality | Speed | Highlights |
|---|---|---|---|
| **Kokoro FR** | ★★★★ | ⚡⚡⚡ | Native French voices |
| **Chatterbox** | ★★★★★ | ⚡⚡ | Zero-shot voice cloning (ResembleAI) |
| **F5-TTS** | ★★★★ | ⚡⚡ | French voice cloning |
| **Fish-Speech 1.5** | ★★★★★ | ⚡⚡ | Multilingual voice cloning (fishaudio) |
| **Qwen3-TTS** | ★★★★★ | ⚡ | Clone · Custom · Voice Design |
| **Voxtral 4B** | ★★★★★ | ⚡⚡ | French-first, 68% win vs ElevenLabs (Mistral AI) |

> **Voxtral** uses vLLM-Omni (`mistralai/Voxtral-4B-TTS-2603`) with voice cloning via a reference WAV. Start it separately with `python3 voxtral_server.py`.

---

## ⚙️ CPU / GPU — HuggingFace Space Settings

In your Space → **Settings → Variables and secrets**, set:

| `ENABLED_ENGINES` | Hardware | Engines available |
|---|---|---|
| `kokoro,f5` | CPU (free tier) | Kokoro · F5-TTS |
| `kokoro,f5,chatterbox` | GPU T4 (~6 GB) | + Chatterbox |
| `all` | GPU A10G / A100 | All engines + Qwen3 |

> Default is `all` — on free CPU tier, set `kokoro,f5` to avoid crashes.

For **Voxtral**, also set `VOXTRAL_URL` to point to your vLLM-Omni server (default: `http://localhost:8000`).

---

## 🚀 Quick Start (Vast.ai / GPU server)

### 1. Install dependencies

```bash
# PyTorch nightly CUDA 12.8 (required)
pip3 install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Register torch libs so torchaudio can find libtorch
echo "/usr/local/lib/python3.12/dist-packages/torch/lib" > /etc/ld.so.conf.d/torch.conf && ldconfig

# Core engines
pip3 install faster-qwen3-tts kokoro f5-tts fastapi uvicorn[standard] python-multipart

# Chatterbox (Python 3.12 fix)
pip3 install conformer==0.3.2 --no-build-isolation
git clone https://github.com/resemble-ai/chatterbox /tmp/chatterbox
cd /tmp/chatterbox && pip3 install -e . --no-deps && cd /root

# Fish-Speech 1.5
git clone https://github.com/fishaudio/fish-speech /tmp/fish-speech
cd /tmp/fish-speech && git checkout v1.5.1
pip3 install -e . --no-deps
huggingface-cli download fishaudio/fish-speech-1.5 --local-dir /root/fish-speech-model
```

### 2. (Optional) Start Voxtral TTS server

Voxtral requires a separate vLLM-Omni process (~8 GB VRAM). Needs a HuggingFace token — accept the CC BY-NC license at [mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) first.

```bash
pip install "vllm[audio]>=0.18.0" httpx soundfile
export HF_TOKEN=hf_xxxx
nohup python3 voxtral_server.py >> /root/voxtral.log 2>&1 &
# Wait 5-10 min for model download + load (first run only)
```

Optionally generate a narrator reference WAV (for voice cloning):

```bash
# While the Qwen3 server is running:
python3 make_narrator_reference.py
# Output: /workspace/narrator_reference.wav
```

### 3. Start the main server

```bash
nohup python3 server.py --port 7860 >> /root/server.log 2>&1 &
```

### 3. Open the UI

```bash
# Local SSH tunnel
ssh -p <PORT> root@<HOST> -L 7860:localhost:7860 -N
# Then open http://localhost:7860
```

---

## 📖 Features

- **TTS Studio** — one-click engine selector (8 pills), single generate button
- **Audiobook Generator** — import `.txt` / `.pdf` / `.epub`, auto-detect chapters, batch generate with any engine, download per chapter or merge into one WAV
- **Voice Cloning** — upload a reference audio clip (Chatterbox, F5-TTS, Fish-Speech, Qwen3)
- **Real-time metrics** — TTFA, RTF, duration, buffer
- **Light / dark theme**
- **Streaming audio** (Qwen3) with CUDA Graphs

---

## 🗂 Project Structure

```
server.py                    — FastAPI backend (6 engines)
index.html                   — UI single-page (vanilla JS, no frontend deps)
voxtral_server.py            — vLLM-Omni server manager (start/stop/status)
make_narrator_reference.py   — Generate narrator reference WAV via Qwen3
narrator_reference.wav       — (generated) voice clone reference for Voxtral
requirements.txt
Dockerfile
```

---

## ⚙️ Requirements

- Python 3.12+
- CUDA 12.8 (RTX 3090 or better recommended)
- PyTorch nightly cu128 (`2.12.0.dev+`)
- VRAM: 8 GB minimum, 24 GB to run all engines simultaneously

---

## 📦 Models (auto-downloaded)

| Modèle | Taille | Engine |
|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | ~1.2 GB | Qwen3-TTS |
| `hexgrad/Kokoro-82M` | ~300 MB | Kokoro FR |
| `SWivid/F5-TTS` | ~1.2 GB | F5-TTS |
| `resemble-ai/chatterbox` | ~1.5 GB | Chatterbox |
| `fishaudio/fish-speech-1.5` | ~1.4 GB | Fish-Speech |
| `mistralai/Voxtral-4B-TTS-2603` | ~8 GB (BF16) | Voxtral (gated — HF token required) |

---

## 🏷️ GitHub Topics

`text-to-speech` `tts` `voice-cloning` `audiobook` `french-tts` `kokoro` `f5-tts` `fish-speech` `chatterbox` `qwen3` `voxtral` `mistral` `vllm` `fastapi` `cuda` `self-hosted` `gpu` `french` `multilingual`

---

## Credits

- [faster-qwen3-tts](https://github.com/huggingfaceM4/faster-qwen3-tts) — Qwen3-TTS engine
- [Fish-Speech](https://github.com/fishaudio/fish-speech) — fishaudio
- [Chatterbox](https://github.com/resemble-ai/chatterbox) — ResembleAI
- [F5-TTS](https://github.com/SWivid/F5-TTS) — SWivid
- [Kokoro](https://github.com/hexgrad/kokoro) — hexgrad
- [Voxtral](https://mistral.ai) — Mistral AI (`mistralai/Voxtral-4B-TTS-2603`, CC BY-NC)
- French prosody preprocessing inspired by [arXiv:2508.17494](https://arxiv.org/abs/2508.17494)

---

MIT License
