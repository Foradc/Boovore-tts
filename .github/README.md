<p align="center">
  <img src="banner.png" alt="Boovore — six open-source TTS engines in one studio" width="100%">
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/terrancemoon9/Boovore_Multi-Engine-TTS-Studio"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Live%20demo-Hugging%20Face%20Space-ffcc4d" alt="Live demo on Hugging Face"></a>
  <a href="../LICENSE"><img src="https://img.shields.io/badge/license-MIT-8b5cf6" alt="MIT license"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-3776ab?logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/FastAPI-backend-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/github/last-commit/Foradc/Boovore-tts?color=3b82f6" alt="Last commit">
</p>

<p align="center">
  <b>English</b> · <a href="../README.fr.md">Français</a>
</p>

**Boovore** is a self-hosted text-to-speech studio. It puts six open-source TTS engines behind one web page, so you can hear the same sentence in every engine, clone a voice from a short clip, and turn a `.txt`, `.pdf` or `.epub` into a chaptered audiobook.

It is French-first (good French voices are hard to find in open-source TTS) and runs on whatever you have: a free CPU Space, a 6 GB laptop GPU, or a 24 GB server.

> **The name:** Boovore = *book* + *-vore* (devourer). It was built to devour books into audio.

<p align="center">
  <img src="../screenshot.png" alt="Boovore studio: text box, engine picker, voice cloning controls and audio player" width="85%">
</p>

## Why Boovore?

- **Compare engines in one click.** Same text, same page, six engines. You find the right voice by listening instead of installing six repos.
- **Books, not just sentences.** Import a `.txt`, `.pdf` or `.epub`, let Boovore detect the chapters, generate them in batch, then download each chapter or merge everything into one WAV.
- **Voice cloning and voice design.** Clone a voice from a few seconds of audio (Chatterbox, F5-TTS, Fish-Speech, Qwen3, Voxtral), or describe a voice in words and let Qwen3 Voice Design create it.
- **Runs on modest hardware.** The `ENABLED_ENGINES` switch loads only what your machine can hold. Qwen3 runs on a 6 GB RTX 2060 laptop.
- **One plain web page.** The UI is a single HTML file in vanilla JS with no build step, and the backend is a single FastAPI file with a documented HTTP API.

## Engines

| Engine | Good at | Voice cloning | Weights license | Commercial use |
|---|---|:---:|---|:---:|
| **[Kokoro 82M](https://huggingface.co/hexgrad/Kokoro-82M)** | Fast, natural French voice (`ff_siwis`) plus English voices; runs on CPU | — | Apache-2.0 | ✅ |
| **[Chatterbox](https://github.com/resemble-ai/chatterbox)** (Resemble AI) | Expressive zero-shot cloning with an emotion slider (English model) | ✅ | MIT | ✅ |
| **[F5-TTS FR](https://huggingface.co/RASPIAUDIO/F5-French-MixedSpeakers-reduced)** | French voice cloning; runs on CPU | ✅ | CC-BY-NC-4.0 | ❌ |
| **[Fish-Speech 1.5](https://github.com/fishaudio/fish-speech)** | Multilingual cloning | ✅ | CC-BY-NC-SA-4.0 | ❌ |
| **[Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)** 0.6B / 1.7B | Three modes (clone, built-in speakers, **Voice Design** from a text description), streaming with CUDA graphs | ✅ | Apache-2.0 | ✅ |
| **[Voxtral 4B](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603)** (Mistral AI) | French-first cloning. In Mistral's own human evaluation, listeners preferred it over ElevenLabs Flash v2.5 in [68.4% of zero-shot cloning comparisons](https://mistral.ai/news/voxtral-tts/) | ✅ | CC-BY-NC-4.0 | ❌ |

Boovore's own code is MIT. Each engine's **model weights** keep their own license (column above). Check it before you publish or sell audio.

## Quick start

### 1. Try it online

Open the **[Hugging Face Space](https://huggingface.co/spaces/terrancemoon9/Boovore_Multi-Engine-TTS-Studio)**. It runs Kokoro on the free CPU tier. The Space sleeps when idle, so the first visit can take about a minute. To try every engine, run Boovore on a GPU (below).

### 2. Docker (CPU: Kokoro + F5-TTS)

```bash
git clone https://github.com/Foradc/Boovore-tts.git && cd Boovore-tts
docker build -t boovore .
docker run -p 7860:7860 -e ENABLED_ENGINES=kokoro,f5 boovore
# open http://localhost:7860
```

The first request to each engine downloads its model. Kokoro is fast on CPU. F5-TTS works but is slow there: about 3 minutes for one short sentence on a 6-core laptop CPU.

### 3. GPU server (all six engines)

Tested on an RTX 3090 (24 GB) rented on Vast.ai, Python 3.12 with CUDA 12.8.

<details>
<summary><b>Install steps</b></summary>

```bash
# PyTorch nightly, CUDA 12.8
pip3 install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
# Let torchaudio find libtorch
echo "/usr/local/lib/python3.12/dist-packages/torch/lib" > /etc/ld.so.conf.d/torch.conf && ldconfig

# Core engines
pip3 install faster-qwen3-tts kokoro f5-tts fastapi "uvicorn[standard]" python-multipart

# Chatterbox (Python 3.12 fix)
pip3 install conformer==0.3.2 --no-build-isolation
git clone https://github.com/resemble-ai/chatterbox /tmp/chatterbox
cd /tmp/chatterbox && pip3 install -e . --no-deps && cd -

# Fish-Speech 1.5
git clone https://github.com/fishaudio/fish-speech /tmp/fish-speech
cd /tmp/fish-speech && git checkout v1.5.1 && pip3 install -e . --no-deps && cd -
huggingface-cli download fishaudio/fish-speech-1.5 --local-dir /root/fish-speech-model

# Start Boovore
nohup python3 server.py --port 7860 >> server.log 2>&1 &
```

**Voxtral (optional).** It runs as a separate vLLM-Omni process and needs about 8 GB of VRAM. First accept the license on [mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603), then:

```bash
pip install "vllm[audio]>=0.18.0" httpx soundfile
export HF_TOKEN=hf_xxxx
python3 voxtral_server.py            # --status / --stop
python3 make_narrator_reference.py   # optional: build a narrator reference WAV with Qwen3
```

Open the UI through an SSH tunnel: `ssh -p <PORT> root@<HOST> -L 7860:localhost:7860 -N`, then go to http://localhost:7860.

</details>

### 4. Windows laptop with a 6 GB GPU (Qwen3 only)

Create a `venv\` with PyTorch (CUDA) and `faster-qwen3-tts`, then double-click **`start_qwen3_local.bat`**. It opens http://localhost:7860 once the model is loaded. It was tested with Python 3.12, torch 2.7.1+cu118 and faster-qwen3-tts 0.2.4 on an RTX 2060 laptop.

<details>
<summary><b>What the launcher sets, and why</b></summary>

| Variable | Value | Why |
|---|---|---|
| `ENABLED_ENGINES` | `qwen3` | Only Qwen3 is installed in the local venv |
| `MODEL_CACHE_SIZE` | `1` | 6 GB holds one 1.7B model (about 4.5 GB peak). Loading another one unloads it and frees the VRAM |
| `TRANSCRIBE_DEVICE` | `off` | The reference-clip transcriber does not fit next to a 1.7B model |
| `QWEN3_DTYPE` | `auto` | bf16 on Ampere and newer. On Turing (RTX 20xx, T4): fp16 for 1.7B and bf16 for 0.6B, because 0.6B overflows in fp16 |

Measured on an RTX 2060 laptop: French output was word-exact (checked by speech recognition), but speed was only 0.1 to 0.5× real time because the GPU throttles when it heats up. For long audiobooks, use a desktop GPU or a rented one.

</details>

## Pick engines for your hardware

| `ENABLED_ENGINES` | Hardware | Engines |
|---|---|---|
| `kokoro,f5` | CPU (e.g. free HF Space) | Kokoro, F5-TTS |
| `kokoro,f5,chatterbox` | GPU with about 6 GB (e.g. T4) | + Chatterbox |
| `qwen3` | 6 GB laptop GPU | Qwen3-TTS |
| `all` (default) | 24 GB GPU (RTX 3090, A10G, A100) | Everything (Voxtral also needs its own server) |

Other settings: `ACTIVE_MODELS` (comma-separated list of the Qwen3 models offered in the UI), `MODEL_CACHE_SIZE` (Qwen3 models kept in memory, default 2), `VOXTRAL_URL` (default `http://localhost:8000`).

## HTTP API

The web page is a thin client over a FastAPI server, so you can script any engine:

```bash
# Kokoro, French voice
curl -X POST http://localhost:7860/generate/kokoro_fr \
     -F "text=Bonjour, je suis Boovore." -F "voice=ff_siwis" -o bonjour.wav

# Qwen3 Voice Design: describe the voice instead of cloning it
curl -X POST http://localhost:7860/load -F "model_id=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
curl -X POST http://localhost:7860/generate -F "mode=voice_design" -F "language=French" \
     -F "instruct=A calm, deep male narrator, warm and slow." \
     -F "text=La nuit tombait lentement sur la plaine." -o narrateur.wav
```

| Endpoint | What it does |
|---|---|
| `GET /status` | Loaded model, available models, enabled engines |
| `POST /load` | Load a Qwen3 model (`model_id`) |
| `POST /generate`, `POST /generate/stream` | Qwen3: `voice_clone`, `custom`, `voice_design` |
| `POST /generate/kokoro_fr` | Kokoro |
| `POST /generate/f5_fr` | F5-TTS (`ref_wav`, `ref_text`) |
| `POST /generate/chatterbox` | Chatterbox (`ref_wav`, `exaggeration`) |
| `POST /generate/fish` | Fish-Speech |
| `POST /generate/voxtral` | Voxtral (proxied to vLLM-Omni) |
| `POST /transcribe` | Transcribe a reference clip (nano-parakeet) |

## Project layout

```
server.py                    FastAPI backend: all engines, model cache, HTTP API
index.html                   The whole UI (studio + audiobook), vanilla JS
voxtral_server.py            Start / stop / check the Voxtral vLLM-Omni server
make_narrator_reference.py   Build a narrator reference WAV with Qwen3 Voice Design
start_qwen3_local.bat        Windows launcher for 6 GB GPUs
xtts_server.py               Optional fine-tuned XTTS French narrator (port 7861)
Dockerfile                   CPU image used by the Hugging Face Space
```

## Contributing

Issues and pull requests are welcome, especially:

- a new engine (one `/generate/<engine>` endpoint plus one pill in `index.html`)
- voices and presets for languages other than French
- install reports for GPUs not listed here

For a larger change, open an issue first so we can agree on the approach.

## Credits

Boovore started as a fork of the web demo of **[faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)** by Andres Marafioti, whose CUDA-graph Qwen3 engine powers the streaming mode. The other engines are by [Resemble AI](https://github.com/resemble-ai/chatterbox), [SWivid](https://github.com/SWivid/F5-TTS) and [RASPIAUDIO](https://huggingface.co/RASPIAUDIO/F5-French-MixedSpeakers-reduced), [fishaudio](https://github.com/fishaudio/fish-speech), [hexgrad](https://github.com/hexgrad/kokoro), the [Qwen team](https://huggingface.co/Qwen) and [Mistral AI](https://mistral.ai/news/voxtral-tts/). The French prosody preprocessing is inspired by [arXiv:2508.17494](https://arxiv.org/abs/2508.17494).

## License

The code is released under the [MIT license](../LICENSE). Model weights are licensed separately; see the [engines table](#engines).

---

<p align="center">
  If Boovore saved you an afternoon of installing TTS repos, a ⭐ helps other people find it.
</p>
