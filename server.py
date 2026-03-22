#!/usr/bin/env python3
"""
TTS Demo Server

Qwen3-TTS (Clone / Custom / VoiceDesign) → local RTX 3090 via faster_qwen3_tts
Kokoro FR   → local (hexgrad/Kokoro-82M)
F5-TTS FR   → local (RASPIAUDIO checkpoint)
Chatterbox  → local (ResembleAI)
Fish-Speech → local (fishaudio/fish-speech-1.5)
"""

import argparse
import asyncio
import base64
from collections import OrderedDict
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

# ── Fish-Speech ───────────────────────────────────────────────────────────────
FISH_SPEECH_REPO = Path("/tmp/fish-speech")
FISH_SPEECH_MODEL = Path("/root/fish-speech-model")

# Patch torchaudio nightly: list_audio_backends removed in recent builds
import torchaudio as _torchaudio
if not hasattr(_torchaudio, "list_audio_backends"):
    _torchaudio.list_audio_backends = lambda: ["ffmpeg", "sox"]
_fish_engine = None
_fish_lock = threading.Lock()

def _get_fish_engine():
    global _fish_engine
    if _fish_engine is not None:
        return _fish_engine
    with _fish_lock:
        if _fish_engine is not None:
            return _fish_engine
        if not FISH_SPEECH_REPO.exists():
            raise RuntimeError("Fish-Speech repo not found at /tmp/fish-speech. Run: git clone https://github.com/fishaudio/fish-speech /tmp/fish-speech && cd /tmp/fish-speech && git checkout v1.5.1")
        if not FISH_SPEECH_MODEL.exists():
            raise RuntimeError("Fish-Speech model not found at /root/fish-speech-model. Download with: python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('fishaudio/fish-speech-1.5', local_dir='/root/fish-speech-model')\"")
        sys.path.insert(0, str(FISH_SPEECH_REPO))
        from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
        from fish_speech.models.vqgan.inference import load_model as load_decoder_model
        from fish_speech.inference_engine import TTSInferenceEngine
        import torch as _torch
        device = "cuda" if _torch.cuda.is_available() else "cpu"
        precision = _torch.bfloat16
        llama_queue = launch_thread_safe_queue(
            checkpoint_path=str(FISH_SPEECH_MODEL),
            device=device,
            precision=precision,
            compile=False,
        )
        decoder_model = load_decoder_model(
            config_name="firefly_gan_vq",
            checkpoint_path=str(FISH_SPEECH_MODEL / "firefly-gan-vq-fsq-8x1024-21hz-generator.pth"),
            device=device,
        )
        _fish_engine = TTSInferenceEngine(
            llama_queue=llama_queue,
            decoder_model=decoder_model,
            precision=precision,
            compile=False,
        )
        return _fish_engine


# ── Enabled engines (CPU/GPU gating) ─────────────────────────────────────────
_ENABLED_ENV = os.environ.get("ENABLED_ENGINES", "all").lower()
ENABLED_ENGINES: set[str] | None = (
    None if _ENABLED_ENV == "all"
    else {e.strip() for e in _ENABLED_ENV.split(",") if e.strip()}
)

def _engine_enabled(name: str) -> bool:
    return ENABLED_ENGINES is None or name in ENABLED_ENGINES

# ── Kokoro TTS (multilingual) ─────────────────────────────────────────────────
_kokoro_pipelines: dict[str, object] = {}
_kokoro_lock = threading.Lock()

KOKORO_VOICES_FR = {
    # French
    "ff_siwis":   "Siwis — FR Femme ★",
    # American English — Female
    "af_heart":   "Heart — EN Femme ★",
    "af_bella":   "Bella — EN Femme",
    "af_nicole":  "Nicole — EN Femme (ASMR)",
    "af_sarah":   "Sarah — EN Femme",
    "af_sky":     "Sky — EN Femme",
    # American English — Male
    "am_echo":    "Echo — EN Homme",
    "am_michael": "Michael — EN Homme",
    "am_adam":    "Adam — EN Homme",
    # British English — Female
    "bf_emma":    "Emma — EN(UK) Femme",
    "bf_isabella":"Isabella — EN(UK) Femme",
    # British English — Male
    "bm_george":  "George — EN(UK) Homme",
    "bm_lewis":   "Lewis — EN(UK) Homme",
}

_VOICE_LANG_CODE = {
    "ff": "f",   # French female
    "af": "a",   # American English female
    "am": "a",   # American English male
    "bf": "b",   # British English female
    "bm": "b",   # British English male
}

def _get_kokoro(voice: str = "ff_siwis"):
    prefix = voice[:2] if len(voice) >= 2 else "ff"
    lang_code = _VOICE_LANG_CODE.get(prefix, "f")
    with _kokoro_lock:
        if lang_code not in _kokoro_pipelines:
            from kokoro import KPipeline
            _kokoro_pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return _kokoro_pipelines[lang_code]

# ── Chatterbox TTS (ResembleAI) ───────────────────────────────────────────────
_chatterbox_model = None
_chatterbox_lock = threading.Lock()

def _get_chatterbox():
    global _chatterbox_model
    if _chatterbox_model is None:
        with _chatterbox_lock:
            if _chatterbox_model is None:
                from chatterbox.tts import ChatterboxTTS
                _device = "cuda" if torch.cuda.is_available() else "cpu"
                _chatterbox_model = ChatterboxTTS.from_pretrained(device=_device)
    return _chatterbox_model

# ── F5-TTS French (RASPIAUDIO checkpoint) ────────────────────────────────────
_f5_model = None
_f5_lock = threading.Lock()
F5_REPO = "RASPIAUDIO/F5-French-MixedSpeakers-reduced"

def _get_f5():
    global _f5_model
    if _f5_model is None:
        with _f5_lock:
            if _f5_model is None:
                from f5_tts.api import F5TTS
                from huggingface_hub import hf_hub_download
                ckpt  = hf_hub_download(F5_REPO, "model_last_reduced.pt")
                vocab = hf_hub_download(F5_REPO, "vocab.txt")
                _f5_model = F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt, vocab_file=vocab)
    return _f5_model

# ── Qwen3-TTS (local) ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from faster_qwen3_tts import FasterQwen3TTS
except ImportError:
    if _engine_enabled("qwen3"):
        print("Error: faster_qwen3_tts not found. Install with: pip install faster-qwen3-tts")
        sys.exit(1)
    FasterQwen3TTS = None  # type: ignore

try:
    from nano_parakeet import from_pretrained as _parakeet_from_pretrained
except ImportError:
    _parakeet_from_pretrained = None

_ALL_MODELS = [
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
]

_active_models_env = os.environ.get("ACTIVE_MODELS", "")
if _active_models_env:
    _allowed = {m.strip() for m in _active_models_env.split(",") if m.strip()}
    AVAILABLE_MODELS = [m for m in _ALL_MODELS if m in _allowed]
else:
    AVAILABLE_MODELS = list(_ALL_MODELS)

BASE_DIR = Path(__file__).resolve().parent
_ASSET_DIR = Path(os.environ.get("ASSET_DIR", "/tmp/faster-qwen3-tts-assets"))
PRESET_TRANSCRIPTS = _ASSET_DIR / "samples" / "parity" / "icl_transcripts.txt"
PRESET_REFS = [
    ("ref_audio_3", _ASSET_DIR / "ref_audio_3.wav", "Clone 1"),
    ("ref_audio_2", _ASSET_DIR / "ref_audio_2.wav", "Clone 2"),
    ("ref_audio",   _ASSET_DIR / "ref_audio.wav",   "Clone 3"),
]

_GITHUB_RAW = "https://raw.githubusercontent.com/andimarafioti/faster-qwen3-tts/main"
_PRESET_REMOTE = {
    "ref_audio":   f"{_GITHUB_RAW}/ref_audio.wav",
    "ref_audio_2": f"{_GITHUB_RAW}/ref_audio_2.wav",
    "ref_audio_3": f"{_GITHUB_RAW}/ref_audio_3.wav",
}
_TRANSCRIPT_REMOTE = f"{_GITHUB_RAW}/samples/parity/icl_transcripts.txt"


def _fetch_preset_assets():
    import urllib.request
    _ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PRESET_TRANSCRIPTS.parent.mkdir(parents=True, exist_ok=True)
    if not PRESET_TRANSCRIPTS.exists():
        try:
            urllib.request.urlretrieve(_TRANSCRIPT_REMOTE, PRESET_TRANSCRIPTS)
        except Exception as e:
            print(f"Warning: could not fetch transcripts: {e}")
    for key, path, _ in PRESET_REFS:
        if not path.exists() and key in _PRESET_REMOTE:
            try:
                urllib.request.urlretrieve(_PRESET_REMOTE[key], path)
                print(f"Downloaded {path.name}")
            except Exception as e:
                print(f"Warning: could not fetch {key}: {e}")

_preset_refs: dict[str, dict] = {}


def _load_preset_transcripts():
    if not PRESET_TRANSCRIPTS.exists():
        return {}
    transcripts = {}
    for line in PRESET_TRANSCRIPTS.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key_part, text = line.split(":", 1)
        key = key_part.split("(")[0].strip()
        transcripts[key] = text.strip()
    return transcripts


def _load_preset_refs():
    transcripts = _load_preset_transcripts()
    for key, path, label in PRESET_REFS:
        if not path.exists():
            continue
        content = path.read_bytes()
        cached_path = _get_cached_ref_path(content)
        _preset_refs[key] = {
            "id": key, "label": label, "filename": path.name,
            "path": cached_path, "ref_text": transcripts.get(key, ""),
            "audio_b64": base64.b64encode(content).decode(),
        }


def _prime_preset_voice_cache(model):
    if not _preset_refs:
        return
    for preset in _preset_refs.values():
        for xvec_only in (True, False):
            try:
                model._prepare_generation(
                    text="Hello.", ref_audio=preset["path"], ref_text=preset["ref_text"],
                    language="English", xvec_only=xvec_only, non_streaming_mode=True,
                )
            except Exception:
                continue


app = FastAPI(title="Faster Qwen3-TTS Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_model_cache: OrderedDict[str, FasterQwen3TTS] = OrderedDict()
_model_cache_max: int = int(os.environ.get("MODEL_CACHE_SIZE", "2"))
_active_model_name: str | None = None
_loading = False
_ref_cache: dict[str, str] = {}
_ref_cache_lock = threading.Lock()
_parakeet = None
_generation_lock = asyncio.Lock()
_generation_waiters: int = 0

MAX_TEXT_CHARS  = 15000
MAX_AUDIO_BYTES = 10 * 1024 * 1024
_AUDIO_TOO_LARGE_MSG = (
    "Audio file too large ({size_mb:.1f} MB). "
    "Please upload a shorter recording (under 1 minute)."
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_wav_b64(audio: np.ndarray, sr: int) -> str:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def _concat_audio(audio_list) -> np.ndarray:
    if isinstance(audio_list, np.ndarray):
        return audio_list.astype(np.float32).squeeze()
    parts = [np.array(a, dtype=np.float32).squeeze() for a in audio_list if len(a) > 0]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _get_cached_ref_path(content: bytes) -> str:
    digest = hashlib.sha1(content).hexdigest()
    with _ref_cache_lock:
        cached = _ref_cache.get(digest)
        if cached and os.path.exists(cached):
            return cached
        path = Path(tempfile.gettempdir()) / f"qwen3tts_ref_{digest}.wav"
        if not path.exists():
            path.write_bytes(content)
        _ref_cache[digest] = str(path)
        return str(path)


# ─── Routes ───────────────────────────────────────────────────────────────────

_fetch_preset_assets()
_load_preset_refs()

@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    if _parakeet is None:
        raise HTTPException(status_code=503, detail="Transcription model not loaded")
    content = await audio.read()
    if len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=400, detail=_AUDIO_TOO_LARGE_MSG.format(size_mb=len(content)/1024/1024))
    def run():
        import torchaudio
        wav, sr = sf.read(io.BytesIO(content), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav_t = torch.from_numpy(wav)
        if sr != 16000:
            wav_t = torchaudio.functional.resample(wav_t.unsqueeze(0), sr, 16000).squeeze(0)
        return _parakeet.transcribe(wav_t.cuda())
    text = await asyncio.to_thread(run)
    return {"text": text}


@app.get("/status")
async def get_status():
    speakers = []
    model_type = None
    active = _model_cache.get(_active_model_name) if _active_model_name else None
    if active is not None:
        try:
            model_type = active.model.model.tts_model_type
            speakers = active.model.get_supported_speakers() or []
        except Exception:
            speakers = []
    return {
        "loaded": active is not None,
        "model": _active_model_name,
        "loading": _loading,
        "available_models": AVAILABLE_MODELS,
        "model_type": model_type,
        "speakers": speakers,
        "transcription_available": _parakeet is not None,
        "preset_refs": [{"id": p["id"], "label": p["label"], "ref_text": p["ref_text"]} for p in _preset_refs.values()],
        "queue_depth": _generation_waiters,
        "cached_models": list(_model_cache.keys()),
        "kokoro_voices": KOKORO_VOICES_FR,
    }


@app.post("/load")
async def load_model(model_id: str = Form(...)):
    if not _engine_enabled("qwen3"):
        raise HTTPException(status_code=503, detail="Qwen3 engine not enabled on this server.")
    global _active_model_name, _loading
    if model_id in _model_cache:
        _active_model_name = model_id
        _model_cache.move_to_end(model_id)
        return {"status": "already_loaded", "model": model_id}
    _loading = True
    def _do_load():
        global _active_model_name, _loading
        try:
            if len(_model_cache) >= _model_cache_max:
                evicted, _ = _model_cache.popitem(last=False)
                print(f"Model cache full — evicted: {evicted}")
            new_model = FasterQwen3TTS.from_pretrained(model_id, device="cuda", dtype=torch.bfloat16)
            print("Capturing CUDA graphs…")
            new_model._warmup(prefill_len=100)
            _model_cache[model_id] = new_model
            _model_cache.move_to_end(model_id)
            _active_model_name = model_id
            _prime_preset_voice_cache(new_model)
            print("CUDA graphs captured — model ready.")
        finally:
            _loading = False
    async with _generation_lock:
        await asyncio.to_thread(_do_load)
    return {"status": "loaded", "model": model_id}


@app.post("/generate/stream")
async def generate_stream(
    text: str = Form(...),
    language: str = Form("English"),
    mode: str = Form("voice_clone"),
    ref_text: str = Form(""),
    speaker: str = Form(""),
    instruct: str = Form(""),
    xvec_only: bool = Form(True),
    chunk_size: int = Form(8),
    temperature: float = Form(0.7),
    top_k: int = Form(30),
    repetition_penalty: float = Form(1.1),
    ref_preset: str = Form(""),
    ref_audio: UploadFile = File(None),
    seed: int = Form(None),
):
    if not _engine_enabled("qwen3"):
        raise HTTPException(status_code=503, detail="Qwen3 engine not enabled on this server.")
    if not _active_model_name or _active_model_name not in _model_cache:
        raise HTTPException(status_code=400, detail="Modèle non chargé. Cliquez sur 'Load' d'abord.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=400, detail=f"Texte trop long ({len(text)} chars). Max {MAX_TEXT_CHARS}.")

    tmp_path = None
    tmp_is_cached = False

    if ref_preset and ref_preset in _preset_refs:
        preset = _preset_refs[ref_preset]
        tmp_path = preset["path"]
        tmp_is_cached = True
        if not ref_text:
            ref_text = preset["ref_text"]
    elif ref_audio and ref_audio.filename:
        content = await ref_audio.read()
        if len(content) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=400, detail=_AUDIO_TOO_LARGE_MSG.format(size_mb=len(content)/1024/1024))
        tmp_path = _get_cached_ref_path(content)
        tmp_is_cached = True

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def run_generation():
        try:
            model = _model_cache.get(_active_model_name)
            if model is None:
                raise RuntimeError("No model loaded.")
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            t0 = time.perf_counter()
            total_audio_s = 0.0
            voice_clone_ms = 0.0

            if mode == "voice_clone":
                gen = model.generate_voice_clone_streaming(
                    text=text, language=language, ref_audio=tmp_path, ref_text=ref_text,
                    xvec_only=xvec_only, chunk_size=chunk_size, temperature=temperature,
                    top_k=top_k, repetition_penalty=repetition_penalty, max_new_tokens=1800,
                )
            elif mode == "custom":
                if not speaker:
                    raise ValueError("Speaker ID required for custom voice")
                gen = model.generate_custom_voice_streaming(
                    text=text, speaker=speaker, language=language, instruct=instruct,
                    chunk_size=chunk_size, temperature=temperature, top_k=top_k,
                    repetition_penalty=repetition_penalty, max_new_tokens=1800,
                )
            else:
                gen = model.generate_voice_design_streaming(
                    text=text, instruct=instruct, language=language, chunk_size=chunk_size,
                    temperature=temperature, top_k=top_k, repetition_penalty=repetition_penalty,
                    max_new_tokens=1800,
                )

            ttfa_ms = None
            total_gen_ms = 0.0
            first_audio = next(gen, None)
            if first_audio is not None:
                audio_chunk, sr, timing = first_audio
                wall_first_ms = (time.perf_counter() - t0) * 1000
                model_ms = timing.get("prefill_ms", 0) + timing.get("decode_ms", 0)
                voice_clone_ms = max(0.0, wall_first_ms - model_ms)
                total_gen_ms += timing.get('prefill_ms', 0) + timing.get('decode_ms', 0)
                if ttfa_ms is None:
                    ttfa_ms = total_gen_ms
                audio_chunk = _concat_audio(audio_chunk)
                dur = len(audio_chunk) / sr
                total_audio_s += dur
                rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                    "type": "chunk", "audio_b64": _to_wav_b64(audio_chunk, sr),
                    "sample_rate": sr, "ttfa_ms": round(ttfa_ms), "voice_clone_ms": round(voice_clone_ms),
                    "rtf": round(rtf, 3), "total_audio_s": round(total_audio_s, 3),
                    "elapsed_ms": round(time.perf_counter() - t0, 3) * 1000,
                }))

            for audio_chunk, sr, timing in gen:
                total_gen_ms += timing.get('prefill_ms', 0) + timing.get('decode_ms', 0)
                if ttfa_ms is None:
                    ttfa_ms = total_gen_ms
                audio_chunk = _concat_audio(audio_chunk)
                dur = len(audio_chunk) / sr
                total_audio_s += dur
                rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0
                loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                    "type": "chunk", "audio_b64": _to_wav_b64(audio_chunk, sr),
                    "sample_rate": sr, "ttfa_ms": round(ttfa_ms), "voice_clone_ms": round(voice_clone_ms),
                    "rtf": round(rtf, 3), "total_audio_s": round(total_audio_s, 3),
                    "elapsed_ms": round(time.perf_counter() - t0, 3) * 1000,
                }))

            rtf = total_audio_s / (total_gen_ms / 1000) if total_gen_ms > 0 else 0.0
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                "type": "done", "ttfa_ms": round(ttfa_ms) if ttfa_ms else 0,
                "voice_clone_ms": round(voice_clone_ms), "rtf": round(rtf, 3),
                "total_audio_s": round(total_audio_s, 3),
                "total_ms": round((time.perf_counter() - t0) * 1000),
            }))

        except Exception as e:
            import traceback
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                "type": "error", "message": str(e), "detail": traceback.format_exc()
            }))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)
            if tmp_path and os.path.exists(tmp_path) and not tmp_is_cached:
                os.unlink(tmp_path)

    async def sse():
        global _generation_waiters
        lock_acquired = False
        _generation_waiters += 1
        people_ahead = _generation_waiters - 1 + (1 if _generation_lock.locked() else 0)
        try:
            if people_ahead > 0:
                yield f"data: {json.dumps({'type': 'queued', 'position': people_ahead})}\n\n"
            await _generation_lock.acquire()
            lock_acquired = True
            _generation_waiters -= 1
            threading.Thread(target=run_generation, daemon=True).start()
            while True:
                msg = await queue.get()
                if msg is None:
                    break
                yield f"data: {msg}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if lock_acquired:
                _generation_lock.release()
            else:
                _generation_waiters -= 1

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/generate")
async def generate_non_streaming(
    text: str = Form(...),
    language: str = Form("English"),
    mode: str = Form("voice_clone"),
    ref_text: str = Form(""),
    speaker: str = Form(""),
    instruct: str = Form(""),
    xvec_only: bool = Form(True),
    temperature: float = Form(0.7),
    top_k: int = Form(30),
    repetition_penalty: float = Form(1.1),
    ref_preset: str = Form(""),
    ref_audio: UploadFile = File(None),
    seed: int = Form(None),
):
    if not _engine_enabled("qwen3"):
        raise HTTPException(status_code=503, detail="Qwen3 engine not enabled on this server.")
    if not _active_model_name or _active_model_name not in _model_cache:
        raise HTTPException(status_code=400, detail="Modèle non chargé.")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=400, detail=f"Texte trop long ({len(text)} chars).")

    tmp_path = None
    tmp_is_cached = False
    if ref_preset and ref_preset in _preset_refs:
        preset = _preset_refs[ref_preset]
        tmp_path = preset["path"]
        tmp_is_cached = True
        if not ref_text:
            ref_text = preset["ref_text"]
    elif ref_audio and ref_audio.filename:
        content = await ref_audio.read()
        if len(content) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=400, detail=_AUDIO_TOO_LARGE_MSG.format(size_mb=len(content)/1024/1024))
        tmp_path = _get_cached_ref_path(content)
        tmp_is_cached = True

    def run():
        model = _model_cache.get(_active_model_name)
        if model is None:
            raise RuntimeError("No model loaded.")
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        t0 = time.perf_counter()
        if mode == "voice_clone":
            audio_list, sr = model.generate_voice_clone(
                text=text, language=language, ref_audio=tmp_path, ref_text=ref_text,
                xvec_only=xvec_only, temperature=temperature, top_k=top_k,
                repetition_penalty=repetition_penalty, max_new_tokens=1800,
            )
        elif mode == "custom":
            if not speaker:
                raise ValueError("Speaker ID required")
            audio_list, sr = model.generate_custom_voice(
                text=text, speaker=speaker, language=language, instruct=instruct,
                temperature=temperature, top_k=top_k, repetition_penalty=repetition_penalty, max_new_tokens=1800,
            )
        else:
            audio_list, sr = model.generate_voice_design(
                text=text, instruct=instruct, language=language,
                temperature=temperature, top_k=top_k, repetition_penalty=repetition_penalty, max_new_tokens=1800,
            )
        elapsed = time.perf_counter() - t0
        audio = _concat_audio(audio_list)
        dur = len(audio) / sr
        return audio, sr, elapsed, dur

    global _generation_waiters
    _generation_waiters += 1
    lock_acquired = False
    try:
        await _generation_lock.acquire()
        lock_acquired = True
        _generation_waiters -= 1
        audio, sr, elapsed, dur = await asyncio.to_thread(run)
        rtf = dur / elapsed if elapsed > 0 else 0.0
        return JSONResponse({
            "audio_b64": _to_wav_b64(audio, sr),
            "sample_rate": sr,
            "metrics": {"total_ms": round(elapsed * 1000), "audio_duration_s": round(dur, 3), "rtf": round(rtf, 3)},
        })
    finally:
        if lock_acquired:
            _generation_lock.release()
        else:
            _generation_waiters -= 1
        if tmp_path and os.path.exists(tmp_path) and not tmp_is_cached:
            os.unlink(tmp_path)


@app.post("/generate/kokoro_fr")
async def generate_kokoro_fr(
    text: str = Form(...),
    voice: str = Form("ff_siwis"),
    speed: float = Form(1.0),
):
    if not _engine_enabled("kokoro"):
        raise HTTPException(status_code=503, detail="Kokoro engine not enabled on this server.")
    if voice not in KOKORO_VOICES_FR:
        voice = "ff_siwis"
    speed = max(0.5, min(2.0, speed))
    def _run():
        pipeline = _get_kokoro(voice)
        chunks = []
        for _gs, _ps, audio in pipeline(text, voice=voice, speed=speed):
            chunks.append(audio.numpy() if hasattr(audio, "numpy") else audio)
        if not chunks:
            raise ValueError("Kokoro: no audio generated")
        combined = np.concatenate(chunks)
        buf = io.BytesIO()
        sf.write(buf, combined, 24000, format="WAV")
        buf.seek(0)
        return buf.read()
    try:
        wav_bytes = await asyncio.get_event_loop().run_in_executor(None, _run)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/f5_fr")
async def generate_f5_fr(
    text: str = Form(...),
    ref_wav: UploadFile = File(None),
    ref_text: str = Form(""),
    speed: float = Form(1.0),
    nfe_step: int = Form(32),
    cross_fade_duration: float = Form(0.15),
    seed: int = Form(None),
):
    if not _engine_enabled("f5"):
        raise HTTPException(status_code=503, detail="F5-TTS engine not enabled on this server.")
    ref_path = None
    cleanup_ref = False
    if ref_wav and ref_wav.filename:
        ref_bytes = await ref_wav.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(ref_bytes)
            ref_path = tmp.name
            cleanup_ref = True
    else:
        # Try local narrator ref, then bundled F5-TTS English ref as fallback
        for candidate in [
            Path(__file__).parent / "narrator_ref.wav",
            Path("/usr/local/lib/python3.12/dist-packages/f5_tts/infer/examples/basic/basic_ref_en.wav"),
        ]:
            if candidate.exists():
                ref_path = str(candidate)
                # Use bundled ref text if it matches the bundled audio
                if "basic_ref_en" in ref_path and not ref_text:
                    ref_text = "Some call me nature, others call me mother nature."
                break

    def _run():
        model = _get_f5()
        out_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out_path = out_tmp.name
        out_tmp.close()
        try:
            if seed is not None:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            model.infer(
                ref_file=ref_path, ref_text=ref_text or "", gen_text=text,
                file_wave=out_path,
                speed=max(0.5, min(2.0, speed)),
                nfe_step=max(8, min(64, nfe_step)),
                cross_fade_duration=max(0.0, min(0.5, cross_fade_duration)),
            )
            return open(out_path, "rb").read()
        finally:
            os.unlink(out_path)
            if cleanup_ref and ref_path:
                os.unlink(ref_path)

    try:
        wav_bytes = await asyncio.get_event_loop().run_in_executor(None, _run)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/chatterbox")
async def generate_chatterbox(
    text: str = Form(...),
    ref_wav: UploadFile = File(None),
    exaggeration: float = Form(0.5),
    cfg_weight: float = Form(0.5),
    temperature: float = Form(0.8),
    seed: int = Form(None),
):
    if not _engine_enabled("chatterbox"):
        raise HTTPException(status_code=503, detail="Chatterbox engine not enabled on this server.")
    ref_path = None
    cleanup_ref = False
    if ref_wav and ref_wav.filename:
        ref_bytes = await ref_wav.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(ref_bytes)
            ref_path = tmp.name
            cleanup_ref = True

    def _run():
        import torchaudio
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        model = _get_chatterbox()
        kwargs = dict(audio_prompt_path=ref_path, exaggeration=exaggeration, cfg_weight=cfg_weight)
        try:
            wav = model.generate(text, temperature=max(0.1, min(2.0, temperature)), **kwargs)
        except TypeError:
            wav = model.generate(text, **kwargs)  # fallback if temperature not supported
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_tmp:
            out_path = out_tmp.name
        try:
            torchaudio.save(out_path, wav, model.sr)
            return open(out_path, "rb").read()
        finally:
            os.unlink(out_path)

    try:
        wav_bytes = await asyncio.get_event_loop().run_in_executor(None, _run)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cleanup_ref and ref_path and os.path.exists(ref_path):
            os.unlink(ref_path)


def _fish_split_text(text: str, max_words: int = 200) -> list[str]:
    """Split long text into paragraph groups to avoid truncation and improve quality."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paragraphs) <= 1:
        return [text]
    chunks, current, current_words = [], [], 0
    for p in paragraphs:
        words = len(p.split())
        if current_words + words > max_words and current:
            chunks.append("\n\n".join(current))
            current, current_words = [p], words
        else:
            current.append(p)
            current_words += words
    if current:
        chunks.append("\n\n".join(current))
    return chunks if len(chunks) > 1 else [text]


@app.post("/generate/fish")
async def generate_fish(
    text: str = Form(...),
    ref_wav: UploadFile = File(None),
    ref_text: str = Form(""),
    temperature: float = Form(0.8),
    top_p: float = Form(0.8),
    repetition_penalty: float = Form(1.1),
    max_new_tokens: int = Form(1024),
    chunk_length: int = Form(200),
    latency: str = Form("normal"),
    normalize: bool = Form(True),
    seed: int = Form(None),
    auto_split: bool = Form(False),
):
    if not _engine_enabled("fish"):
        raise HTTPException(status_code=503, detail="Fish-Speech engine not enabled on this server.")
    ref_path = None
    cleanup_ref = False
    if ref_wav and ref_wav.filename:
        ref_bytes = await ref_wav.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(ref_bytes)
            ref_path = tmp.name
            cleanup_ref = True

    def _run():
        sys.path.insert(0, str(FISH_SPEECH_REPO))
        from fish_speech.utils.schema import ServeTTSRequest, ServeReferenceAudio
        engine = _get_fish_engine()

        # Build references once — reused across all text splits
        references = []
        ref_audio_bytes = None
        if ref_path:
            with open(ref_path, "rb") as f:
                ref_audio_bytes = f.read()
            references = [ServeReferenceAudio(audio=ref_audio_bytes, text=ref_text or "")]

        # Optionally split long text at paragraph boundaries
        texts = _fish_split_text(text) if auto_split and len(text) > 600 else [text]

        common = dict(
            references=references,
            temperature=max(0.1, min(1.0, temperature)),
            top_p=max(0.1, min(1.0, top_p)),
            repetition_penalty=max(0.9, min(2.0, repetition_penalty)),
            max_new_tokens=max(64, min(8192, max_new_tokens)),
            chunk_length=max(100, min(600, chunk_length)),
            latency=latency if latency in ("normal", "balanced") else "normal",
            normalize=normalize,
            seed=seed,
            format="wav",
            streaming=False,
        )

        all_audio = []
        sample_rate = 44100
        for txt in texts:
            req = ServeTTSRequest(text=txt, **common)
            for result in engine.inference(req):
                if result.code == "header":
                    if isinstance(result.audio, tuple):
                        sample_rate = result.audio[0]
                elif result.code in ("segment", "final"):
                    if isinstance(result.audio, tuple):
                        all_audio.append(result.audio[1])

        if not all_audio:
            raise ValueError("Fish-Speech: no audio generated")
        combined = np.concatenate(all_audio)
        buf = io.BytesIO()
        sf.write(buf, combined, sample_rate, format="WAV")
        buf.seek(0)
        return buf.read()

    try:
        wav_bytes = await asyncio.get_event_loop().run_in_executor(None, _run)
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if cleanup_ref and ref_path and os.path.exists(ref_path):
            os.unlink(ref_path)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Faster Qwen3-TTS Demo Server")
    parser.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                        help="Model to preload at startup")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7860)))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--no-preload", action="store_true",
                        help="Skip model loading at startup")
    args = parser.parse_args()

    if not args.no_preload:
        global _active_model_name, _parakeet
        print(f"Loading model: {args.model}")
        _startup_model = FasterQwen3TTS.from_pretrained(args.model, device="cuda", dtype=torch.bfloat16)
        print("Capturing CUDA graphs…")
        _startup_model._warmup(prefill_len=100)
        _model_cache[args.model] = _startup_model
        _active_model_name = args.model
        _prime_preset_voice_cache(_startup_model)
        print("TTS model ready.")

        if _parakeet_from_pretrained:
            print("Loading transcription model (nano-parakeet)…")
            _parakeet = _parakeet_from_pretrained(device="cuda")
            print("Transcription model ready.")

        print(f"Ready. Open http://localhost:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
