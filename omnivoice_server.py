"""
OmniVoice (k2-fsa) — standalone server on port 7863, proxied by server.py (/generate/omnivoice).

OmniVoice needs transformers 5.x while Qwen3-TTS pins 4.57, so it runs in its own env:
  setup_omnivoice.bat (Windows)  or  pip install omnivoice fastapi uvicorn python-multipart
Run: python omnivoice_server.py

The model loads on the first request, on the GPU when at least 3 GB are free, otherwise on the
CPU (about 25x slower). server.py unloads Qwen3 before calling it, and calls /unload before
loading Qwen3 again, so both can take turns on a 6 GB card.

Env:
  OMNIVOICE_PORT  default 7863
  OMNIVOICE_HOST  default 127.0.0.1 (only server.py on the same machine needs to reach it)
  OMNIVOICE_ASR   1 = transcribe reference clips with Whisper large-v3 turbo (~1.6 GB) when no
                  ref_text is given; default 0 = ref_text is required for cloning

Weights: CC-BY-NC (training data includes Emilia) — no commercial use.
"""
import asyncio
import gc
import io
import os
import tempfile
import threading
import time

import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

MODEL_ID = "k2-fsa/OmniVoice"
PORT = int(os.environ.get("OMNIVOICE_PORT", "7863"))
HOST = os.environ.get("OMNIVOICE_HOST", "127.0.0.1")
ALLOW_ASR = os.environ.get("OMNIVOICE_ASR", "0") == "1"
SAMPLE_RATE = 24000  # fallback; the model exposes sampling_rate
MIN_FREE_VRAM = 3 * 1024**3
MAX_REF_BYTES = 10 * 1024 * 1024

app = FastAPI()
_model = None
_device = None
_loading = False
_load_error = None
# One generation at a time; also guards load / unload
_lock = threading.Lock()


def _pick_device():
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free >= MIN_FREE_VRAM:
            return "cuda:0", torch.float16
        print(f"Carte graphique occupée ({free / 1024**3:.1f} Go libres) : OmniVoice sur le processeur", flush=True)
    return "cpu", torch.float32


def _unload_locked():
    global _model, _device
    if _model is not None:
        _model, _device = None, None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("OmniVoice déchargé", flush=True)


def _ensure_model_locked():
    """Load the model, or move it to the GPU if it was loaded on the CPU and the card is free now."""
    global _model, _device, _loading, _load_error
    if _model is not None and _device == "cpu" and torch.cuda.is_available():
        if torch.cuda.mem_get_info()[0] >= MIN_FREE_VRAM:
            _unload_locked()
    if _model is not None:
        return
    from omnivoice import OmniVoice
    device, dtype = _pick_device()
    _loading, _load_error = True, None
    try:
        t0 = time.time()
        _model = OmniVoice.from_pretrained(MODEL_ID, device_map=device, dtype=dtype)
        _device = "cpu" if device == "cpu" else "cuda"
        print(f"OmniVoice prêt sur {_device} en {time.time() - t0:.0f} s", flush=True)
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        raise
    finally:
        _loading = False


@app.get("/status")
def status():
    return {
        "loaded": _model is not None,
        "loading": _loading,
        "error": _load_error,
        "model": MODEL_ID,
        "device": _device,
        "lazy": True,  # loads on the first generation
        "asr": ALLOW_ASR,
    }


@app.post("/unload")
def unload():
    with _lock:
        _unload_locked()
    return {"status": "unloaded"}


@app.post("/generate")
async def generate(
    text: str = Form(...),
    language: str = Form(""),
    ref_wav: UploadFile = File(None),
    ref_text: str = Form(""),
    instruct: str = Form(""),
    speed: float = Form(1.0),
    seed: int | None = Form(None),
):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Texte vide.")
    ref_path = None
    if ref_wav and ref_wav.filename:
        ref_bytes = await ref_wav.read()
        if len(ref_bytes) > MAX_REF_BYTES:
            raise HTTPException(status_code=400, detail="Extrait de référence trop lourd (10 Mo max).")
        if not ref_text.strip() and not ALLOW_ASR:
            raise HTTPException(
                status_code=400,
                detail="Pour cloner avec OmniVoice, donne aussi la transcription exacte de l'extrait "
                       "(sinon il faut Whisper, ~1,6 Go : lancer avec OMNIVOICE_ASR=1).",
            )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(ref_bytes)
            ref_path = tmp.name

    options = {"language": language or None, "speed": speed if speed != 1.0 else None}
    if ref_path:
        options.update(ref_audio=ref_path, ref_text=ref_text.strip() or None)
    elif instruct.strip():
        options["instruct"] = instruct.strip()

    def _run():
        with _lock:
            _ensure_model_locked()
            if seed is not None:
                torch.manual_seed(seed)
            audio = _model.generate(text=text.strip(), **options)[0]
        rate = getattr(_model, "sampling_rate", None) or SAMPLE_RATE
        buf = io.BytesIO()
        sf.write(buf, audio, rate, format="WAV", subtype="PCM_16")
        return buf.getvalue(), len(audio) / rate

    t0 = time.time()
    try:
        wav, secs = await asyncio.to_thread(_run)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"OmniVoice : {e}"})
    finally:
        if ref_path and os.path.exists(ref_path):
            os.unlink(ref_path)
    took = time.time() - t0
    print(f"{secs:.1f} s d'audio en {took:.1f} s sur {_device} ({secs / took:.2f}x temps réel)", flush=True)
    return Response(content=wav, media_type="audio/wav",
                    headers={"X-Audio-Seconds": f"{secs:.2f}", "X-Gen-Seconds": f"{took:.2f}"})


if __name__ == "__main__":
    print(f"OmniVoice → http://{HOST}:{PORT} (modèle chargé à la première demande)", flush=True)
    uvicorn.run(app, host=HOST, port=PORT)
