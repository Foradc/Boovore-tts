"""
Kitten TTS 2 — standalone server on port 7862, proxied by server.py (/generate/kitten).

Kitten TTS 2 needs transformers 5.x while Qwen3-TTS pins 4.57, so it runs in its own env:
  setup_kitten.bat (Windows)  or  pip install kittenml fastapi uvicorn python-multipart
Run: python kitten_server.py

Env:
  KITTEN_PORT     default 7862
  KITTEN_HOST     default 127.0.0.1 (only server.py on the same machine needs to reach it)
  KITTEN_DEVICE   cpu | cuda (default: cuda when available)
  KITTEN_WEIGHTS  packed (default, 954 MB, lossless) | emb4 (469 MB) | full
  KITTEN_DECODER  default | student_w4 (39 MB, lower fidelity, not faster)
  KITTEN_WHISPER  1 = transcribe reference clips with Whisper large-v3 (~3 GB) when no
                  ref_text is given; default 0 = ref_text is required for cloning

Weights: Stellon Labs Community License — Powered by Stellon Labs.
"""
import asyncio
import io
import os
import tempfile
import threading
import time

import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

MODEL_ID = "KittenML/kitten-tts-2"
PORT = int(os.environ.get("KITTEN_PORT", "7862"))
HOST = os.environ.get("KITTEN_HOST", "127.0.0.1")
DEVICE = os.environ.get("KITTEN_DEVICE") or None
WEIGHTS = os.environ.get("KITTEN_WEIGHTS") or None
DECODER = os.environ.get("KITTEN_DECODER") or None
ALLOW_WHISPER = os.environ.get("KITTEN_WHISPER", "0") == "1"
MAX_REF_BYTES = 10 * 1024 * 1024

app = FastAPI()
_model = None
_load_error = None
# The model is not safe to call from two threads at once; requests queue here
_lock = threading.Lock()


def load_model():
    global _model, _load_error
    try:
        from kittenml import KittenTTS
        t0 = time.time()
        print(f"Chargement de {MODEL_ID} (device={DEVICE or 'auto'}, weights={WEIGHTS or 'packed'})…", flush=True)
        _model = KittenTTS(MODEL_ID, device=DEVICE, weights=WEIGHTS, decoder=DECODER)
        print(f"Kitten TTS 2 prêt sur {_model.device} en {time.time() - t0:.0f} s, "
              f"{len(_model.available_voices)} voix", flush=True)
    except Exception as e:  # reported by /status instead of killing the server
        _load_error = f"{type(e).__name__}: {e}"
        print(f"ERREUR chargement Kitten TTS 2 : {_load_error}", flush=True)


@app.get("/status")
def status():
    return {
        "loaded": _model is not None,
        "loading": _model is None and _load_error is None,
        "error": _load_error,
        "model": MODEL_ID,
        "device": str(_model.device) if _model else None,
        "voices": sorted(_model.available_voices) if _model else [],
        "presets": sorted(_model.decode_presets) if _model else [],
        "sample_rate": getattr(_model, "sample_rate", 24000),
        "whisper": ALLOW_WHISPER,
    }


@app.post("/generate")
async def generate(
    text: str = Form(...),
    voice: str = Form("French"),
    ref_wav: UploadFile = File(None),
    ref_text: str = Form(""),
    preset: str = Form("stable"),
    normalize: bool = Form(False),
    seed: int | None = Form(None),
):
    if _model is None:
        raise HTTPException(status_code=503, detail=_load_error or "Kitten TTS 2 est encore en chargement.")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Texte vide.")
    if preset not in _model.decode_presets:
        preset = "stable"

    ref_path = None
    if ref_wav and ref_wav.filename:
        ref_bytes = await ref_wav.read()
        if len(ref_bytes) > MAX_REF_BYTES:
            raise HTTPException(status_code=400, detail="Extrait de référence trop lourd (10 Mo max).")
        if not ref_text.strip() and not ALLOW_WHISPER:
            raise HTTPException(
                status_code=400,
                detail="Pour cloner avec Kitten, donne aussi la transcription exacte de l'extrait "
                       "(sinon il faut Whisper large-v3, ~3 Go : lancer avec KITTEN_WHISPER=1).",
            )
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(ref_bytes)
            ref_path = tmp.name
    elif voice not in _model.available_voices:
        raise HTTPException(status_code=400, detail=f"Voix inconnue : {voice}")

    options = {"preset": preset, "normalize": normalize}
    if seed is not None:
        options["advanced"] = {"seed": seed}
    if ref_path:
        options["reference"] = ref_path
        if ref_text.strip():
            options["reference_text"] = ref_text.strip()
    else:
        options["voice"] = voice

    def _run():
        with _lock:
            audio = _model.generate(text.strip(), **options)
        buf = io.BytesIO()
        sf.write(buf, audio, _model.sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue(), len(audio) / _model.sample_rate

    t0 = time.time()
    try:
        wav, secs = await asyncio.to_thread(_run)
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"Kitten TTS 2 : {e}"})
    finally:
        if ref_path and os.path.exists(ref_path):
            os.unlink(ref_path)
    took = time.time() - t0
    print(f"{secs:.1f} s d'audio en {took:.0f} s ({secs / took:.2f}x temps réel)", flush=True)
    return Response(content=wav, media_type="audio/wav",
                    headers={"X-Audio-Seconds": f"{secs:.2f}", "X-Gen-Seconds": f"{took:.2f}"})


if __name__ == "__main__":
    print(f"Kitten TTS 2 → http://{HOST}:{PORT}", flush=True)
    threading.Thread(target=load_model, daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT)
