"""
XTTS Narrator FR — standalone server on port 7861
Requires Python 3.11 + Coqui TTS
Run: py -3.11 xtts_server.py   (Windows)
     python3.11 xtts_server.py  (Linux)

Model is auto-downloaded from HuggingFace Hub on first start:
  terrancemoon9/xtts-narrator-fr
"""
import os, io, sys, tempfile, functools
from pathlib import Path

os.environ["COQUI_TOS_AGREED"] = "1"

import torch
_orig_load = torch.load
@functools.wraps(_orig_load)
def _patched_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_load(*args, **kwargs)
torch.load = _patched_load

from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import Response, JSONResponse
import uvicorn

HF_REPO    = "terrancemoon9/xtts-narrator-fr"
CACHE_DIR  = Path(__file__).parent / "xtts-narrator-fr-cache"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
PORT       = 7861

app = FastAPI()
tts_model = None
_model_dir = None

def get_model_dir() -> Path:
    """Download model from HF Hub if not already cached, return local path."""
    global _model_dir
    if _model_dir is not None:
        return _model_dir

    # Check for a local override (for dev/offline use)
    local_override = Path(__file__).parent.parent / "xtts-narrator-fr"
    if local_override.exists() and (local_override / "model.pth").exists():
        print(f"Utilisation du modèle local: {local_override}")
        _model_dir = local_override
        return _model_dir

    # Download from HuggingFace Hub
    print(f"Téléchargement du modèle depuis HuggingFace Hub: {HF_REPO}")
    try:
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(
            repo_id=HF_REPO,
            cache_dir=str(CACHE_DIR),
            local_files_only=False,
        )
        _model_dir = Path(local_path)
        print(f"Modèle téléchargé dans: {_model_dir}")
    except Exception as e:
        print(f"ERREUR téléchargement HF Hub: {e}")
        sys.exit(1)
    return _model_dir

def load_model():
    global tts_model
    if tts_model is not None:
        return tts_model
    model_dir = get_model_dir()
    print(f"Chargement XTTS narrateur FR depuis {model_dir}...")
    from TTS.api import TTS
    tts_model = TTS(
        model_path=str(model_dir),
        config_path=str(model_dir / "config.json"),
    ).to(DEVICE)
    print(f"XTTS chargé sur {DEVICE}")
    return tts_model

def get_ref_wav() -> str:
    """Return path to reference wav — from model dir or local fallback."""
    # Prefer narrator_ref.wav bundled in the model repo
    model_dir = get_model_dir()
    ref = model_dir / "narrator_ref.wav"
    if ref.exists():
        return str(ref)
    # Fallback: local file next to this script
    local_ref = Path(__file__).parent / "narrator_ref.wav"
    return str(local_ref)

@app.get("/status")
def status():
    return {"loaded": tts_model is not None, "device": DEVICE, "model": HF_REPO}

@app.post("/generate")
async def generate(
    text: str = Form(...),
    ref_wav: UploadFile = File(None),
):
    model = load_model()

    # Reference audio: use uploaded file or default narrator_ref.wav
    if ref_wav and ref_wav.filename:
        ref_bytes = await ref_wav.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(ref_bytes)
            ref_path = tmp.name
    else:
        ref_path = get_ref_wav()

    # Generate to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_tmp:
        out_path = out_tmp.name

    try:
        model.tts_to_file(
            text=text,
            speaker_wav=ref_path,
            language="fr",
            file_path=out_path,
        )
        wav_bytes = open(out_path, "rb").read()
        return Response(content=wav_bytes, media_type="audio/wav")
    finally:
        os.unlink(out_path)
        if ref_wav and ref_wav.filename:
            os.unlink(ref_path)

if __name__ == "__main__":
    print(f"XTTS Narrator FR → http://localhost:{PORT}")
    print(f"Repo HF: {HF_REPO}")
    print(f"Cache: {CACHE_DIR}")
    # Pre-download model at startup
    get_model_dir()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
