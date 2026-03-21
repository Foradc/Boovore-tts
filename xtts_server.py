"""
XTTS Narrator FR — standalone server on port 7861
Requires Python 3.11 + Coqui TTS
Run: py -3.11 xtts_server.py   (Windows)
     python3.11 xtts_server.py  (Linux)
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

MODEL_DIR = Path(__file__).parent.parent / "xtts-narrator-fr"
REF_WAV   = Path(__file__).parent / "narrator_ref.wav"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
PORT      = 7861

app = FastAPI()
tts_model = None

def load_model():
    global tts_model
    if tts_model is not None:
        return tts_model
    print(f"Chargement XTTS narrateur FR depuis {MODEL_DIR}...")
    from TTS.api import TTS
    tts_model = TTS(
        model_path=str(MODEL_DIR),
        config_path=str(MODEL_DIR / "config.json"),
    ).to(DEVICE)
    print(f"XTTS chargé sur {DEVICE}")
    return tts_model

@app.get("/status")
def status():
    return {"loaded": tts_model is not None, "device": DEVICE, "model": "xtts-narrator-fr"}

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
        ref_path = str(REF_WAV)

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
    print(f"Modèle: {MODEL_DIR}")
    print(f"Référence: {REF_WAV}")
    if not MODEL_DIR.exists():
        print(f"ERREUR: Dossier modèle introuvable: {MODEL_DIR}")
        sys.exit(1)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
