#!/usr/bin/env python3
"""
make_narrator_reference.py — Generate the canonical narrator reference audio.

Generates a 12-15s French narrator sample using Qwen3-TTS-VoiceDesign with the
optimized NARRATOR_PROMPT. This WAV is used as the voice clone reference for
Voxtral TTS on all subsequent books.

Run on Vast.ai after the Qwen3 server (server.py) is loaded:
  python3 /workspace/make_narrator_reference.py

Output: /workspace/narrator_reference.wav
"""

import requests
import base64
import struct
import time
from pathlib import Path

SERVER = "http://localhost:7860"
OUTPUT = Path("/workspace/narrator_reference.wav")

# Reference text: varied French narration covering different prosodic patterns.
# ~14s at normal audiobook pace (~130 words/min).
REFERENCE_TEXT = (
    "La nuit tombait lentement sur la plaine silencieuse. "
    "Au loin, les lumières d'un village clignotaient dans l'obscurité. "
    "«\u00a0Il faut partir maintenant\u00a0», murmura-t-il, la voix grave et posée. "
    "Il prit sa veste, jeta un dernier regard sur la pièce... "
    "et ferma la porte sans bruit."
)

NARRATOR_PROMPT = (
    "Narrateur professionnel de livres audio français, voix grave, chaude et captivante. "
    "Débit naturellement mesuré — ni précipité ni traînant. "
    "Respectez scrupuleusement la ponctuation : "
    "légère pause aux virgules, souffle marqué aux points, "
    "pause longue et respirée aux doubles sauts de paragraphe. "
    "Aux guillemets « », adoptez un registre légèrement plus direct et personnel pour le dialogue, "
    "puis revenez au ton narratif après le ». "
    "Les points de suspension (...) et les tirets (—) appellent une vraie pause respiratoire. "
    "Ton légèrement plus grave et plus riche que la conversation ordinaire, "
    "comme un conteur au coin du feu. "
    "Restez cohérent du début à la fin — même timbre, même rythme de fond."
)

SEED = 42069
MIN_DURATION = 10.0  # seconds


def wait_server(timeout=120):
    print("Waiting for Qwen3 server...", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{SERVER}/status", timeout=5)
            d = r.json()
            if d.get("loaded") and not d.get("loading"):
                print(f" ready ({d['model']})")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(5)
    print(" TIMEOUT")
    return False


def generate():
    data = {
        "text": REFERENCE_TEXT,
        "language": "French",
        "mode": "voice_design",
        "instruct": NARRATOR_PROMPT,
        "temperature": "0.65",
        "top_k": "30",
        "repetition_penalty": "1.1",
        "seed": str(SEED),
    }
    r = requests.post(f"{SERVER}/generate", data=data, timeout=120)
    r.raise_for_status()
    d = r.json()
    return base64.b64decode(d["audio_b64"])


def wav_duration(wav_bytes):
    try:
        sr = struct.unpack_from("<I", wav_bytes, 24)[0]
        ch = struct.unpack_from("<H", wav_bytes, 22)[0]
        bits = struct.unpack_from("<H", wav_bytes, 34)[0]
        data_size = struct.unpack_from("<I", wav_bytes, 40)[0]
        return data_size / (sr * ch * bits // 8)
    except Exception:
        return 0.0


if __name__ == "__main__":
    if not wait_server():
        print("ERROR: Qwen3 server not ready")
        exit(1)

    print(f"\nGenerating narrator reference audio...")
    print(f"Text ({len(REFERENCE_TEXT)} chars): {REFERENCE_TEXT[:80]}...")

    for attempt in range(3):
        wav = generate()
        dur = wav_duration(wav)
        print(f"  Attempt {attempt + 1}: {dur:.1f}s, {len(wav) // 1024}KB")
        if dur >= MIN_DURATION:
            OUTPUT.write_bytes(wav)
            print(f"\n✓ Saved: {OUTPUT} ({dur:.1f}s)")
            print(f"  Upload locally: ssh root@HOST -p PORT 'cat {OUTPUT}' > tts-projects/narrator_reference.wav")
            print(f"  Then use with Voxtral via generate_audiobook_voxtral.py")
            break
        print(f"  Too short (< {MIN_DURATION}s), retrying...")
    else:
        print(f"FAILED: Could not generate reference audio >= {MIN_DURATION}s after 3 attempts")
        exit(1)
