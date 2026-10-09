#!/usr/bin/env python3
"""
voxtral_server.py — Start, stop, and check Voxtral TTS inference server.

Voxtral TTS uses vLLM-Omni for self-hosted inference.
Model: mistralai/Voxtral-4B-TTS-2603 (~8GB BF16, fits RTX 3090 24GB)

Usage on Vast.ai:
  python3 /workspace/voxtral_server.py           # start server
  python3 /workspace/voxtral_server.py --status  # check if running
  python3 /workspace/voxtral_server.py --stop    # kill server

Once started, inference endpoint is at: http://localhost:8000/v1/audio/speech
"""

import os
import sys
import time
import subprocess
import signal
from pathlib import Path

MODEL_ID = "mistralai/Voxtral-4B-TTS-2603"
VLLM_PORT = 8000
VLLM_HOST = "0.0.0.0"
PID_FILE = Path("/tmp/voxtral_vllm.pid")
LOG_FILE = Path("/tmp/voxtral_vllm.log")
STARTUP_TIMEOUT = 300  # 5 min for model download + load


def check_dependencies():
    """Install required packages if missing."""
    missing = []
    try:
        import vllm  # noqa: F401
    except ImportError:
        missing.append("vllm[audio]>=0.18.0")
    try:
        import httpx  # noqa: F401
    except ImportError:
        missing.append("httpx")
    try:
        import soundfile  # noqa: F401
    except ImportError:
        missing.append("soundfile")

    if missing:
        print(f"Installing: {', '.join(missing)}")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing,
            check=True
        )
        # vllm-omni must be installed separately (pip package)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "vllm-omni>=0.18.0"],
            check=False  # may not exist as standalone package yet
        )
        print("Dependencies installed.")


def is_running():
    """Return True if vLLM server is already up and responding."""
    try:
        import httpx
        r = httpx.get(f"http://localhost:{VLLM_PORT}/health", timeout=3.0)
        return r.status_code == 200
    except Exception:
        return False


def get_pid():
    """Return PID from pid file, or None if not found."""
    if PID_FILE.exists():
        try:
            return int(PID_FILE.read_text().strip())
        except ValueError:
            pass
    return None


def check_hf_token():
    """Warn if HF_TOKEN is not set — required to download Mistral gated models."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN env var not set.")
        print("  Mistral models require a HuggingFace token to download.")
        print("  Steps:")
        print("    1. Go to https://huggingface.co/mistralai/Voxtral-4B-TTS-2603")
        print("    2. Accept the CC BY-NC license")
        print("    3. Get your token at https://huggingface.co/settings/tokens")
        print("    4. Set it: export HF_TOKEN=hf_xxxx")
        print("  Or pass directly: HF_TOKEN=hf_xxxx python3 voxtral_server.py")
        print()
    return token


def start():
    """Start vLLM-Omni server for Voxtral TTS."""
    if is_running():
        print(f"Server already running on port {VLLM_PORT}")
        return True

    check_dependencies()
    hf_token = check_hf_token()

    env = os.environ.copy()
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token  # vLLM reads both

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_ID,
        "--omni",
        "--host", VLLM_HOST,
        "--port", str(VLLM_PORT),
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
    ]

    print(f"Starting Voxtral TTS server...")
    print(f"  Model: {MODEL_ID}")
    print(f"  Port:  {VLLM_PORT}")
    print(f"  Log:   {LOG_FILE}")
    print(f"  HF_TOKEN: {'set' if hf_token else 'NOT SET (download may fail)'}")
    print(f"  (First run will download ~8GB model weights)")

    log_fd = open(LOG_FILE, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fd,
        stderr=log_fd,
        env=env,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))
    print(f"  PID:   {proc.pid}")

    # Wait for server to be ready
    print(f"Waiting for server (up to {STARTUP_TIMEOUT}s)...", end="", flush=True)
    deadline = time.time() + STARTUP_TIMEOUT
    while time.time() < deadline:
        if is_running():
            print(f" ready!")
            print(f"\nVoxtral TTS ready at http://localhost:{VLLM_PORT}/v1/audio/speech")
            return True
        print(".", end="", flush=True)
        time.sleep(5)

    print(f" TIMEOUT after {STARTUP_TIMEOUT}s")
    print(f"Check log: {LOG_FILE}")
    return False


def status():
    """Print server status."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if is_running():
        pid = get_pid()
        print(f"Server running on port {VLLM_PORT}" + (f" (PID {pid})" if pid else ""))
        print(f"  Endpoint: http://localhost:{VLLM_PORT}/v1/audio/speech")
        print(f"  Health:   http://localhost:{VLLM_PORT}/health")
    else:
        print(f"Server not running on port {VLLM_PORT}")
        pid = get_pid()
        if pid:
            print(f"  (Stale PID file: {pid} — process may have crashed)")
            print(f"  Check log: {LOG_FILE}")
    print(f"  HF_TOKEN: {'set' if token else 'NOT SET'}")


def stop():
    """Stop vLLM server."""
    pid = get_pid()
    if pid:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            PID_FILE.unlink(missing_ok=True)
            print(f"Sent SIGTERM to process group of PID {pid}")
        except ProcessLookupError:
            print(f"Process {pid} not found (already stopped)")
            PID_FILE.unlink(missing_ok=True)
    elif is_running():
        print("Server is running but PID file not found.")
        print(f"Kill manually: kill $(lsof -t -i:{VLLM_PORT})")
    else:
        print("Server is not running.")


if __name__ == "__main__":
    if "--status" in sys.argv:
        status()
    elif "--stop" in sys.argv:
        stop()
    else:
        success = start()
        sys.exit(0 if success else 1)
