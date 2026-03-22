# 🎙 Boovore — Multi-Engine TTS Studio

**Boovore** is a self-hosted, GPU-accelerated Text-to-Speech studio with 5 best-in-class engines and a built-in audiobook generator. Run it on any CUDA machine (tested on RTX 3090) via a clean, dark-mode web UI.

> **Nom** : Boovore = *Book* + *Dévorer* — pour dévorer les livres en audio.

---

## ✨ Engines intégrés

| Engine | Qualité | Vitesse | Particularité |
|---|---|---|---|
| **Kokoro FR** | ★★★★ | ⚡⚡⚡ | Voix françaises natives |
| **Chatterbox** | ★★★★★ | ⚡⚡ | Zero-shot voice cloning (ResembleAI) |
| **F5-TTS FR** | ★★★★ | ⚡⚡ | Clonage vocal français |
| **Fish-Speech 1.5** | ★★★★★ | ⚡⚡ | Voice cloning multilingue (fishaudio) |
| **Qwen3-TTS** | ★★★★★ | ⚡ | Clone · Custom · Voice Design |

---

## 🚀 Démarrage rapide (Vast.ai / serveur GPU)

### 1. Cloner et installer les dépendances

```bash
# PyTorch nightly CUDA 12.8 (requis)
pip3 install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# ldconfig pour que torchaudio trouve libtorch
echo "/usr/local/lib/python3.12/dist-packages/torch/lib" > /etc/ld.so.conf.d/torch.conf && ldconfig

# Engines
pip3 install faster-qwen3-tts kokoro f5-tts fastapi uvicorn[standard] python-multipart

# Chatterbox (Python 3.12)
pip3 install conformer==0.3.2 --no-build-isolation
git clone https://github.com/resemble-ai/chatterbox /tmp/chatterbox
cd /tmp/chatterbox && pip3 install -e . --no-deps && cd /root

# Fish-Speech 1.5
git clone https://github.com/fishaudio/fish-speech /tmp/fish-speech
cd /tmp/fish-speech && git checkout v1.5.1
pip3 install -e . --no-deps
huggingface-cli download fishaudio/fish-speech-1.5 --local-dir /root/fish-speech-model
```

### 2. Lancer le serveur

```bash
nohup python3 server.py --port 7860 >> /root/server.log 2>&1 &
```

### 3. Accéder à l'UI

```bash
# Tunnel SSH local
ssh -p <PORT> root@<HOST> -L 7860:localhost:7860 -N
# Ouvrir http://localhost:7860
```

---

## 📖 Fonctionnalités

- **Studio TTS** : sélecteur de moteur en un clic (7 pills), génération centrale
- **Livre Audio** : importe `.txt` / `.pdf` / `.epub`, découpe en chapitres, génère en batch avec n'importe quel moteur, télécharge chapitre par chapitre ou fusionne en 1 WAV
- **Voice Cloning** : upload d'un clip de référence audio (Chatterbox, F5-TTS, Fish-Speech, Qwen3)
- **Métriques temps réel** : TTFA, RTF, durée, buffer
- **Thème clair/sombre**
- **Streaming audio** (Qwen3) avec CUDA Graphs

---

## 🗂 Structure

```
server.py       — FastAPI backend (5 engines)
index.html      — UI single-page (vanilla JS, aucune dépendance frontend)
requirements.txt
Dockerfile
```

---

## ⚙️ Requirements

- Python 3.12+
- CUDA 12.8 (RTX 3090 ou supérieur recommandé)
- PyTorch nightly cu128 (`2.12.0.dev+`)
- RAM GPU : 8 GB minimum, 24 GB pour tous les modèles en même temps

---

## 📦 Modèles téléchargés automatiquement

| Modèle | Taille | Engine |
|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | ~1.2 GB | Qwen3-TTS |
| `hexgrad/Kokoro-82M` | ~300 MB | Kokoro FR |
| `SWivid/F5-TTS` | ~1.2 GB | F5-TTS |
| `resemble-ai/chatterbox` | ~1.5 GB | Chatterbox |
| `fishaudio/fish-speech-1.5` | ~1.4 GB | Fish-Speech |

---

## 🏷️ Topics

`text-to-speech` `tts` `voice-cloning` `audiobook` `french-tts` `kokoro` `f5-tts` `fish-speech` `chatterbox` `qwen3` `fastapi` `cuda` `self-hosted` `gpu` `french` `multilingual`

---

## Crédits

- [faster-qwen3-tts](https://github.com/huggingfaceM4/faster-qwen3-tts) — Qwen3-TTS engine
- [Fish-Speech](https://github.com/fishaudio/fish-speech) — fishaudio
- [Chatterbox](https://github.com/resemble-ai/chatterbox) — ResembleAI
- [F5-TTS](https://github.com/SWivid/F5-TTS) — SWivid
- [Kokoro](https://github.com/hexgrad/kokoro) — hexgrad

---

MIT License
