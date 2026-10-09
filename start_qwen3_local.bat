@echo off
title Boovore - Qwen3-TTS local
echo === Boovore : Qwen3-TTS en local (RTX 2060 6 Go) ===
cd /d "%~dp0"

rem Seul Qwen3 est installe dans ce venv ; les autres moteurs tournent sur le serveur GPU
set ENABLED_ENGINES=qwen3
rem 6 Go de VRAM : un seul modele en memoire, en charger un autre decharge le precedent
set MODEL_CACHE_SIZE=1
rem Les 4 modeles deja telecharges (1.7B-CustomVoice ferait telecharger 4 Go de plus)
set ACTIVE_MODELS=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign,Qwen/Qwen3-TTS-12Hz-1.7B-Base,Qwen/Qwen3-TTS-12Hz-0.6B-Base,Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
rem Transcription auto de la voix de reference (parakeet) : pas de place a cote d'un 1.7B
set TRANSCRIBE_DEVICE=off
set PYTHONUTF8=1

rem Modele charge au demarrage (autre choix : start_qwen3_local.bat Qwen/Qwen3-TTS-12Hz-1.7B-Base)
set MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
if not "%~1"=="" set MODEL=%~1

rem Ouvre le navigateur des que le serveur repond (le chargement prend ~30-60 s)
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "while (-not (Test-NetConnection 127.0.0.1 -Port 7860 -InformationLevel Quiet -WarningAction SilentlyContinue)) { Start-Sleep 2 }; Start-Process http://localhost:7860"

venv\Scripts\python.exe server.py --host 127.0.0.1 --port 7860 --model %MODEL%
pause
