@echo off
title Boovore
echo === Boovore en local (RTX 2060 6 Go) ===
cd /d "%~dp0"

rem Dans ce venv : Qwen3 (carte graphique), Kokoro et Pocket TTS (legers). Chatterbox, F5, Fish et
rem Voxtral tournent sur le serveur GPU loue. Kitten et OmniVoice sont ajoutes plus bas s'ils sont installes.
set ENABLED_ENGINES=qwen3,kokoro,pocket
rem 6 Go de VRAM : un seul modele en memoire, en charger un autre decharge le precedent
set MODEL_CACHE_SIZE=1
rem Les 4 modeles deja telecharges (1.7B-CustomVoice ferait telecharger 4 Go de plus)
set ACTIVE_MODELS=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign,Qwen/Qwen3-TTS-12Hz-1.7B-Base,Qwen/Qwen3-TTS-12Hz-0.6B-Base,Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
rem Transcription auto de la voix de reference (parakeet) : pas de place a cote d'un 1.7B
set TRANSCRIBE_DEVICE=off
set PYTHONUTF8=1

rem Moteurs dans leur propre environnement, chacun dans sa fenetre (a fermer aussi en quittant)
if exist "%~dp0local_env.bat" call "%~dp0local_env.bat"
if "%KITTEN_ENV%"=="" set KITTEN_ENV=%LOCALAPPDATA%\boovore\kitten-env
if "%OMNIVOICE_ENV%"=="" set OMNIVOICE_ENV=%LOCALAPPDATA%\boovore\omnivoice-env
rem Kitten sur le processeur pour laisser la carte a Qwen3 / OmniVoice
set KITTEN_DEVICE=cpu
if exist "%KITTEN_ENV%\Scripts\python.exe" set ENABLED_ENGINES=%ENABLED_ENGINES%,kitten
if exist "%KITTEN_ENV%\Scripts\python.exe" start "Boovore - Kitten TTS 2" /min "%~dp0start_kitten.bat"
rem OmniVoice prend la carte quand on l'utilise : Boovore decharge alors Qwen3, et inversement
if exist "%OMNIVOICE_ENV%\Scripts\python.exe" set ENABLED_ENGINES=%ENABLED_ENGINES%,omnivoice
if exist "%OMNIVOICE_ENV%\Scripts\python.exe" start "Boovore - OmniVoice" /min "%~dp0start_omnivoice.bat"

rem Modele charge au demarrage (autre choix : start_qwen3_local.bat Qwen/Qwen3-TTS-12Hz-1.7B-Base)
set MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
if not "%~1"=="" set MODEL=%~1

rem Ouvre le navigateur des que le serveur repond (le chargement prend ~30-60 s)
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "while (-not (Test-NetConnection 127.0.0.1 -Port 7860 -InformationLevel Quiet -WarningAction SilentlyContinue)) { Start-Sleep 2 }; Start-Process http://localhost:7860"

venv\Scripts\python.exe server.py --host 127.0.0.1 --port 7860 --model %MODEL%
pause
