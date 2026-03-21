@echo off
echo === Setup XTTS Narrateur FR ===
py -3.11 -m venv xtts-env
call xtts-env\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install TTS==0.22.0 transformers==4.39.3 fastapi uvicorn
echo === Setup terminé ===
echo Lancez start_xtts.bat pour démarrer le serveur
pause
