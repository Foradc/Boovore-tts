@echo off
echo === Setup Kitten TTS 2 ===
rem Environnement a part : Kitten TTS 2 demande transformers 5.x, Qwen3-TTS reste en 4.57.
rem Hors du dossier du projet pour ne pas synchroniser 1,5 Go (OneDrive, Dropbox...).
if "%KITTEN_ENV%"=="" set KITTEN_ENV=%LOCALAPPDATA%\boovore\kitten-env
set KPY="%KITTEN_ENV%\Scripts\python.exe"
py -3.12 -m venv "%KITTEN_ENV%"
if errorlevel 1 (echo ECHEC : impossible de creer l'environnement - Python 3.12 est-il installe ? & pause & exit /b 1)
%KPY% -m pip install --upgrade pip
rem PyTorch CPU : sur une carte de 6 Go, Qwen3 garde le GPU. Avec 8 Go libres ou plus,
rem remplacer l'index par https://download.pytorch.org/whl/cu126 et lancer avec KITTEN_DEVICE=cuda.
%KPY% -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (echo ECHEC : installation de PyTorch & pause & exit /b 1)
%KPY% -m pip install kittenml fastapi uvicorn python-multipart
if errorlevel 1 (echo ECHEC : installation de kittenml & pause & exit /b 1)
echo === Setup termine : %KITTEN_ENV% ===
echo Lancez start_kitten.bat. Le premier lancement telecharge le modele, environ 1,5 Go.
pause
