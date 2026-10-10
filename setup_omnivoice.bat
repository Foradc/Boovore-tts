@echo off
echo === Setup OmniVoice ===
rem Environnement a part : OmniVoice demande transformers 5.x, Qwen3-TTS reste en 4.57.
rem Si OmniVoice est deja installe ailleurs, inutile : mettre OMNIVOICE_ENV dans local_env.bat.
if "%OMNIVOICE_ENV%"=="" set OMNIVOICE_ENV=%LOCALAPPDATA%\boovore\omnivoice-env
set OPY="%OMNIVOICE_ENV%\Scripts\python.exe"
py -3.12 -m venv "%OMNIVOICE_ENV%"
if errorlevel 1 (echo ECHEC : impossible de creer l'environnement - Python 3.12 est-il installe ? & pause & exit /b 1)
%OPY% -m pip install --upgrade pip
rem PyTorch CUDA 12.6 : OmniVoice tient dans environ 2 Go de carte graphique
%OPY% -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
if errorlevel 1 (echo ECHEC : installation de PyTorch & pause & exit /b 1)
%OPY% -m pip install omnivoice fastapi uvicorn python-multipart
if errorlevel 1 (echo ECHEC : installation d'OmniVoice & pause & exit /b 1)
echo === Setup termine : %OMNIVOICE_ENV% ===
echo Lancez start_omnivoice.bat. La premiere generation telecharge le modele, environ 3 Go.
pause
