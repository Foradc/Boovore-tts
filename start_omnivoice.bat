@echo off
title Boovore - OmniVoice
cd /d "%~dp0"
if exist "%~dp0local_env.bat" call "%~dp0local_env.bat"
if "%OMNIVOICE_ENV%"=="" set OMNIVOICE_ENV=%LOCALAPPDATA%\boovore\omnivoice-env
rem Pas de bloc entre parentheses : un ")" dans le chemin le casserait
if not exist "%OMNIVOICE_ENV%\Scripts\python.exe" echo Environnement OmniVoice introuvable dans %OMNIVOICE_ENV% : lancez setup_omnivoice.bat d'abord.
if not exist "%OMNIVOICE_ENV%\Scripts\python.exe" pause
if not exist "%OMNIVOICE_ENV%\Scripts\python.exe" exit /b 1
set PYTHONUTF8=1
"%OMNIVOICE_ENV%\Scripts\python.exe" omnivoice_server.py
pause
