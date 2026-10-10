@echo off
title Boovore - Kitten TTS 2
cd /d "%~dp0"
if exist "%~dp0local_env.bat" call "%~dp0local_env.bat"
if "%KITTEN_ENV%"=="" set KITTEN_ENV=%LOCALAPPDATA%\boovore\kitten-env
rem Pas de bloc entre parentheses : un ")" dans le chemin le casserait
if not exist "%KITTEN_ENV%\Scripts\python.exe" echo Environnement Kitten introuvable dans %KITTEN_ENV% : lancez setup_kitten.bat d'abord.
if not exist "%KITTEN_ENV%\Scripts\python.exe" pause
if not exist "%KITTEN_ENV%\Scripts\python.exe" exit /b 1
set PYTHONUTF8=1
"%KITTEN_ENV%\Scripts\python.exe" kitten_server.py
pause
