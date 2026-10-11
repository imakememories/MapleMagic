@echo off
rem Play Ognis against an agent in the browser: double-click, then use the page that opens.
rem Add --host 0.0.0.0 to play from another device on your network.
cd /d %~dp0..
start "" http://127.0.0.1:8765
"C:\Users\manni\miniconda3\envs\rocm_env\python.exe" -m mymagezero.cli web %*
pause
