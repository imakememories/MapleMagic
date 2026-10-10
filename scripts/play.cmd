@echo off
rem Play Ognis against the best network: double-click, or run with options, e.g. play.cmd --seat 0 --sims 800
cd /d %~dp0..
"C:\Users\manni\miniconda3\envs\rocm_env\python.exe" -m mymagezero.cli play %*
pause
