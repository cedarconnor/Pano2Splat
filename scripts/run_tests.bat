@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
"%SCRIPT_DIR%..\tests\run_with_vs.bat" -m pytest %*
