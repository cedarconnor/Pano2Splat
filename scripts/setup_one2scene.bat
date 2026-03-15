@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "BASH_EXE="

if exist "C:\Program Files\Git\bin\bash.exe" set "BASH_EXE=C:\Program Files\Git\bin\bash.exe"
if not defined BASH_EXE if exist "C:\Program Files\Git\usr\bin\bash.exe" set "BASH_EXE=C:\Program Files\Git\usr\bin\bash.exe"

if not defined BASH_EXE (
    echo [ERROR] Git Bash not found.
    echo         Install Git for Windows or run scripts\setup_one2scene.sh from a working bash shell.
    exit /b 1
)

"%BASH_EXE%" "%SCRIPT_DIR%setup_one2scene.sh" %*
