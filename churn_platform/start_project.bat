@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating Python 3.11 virtual environment...
    py -3.11 -m venv .venv
    if errorlevel 1 goto :failed
)

".venv\Scripts\python.exe" scripts\start_project.py
if errorlevel 1 goto :failed
goto :eof

:failed
echo.
echo Project startup failed. Review the error above and confirm Python 3.11 is installed.
pause
exit /b 1
