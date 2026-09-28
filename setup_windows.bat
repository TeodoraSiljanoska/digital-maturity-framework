@echo off
cd /d "%~dp0"
echo Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
  echo Python was not found. Install Python 3.10-3.12 from python.org and tick Add to PATH.
  pause
  exit /b 1
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install python-docx openpyxl
echo.
echo Setup finished. Next: run_dashboard.bat
pause
