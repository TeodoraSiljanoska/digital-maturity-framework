@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate
set PYTHONPATH=src
echo Starting Streamlit. Keep this window open. Browser should open automatically.
streamlit run src\dashboard\app.py
