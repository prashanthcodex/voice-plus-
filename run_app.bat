@echo off
echo Starting VocalStrong AI Streamlit Application...
cd /d "%~dp0"
call .venv\Scripts\streamlit.exe run app.py
pause
