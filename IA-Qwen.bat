@echo off
cd /d "%~dp0"
python -m streamlit run IA-Qwen.py --server.address 127.0.0.1 --server.port 8501
pause