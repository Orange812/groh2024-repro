@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo [1/2] Generating simulated data and running the analysis (rehearsal only)...
if not exist data\synthetic_experiment.csv python analysis\make_synthetic_data.py
python analysis\key_results.py --data data\synthetic_experiment.csv --out outputs\synthetic
echo.
echo [2/2] Starting the experiment-platform replica: open http://127.0.0.1:5000 in a browser (live summary at /dashboard)
start "" http://127.0.0.1:5000
python demo_app\app.py
pause
