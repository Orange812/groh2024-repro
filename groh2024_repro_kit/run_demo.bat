@echo off
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo [1/2] 生成模拟数据并运行分析（彩排用）...
if not exist data\synthetic_experiment.csv python analysis\make_synthetic_data.py
python analysis\key_results.py --data data\synthetic_experiment.csv --out outputs\synthetic
echo.
echo [2/2] 启动实验平台复刻版：浏览器打开 http://127.0.0.1:5000  （实时汇总 /dashboard）
start "" http://127.0.0.1:5000
python demo_app\app.py
pause
