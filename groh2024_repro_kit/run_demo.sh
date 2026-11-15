#!/usr/bin/env bash
# macOS / Linux：bash run_demo.sh
set -e
cd "$(dirname "$0")"
echo "[1/2] 生成模拟数据并运行分析（彩排用）..."
[ -f data/synthetic_experiment.csv ] || python3 analysis/make_synthetic_data.py
python3 analysis/key_results.py --data data/synthetic_experiment.csv --out outputs/synthetic
echo
echo "[2/2] 启动实验平台复刻版：浏览器打开 http://127.0.0.1:5000 （实时汇总 /dashboard）"
( sleep 1.5; (open http://127.0.0.1:5000 2>/dev/null || xdg-open http://127.0.0.1:5000 2>/dev/null) ) &
python3 demo_app/app.py
