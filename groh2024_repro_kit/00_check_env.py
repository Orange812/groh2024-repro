"""
检查电脑环境是否能运行各个部分：python 00_check_env.py
"""
import importlib
import sys

PARTS = {
    "统计分析（analysis/）": ["pandas", "numpy", "scipy", "matplotlib"],
    "演示网站（demo_app/）": ["flask", "pandas"],
    "AI 训练/推理（dls/，可选）": ["torch", "torchvision", "PIL"],
    "作者官方 notebook（可选）": ["notebook", "statsmodels", "seaborn", "stargazer"],
}


def ver(name):
    try:
        m = importlib.import_module(name)
        try:
            from importlib.metadata import version
            return version({"PIL": "pillow"}.get(name, name))
        except Exception:  # noqa: BLE001
            return getattr(m, "__version__", "ok")
    except ImportError:
        return None


print(f"Python {sys.version.split()[0]}  ({sys.executable})")
if sys.version_info < (3, 9):
    print("  ⚠ 建议使用 Python 3.9 或更高版本")
for part, mods in PARTS.items():
    res = {m: ver(m) for m in mods}
    ok = all(res.values())
    print(f"\n{'✓' if ok else '✗'} {part}")
    for m, v in res.items():
        print(f"    {m:<12} {v if v else '缺少 → pip install ' + ('pillow' if m == 'PIL' else m)}")
print("\n安装本工具包依赖：pip install -r requirements.txt")
