"""
Check whether this computer can run each part of the toolkit: python 00_check_env.py
"""
import importlib
import sys

PARTS = {
    "Statistical analysis (analysis/)": ["pandas", "numpy", "scipy", "matplotlib"],
    "Demo website (demo_app/)": ["flask", "pandas"],
    "AI training/inference (dls/, optional)": ["torch", "torchvision", "PIL"],
    "Authors' official notebook (optional)": ["notebook", "statsmodels", "seaborn", "stargazer"],
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
    print("  ⚠ Python 3.9 or later is recommended")
for part, mods in PARTS.items():
    res = {m: ver(m) for m in mods}
    ok = all(res.values())
    print(f"\n{'✓' if ok else '✗'} {part}")
    for m, v in res.items():
        print(f"    {m:<12} {v if v else 'missing → pip install ' + ('pillow' if m == 'PIL' else m)}")
print("\nInstall this toolkit's dependencies: pip install -r requirements.txt")
