import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VENV = Path(r"C:\DevEnvs\indicnlp21")

print("=== Indic NLP Challenge Environment ===")
print()

print("Python:")
print(f"  Version : {sys.version.split()[0]}")
print(f"  Path    : {sys.executable}")
print()

print("Virtual environment:")
print(f"  Expected: {EXPECTED_VENV}")
print(f"  Active  : {sys.prefix}")
print(f"  Status  : {'OK' if Path(sys.prefix).resolve() == EXPECTED_VENV.resolve() else 'CHECK'}")
print()

print("Project:")
print(f"  Root    : {PROJECT_ROOT}")
print(f"  Status  : {'OK' if PROJECT_ROOT.name == 'indic-nlp-challenge' else 'CHECK'}")
print()

print("Package:")
try:
    import indicnlp_pipeline

    print("  Import  : OK")
    print(f"  Path    : {indicnlp_pipeline.__file__}")
except ImportError as exc:
    print(f"  Import  : FAILED")
    print(f"  Error   : {exc}")

print()
print("Environment verification complete.")