from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "evaluation" / "metrics"
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

def load_csv(name):
    path = METRICS / name
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()

def save_empty(title, filename, note):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.text(0.5, 0.5, note, ha="center", va="center", wrap=True)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=160)
    plt.close(fig)

ocr = load_csv("ocr_metrics.csv")
if ocr.empty:
    save_empty("OCR CER and WER", "ocr_cer_wer.png", "No OCR metric rows available.")
else:
    for metric in ("cer", "wer"):
        if metric not in ocr.columns:
            continue
        data = ocr.copy()
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
        data = data.dropna(subset=[metric])
        if data.empty:
            continue
        grouping = [c for c in ("preprocessing_variant", "ocr_engine") if c in data.columns]
        if grouping:
            summary = data.groupby(grouping, dropna=False)[metric].mean().reset_index()
            summary["configuration"] = summary[grouping].astype(str).agg(" / ".join, axis=1)
        else:
            summary = data[[metric]].copy()
            summary["configuration"] = "all"
        fig, ax = plt.subplots(figsize=(12, 6))
        summary.plot.bar(x="configuration", y=metric, ax=ax, legend=False)
        ax.set_title(f"Mean OCR {metric.upper()} by preprocessing and engine")
        ax.set_ylabel(metric.upper())
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=60)
        fig.tight_layout()
        fig.savefig(OUT / f"ocr_{metric}_by_configuration.png", dpi=160)
        plt.close(fig)

translation = load_csv("translation_metrics.csv")
for metric in ("bleu", "chrf"):
    if translation.empty or not {"target_language", metric}.issubset(translation.columns):
        save_empty(f"{metric.upper()} by target language", f"translation_{metric}_by_language.png",
                   "No aligned reference scores available; no scores were fabricated.")
        continue
    translation[metric] = pd.to_numeric(translation[metric], errors="coerce")
    valid = translation.dropna(subset=[metric])
    if valid.empty:
        save_empty(f"{metric.upper()} by target language", f"translation_{metric}_by_language.png",
                   "No valid scores: aligned human reference translations are unavailable.")
        continue
    summary = valid.groupby("target_language")[metric].mean().sort_index()
    ax = summary.plot.bar(figsize=(12, 6))
    ax.set_title(f"Mean {metric.upper()} by target language")
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("Target language")
    ax.tick_params(axis="x", labelrotation=45)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(OUT / f"translation_{metric}_by_language.png", dpi=160)
    plt.close(fig)

errors_path = ROOT / "evaluation" / "error_logs" / "ocr_errors.csv"
errors = pd.read_csv(errors_path) if errors_path.is_file() else pd.DataFrame()
if not errors.empty and "category" in errors.columns:
    counts = errors["category"].fillna("unknown").value_counts()
    fig, ax = plt.subplots(figsize=(9, 7))
    counts.plot.pie(ax=ax, autopct="%1.1f%%", ylabel="")
    ax.set_title("OCR error distribution by category")
    fig.tight_layout()
    fig.savefig(OUT / "ocr_error_distribution.png", dpi=160)
    plt.close(fig)
else:
    save_empty("OCR error distribution", "ocr_error_distribution.png",
               "No categorized OCR errors available.")

print("Generated figures:")
for path in sorted(OUT.glob("*.png")):
    print(path.relative_to(ROOT))
