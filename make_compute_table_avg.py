# scripts/make_compute_table_avg.py
import json, glob
import pandas as pd
from pathlib import Path

IN_DIR = Path("./compute_profiles")
OUT_CSV = IN_DIR / "compute_profiles_avg.csv"
OUT_LATEX = IN_DIR / "compute_profiles_avg.tex"

def load_records():
    rows = []
    for p in glob.glob(str(IN_DIR / "*.json")):
        try:
            with open(p, "r") as f:
                rec = json.load(f)
        except Exception:
            continue
        # Require minimum fields
        if not all(k in rec for k in ("model","sample_size","device")):
            continue
        rows.append(rec)
    return pd.DataFrame(rows)

def main():
    df = load_records()
    if df.empty:
        print("No compute profile JSONs found in ./compute_profiles")
        return

    # Keep relevant columns (when present)
    keep = [
        "model","feature_set","sample_size","device",
        "test_time_s","throughput_apps_per_s",
        "cpu_max_rss_mb","peak_vram_mb","model_bytes"
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Average MI-25 & FI-25 into one row per {model, sample_size, device}
    # Policy:
    #  - time / throughput: mean
    #  - memory peaks: max (conservative)
    #  - model_bytes: mean
    grouped = df.groupby(["model","sample_size","device"], as_index=False).agg({
        "test_time_s": "mean",
        "throughput_apps_per_s": "mean",
        "cpu_max_rss_mb": "max",
        "peak_vram_mb": "max",
        "model_bytes": "mean",
    })

    # Optional spread across MI/FI for transparency
    spread = (
        df.groupby(["model","sample_size","device"])
          .agg(test_time_s_std=("test_time_s","std"),
               throughput_std=("throughput_apps_per_s","std"))
          .reset_index()
    )
    out = grouped.merge(spread, on=["model","sample_size","device"], how="left")

    out.sort_values(["model","sample_size","device"], inplace=True)
    IN_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote averaged CSV → {OUT_CSV}")

    # Quick LaTeX (edit columns as needed)
    cols = ["model","sample_size","device","test_time_s","test_time_s_std",
            "throughput_apps_per_s","throughput_std","cpu_max_rss_mb",
            "peak_vram_mb","model_bytes"]
    have = [c for c in cols if c in out.columns]
    OUT_LATEX.write_text(out[have].to_latex(index=False, float_format="%.3f"))
    print(f"Wrote LaTeX table → {OUT_LATEX}")

if __name__ == "__main__":
    main()
