#!/usr/bin/env python3
# No-args runner for XGBoost → dumps JSONs into roc_dumps/compute_profiles

import sys, subprocess
from pathlib import Path

# ---- fixed paths ----
XGB_ROOT  = Path(__file__).parent
THIRD_OUT = Path("/home/umflint.edu/koernerg/roc_dumps/compute_profiles")
DF_PATH   = "/home/umflint.edu/koernerg/xgboost/content/sample_data/corrected_permacts.csv"
INDICES   = "/home/umflint.edu/koernerg/xgboost/standardized_data"

FEATURE_SETS = ["MI-25", "FI-25"]
SIZES        = ["10000", "100000", "full"]

try:
    import xgboost  # just to log version / ensure import
    DEVICES = ["cpu", "cuda"]
except Exception:
    DEVICES = ["cpu"]

THIRD_OUT.mkdir(parents=True, exist_ok=True)
script = XGB_ROOT / "profile_xgboost.py"

def run(cmd):
    print("\n$ " + " ".join(cmd) + f"   (cwd={XGB_ROOT})")
    return subprocess.call(cmd, cwd=str(XGB_ROOT))

for fs in FEATURE_SETS:
    for dev in DEVICES:
        # prefer per-size invocation; if script lacks --sizes, retry once without it (runs all sizes)
        retried_without_sizes = False
        for size in SIZES:
            cmd = [
                sys.executable, "-u", str(script),
                "--feature_set", fs,
                "--device", dev,
                "--df_path", DF_PATH,
                "--indices_dir", INDICES,
                "--out_dir", str(THIRD_OUT),
                "--single_model",
                "--sizes", size,   # if unsupported, we'll retry without --sizes once
            ]
            rc = run(cmd)
            if rc != 0 and not retried_without_sizes:
                base_cmd = [
                    sys.executable, "-u", str(script),
                    "--feature_set", fs,
                    "--device", dev,
                    "--df_path", DF_PATH,
                    "--indices_dir", INDICES,
                    "--out_dir", str(THIRD_OUT),
                    "--single_model",
                ]
                print(f"⚠️ sizes arg failed (rc={rc}); retrying once without --sizes to run all sizes")
                run(base_cmd)
                retried_without_sizes = True

print("\n[XGB] done → JSONs in", THIRD_OUT)
