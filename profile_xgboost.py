# profile_xgboost.py
# Fair-comparison profiler for XGBoost.
# Loads the *same* models and indices as the PR plotting script and dumps
# per-size compute-profile JSONs to ./compute_profiles for the table builder.

import argparse, os, sys, time, json, shlex, subprocess, threading, queue, csv
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import joblib

try:
    import pynvml
except Exception:
    pynvml = None

try:
    import cupy as cp  # optional, for true GPU path
except Exception:
    cp = None

# ---------- Constants ----------
POLL_SECS = 0.5
IDX_DIR_DEFAULT = "../../data/splits"
OUT_DIR_DEFAULT = "../../results/figures/compute_profiles"
CSV_NAME = "compute_profiles_summary.csv"

# ---------- Model paths for each feature type (glob patterns) ----------
MODEL_PATHS = {
    "MI-25": {
        "10000": "../../results/xgboost/xgboost_ensemble_mi-25_10000_run_*.joblib",
        "100000": "../../results/xgboost/xgboost_ensemble_mi-25_100000_run_*.joblib",
        "full": "../../results/xgboost/xgboost_ensemble_mi-25_full_run_*.joblib",
    },
    "FI-25": {
        "10000": "../../results/xgboost/xgboost_ensemble_fi-25_10000_run_*.joblib",
        "100000": "../../results/xgboost/xgboost_ensemble_fi-25_100000_run_*.joblib",
        "full": "../../results/xgboost/xgboost_ensemble_fi-25_full_run_*.joblib",
    },
}

# ---------- Optional deps ----------
try:
    import psutil
except Exception:
    psutil = None

def _poll_vram(stop_event, max_used_ref):
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    while not stop_event.is_set():
        used = pynvml.nvmlDeviceGetMemoryInfo(handle).used
        max_used_ref[0] = max(max_used_ref[0], used)
        time.sleep(0.05)


# ---------- Helpers ----------
def _gpu_mem_mb_for_pids(pids):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", errors="ignore").strip().splitlines()
    except Exception:
        return 0
    want = set(int(p) for p in pids if str(p).isdigit())
    total = 0
    for line in out:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
            mem = int(parts[1])  # MB
        except Exception:
            continue
        if pid in want:
            total += mem
    return int(total)


def _ram_rss_mb_for_pids(pids):
    if psutil is None:
        return 0
    total = 0
    for pid in pids:
        try:
            p = psutil.Process(pid)
            total += p.memory_info().rss
        except Exception:
            pass
    return int(total / (1024 * 1024))


def _proc_tree_pids(root_pid):
    if psutil is None:
        return [root_pid]
    try:
        root = psutil.Process(root_pid)
        children = root.children(recursive=True)
        return [root_pid] + [c.pid for c in children]
    except Exception:
        return [root_pid]


def _tail_proc_stdout(proc, sink_q):
    for line in iter(proc.stdout.readline, ""):
        sys.stdout.write(line)
        sys.stdout.flush()
        sink_q.put(line)
    proc.stdout.close()


def _latest_file_bytes(path):
    try:
        p = Path(path)
        return p.stat().st_size if p.exists() else 0
    except Exception:
        return 0


def _append_csv(row, csv_path):
    csv_path = Path(csv_path)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _load_dataframe(df_path):
    df = pd.read_csv(df_path)
    # Match PR script behavior
    df = df.dropna(ignore_index=False)
    return df


def _auto_cols_from_pipeline(pipeline):
    """Infer original categorical & numeric columns from the ColumnTransformer."""
    pre = pipeline.named_steps["preprocessor"]
    cat_cols = None
    num_cols = None
    for name, _, cols in pre.transformers:
        if name == "cat":
            cat_cols = list(cols)
        if name == "num":
            num_cols = list(cols)
    if cat_cols is None or num_cols is None:
        try:
            cat_cols = list(pre.transformers_[0][2])
            num_cols = list(pre.transformers_[1][2])
        except Exception:
            raise RuntimeError("Could not infer cat/num columns from pipeline.")
    return cat_cols, num_cols


# ---------- Core ----------
def measure_inference(
    model_path,
    df_path,
    indices_dir,
    sample_size,
    device="cpu",
    single_model=False,
    report_auc=False,
    feature_set=None,
):
    """
    Loads a saved joblib ensemble (list of sklearn Pipelines),
    builds X_test using the pipeline's own ColumnTransformer (matching training),
    and times predict_proba. Returns a compute-profile payload.
    """
    print(f"\n[XGB-PROFILER] Loading model: {model_path}")
    try:
        models = joblib.load(model_path)
        if not isinstance(models, (list, tuple)):
            models = [models]
        if single_model:
            models = [models[0]]
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

    first = models[0]
    try:
        cat_cols, num_cols = _auto_cols_from_pipeline(first)
    except Exception as e:
        print(f"❌ Error inferring columns: {e}")
        return None

    df = _load_dataframe(df_path)
    required_cols = set(cat_cols + num_cols)
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Dataframe missing columns required by model: {missing}")

    tag = sample_size if sample_size else "full"
    test_idx_path = Path(indices_dir) / f"test_indices_{tag}.npy"
    if not test_idx_path.exists():
        raise FileNotFoundError(f"Missing test indices: {test_idx_path}")
    test_idx = np.load(test_idx_path)

    X_test = df.loc[test_idx, cat_cols + num_cols]
    y_test = df.loc[test_idx, "status"].values.astype(int)

    # Device/predictor settings
    predictor = "gpu_predictor" if device == "cuda" else "cpu_predictor"

    # --- VRAM peak tracking ---
    peak_vram_mb = 0
    baseline_vram = 0
    stop_evt = None
    max_used_ref = [0]  # Initialize outside the if block
    if device == "cuda" and pynvml is not None:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            baseline_vram = pynvml.nvmlDeviceGetMemoryInfo(handle).used
            max_used_ref[0] = baseline_vram
            stop_evt = threading.Event()
            t_vram = threading.Thread(target=_poll_vram, args=(stop_evt, max_used_ref), daemon=True)
            t_vram.start()
        except Exception:
            stop_evt = None

    # ---- Timing ----
    start = time.time()

    def _predict_pipe(pipe, X_df):
        # Transform on CPU via sklearn preprocessor
        pre = pipe.named_steps["preprocessor"]
        clf = pipe.named_steps["classifier"]
        X_trans = pre.transform(X_df)  # numpy / scipy on CPU

        # If CUDA + CuPy available: keep prediction on GPU using booster.inplace_predict
        if device == "cuda" and cp is not None:
            try:
                booster = clf.get_booster()
                X_cu = cp.asarray(X_trans)            # move features to GPU
                pred = booster.inplace_predict(X_cu)  # prediction on GPU
                # convert back to numpy if needed
                try:
                    pred = cp.asnumpy(pred)
                except Exception:
                    pass
                return pred if pred.ndim == 1 else pred[:, 1]
            except Exception:
                # fall back to sklearn API
                pass

        # CPU (or fallback) path — respect requested device/predictor if supported
        try:
            predictor = "gpu_predictor" if device == "cuda" else "cpu_predictor"
            clf.set_params(device=device, predictor=predictor)
        except Exception:
            pass
        prob = clf.predict_proba(X_trans)[:, 1]
        return prob

    if single_model:
        y_pred = _predict_pipe(first, X_test)
    else:
        preds = []
        for i, pipe in enumerate(models, 1):
            try:
                preds.append(_predict_pipe(pipe, X_test))
                print(f"✅ Model {i}/{len(models)} predicted")
            except Exception as e:
                print(f"❌ Model {i} failed: {e}")
        if not preds:
            return None
        y_pred = np.mean(preds, axis=0)

    total_infer_s = time.time() - start

    # --- stop VRAM tracking ---
    if device == "cuda" and pynvml is not None and stop_evt is not None:
        stop_evt.set()
        t_vram.join(timeout=1.0)
        try:
            peak_vram_mb = int(max(0, max_used_ref[0] - baseline_vram) / (1024**2))
        except Exception:
            peak_vram_mb = 0
    elif device == "cuda" and pynvml is not None:
        # Fallback if threading failed
        try:
            current = pynvml.nvmlDeviceGetMemoryInfo(handle).used
            peak_vram_mb = int(max(0, current - baseline_vram) / (1024**2))
        except Exception:
            peak_vram_mb = 0
    else:
        peak_vram_mb = 0

    # Nice-to-have logs
    if device == "cuda":
        print(f"🔍 Peak VRAM delta: {peak_vram_mb} MB")

    test_auc = float(roc_auc_score(y_test, y_pred)) if report_auc else None
    throughput = len(y_test) / total_infer_s if total_infer_s > 0 else None
    model_bytes = _latest_file_bytes(model_path)

    # CPU RSS of this process
    if psutil is not None:
        try:
            proc = psutil.Process(os.getpid())
            cpu_max_rss_mb = int(proc.memory_info().rss / (1024**2))
        except Exception:
            cpu_max_rss_mb = 0
    else:
        cpu_max_rss_mb = 0

    # Ensemble metadata (best-effort)
    try:
        ensemble_size = len(models)
        first_clf = first.named_steps["classifier"]
        trees_per_model = getattr(first_clf, "n_estimators", "unknown")
    except Exception:
        ensemble_size = len(models)
        trees_per_model = "unknown"

    payload = {
        "model": "XGBoost",
        "feature_set": feature_set,  # MI-25 or FI-25
        "sample_size": str(tag),
        "device": "cuda" if device == "cuda" else "cpu",
        "test_size": int(len(y_test)),
        "test_time_s": round(total_infer_s, 3),
        "throughput_apps_per_s": None if throughput is None else round(throughput, 3),
        "test_auc": test_auc,
        "peak_vram_mb": int(peak_vram_mb),
        "cpu_max_rss_mb": int(cpu_max_rss_mb),
        "model_bytes": int(model_bytes),
        "notes": {
            "ensemble_size": int(ensemble_size),
            "trees_per_model": trees_per_model,
        },
    }
    return payload


def profile_train_cmd(cmd_str, out_dir):
    """(Optional) Run a training command and capture wall time + peaks."""
    os.makedirs(out_dir, exist_ok=True)
    print("\n[XGB-PROFILER] Training command:", cmd_str)
    start = time.time()
    proc = subprocess.Popen(
        shlex.split(cmd_str),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    log_q = queue.Queue()
    t_tail = threading.Thread(target=_tail_proc_stdout, args=(proc, log_q), daemon=True)
    t_tail.start()

    peak_ram_mb = 0
    peak_vram_mb = 0
    while proc.poll() is None:
        pids = _proc_tree_pids(proc.pid)
        peak_ram_mb = max(peak_ram_mb, _ram_rss_mb_for_pids(pids))
        peak_vram_mb = max(peak_vram_mb, _gpu_mem_mb_for_pids(pids))
        time.sleep(POLL_SECS)

    wall_train_s = time.time() - start
    rec = {
        "run_label": "XGB-train",
        "dataset": "android_security",
        "feature_set": None,
        "sample_size": "full",
        "wall_train_s": round(wall_train_s, 3),
        "peak_ram_mb": int(peak_ram_mb),
        "peak_vram_mb": int(peak_vram_mb),
        "val_auc_last": None,
        "test_auc_last": None,
        "test_time_s": None,
        "test_size": None,
        "throughput_apps_per_s": None,
        "checkpoint_bytes": None,
        "checkpoint_path": None,
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = Path(out_dir) / f"XGB-train_{stamp}.json"
    with open(out_json, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"[XGB-PROFILER] Wrote: {out_json}")
    return rec


def main():
    ap = argparse.ArgumentParser(description="Profile XGBoost inference against standardized indices.")
    ap.add_argument("--feature_set", choices=["MI-25", "FI-25"], required=True,
                    help="Feature regime of the loaded pipelines (MI-25 or FI-25).")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                    help="Inference device for timing (default: cpu).")
    ap.add_argument("--single_model", action="store_true",
                    help="Use only the first pipeline in the joblib (fair single-model timing).")
    ap.add_argument("--report_auc", action="store_true", help="Also compute ROC AUC (off by default).")
    ap.add_argument("--df_path", default="../../data/raw/corrected_permacts.csv",
                    help="CSV with original columns used by the pipelines.")
    ap.add_argument("--indices_dir", default=IDX_DIR_DEFAULT,
                    help="Folder with test_indices_{size}.npy files (matches PR script).")
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT, help="Where to write compute-profile JSONs.")
    ap.add_argument("--sizes", nargs="*", choices=["10000", "100000", "full"],
                    help="Subset of sizes to run. Default = all present in MODEL_PATHS.")
    ap.add_argument("--train_cmd", type=str, help="(Optional) shell command to run training and profile.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []

    # Optional: profile training phase
    if args.train_cmd:
        rows.append(profile_train_cmd(args.train_cmd, args.out_dir))

    # Inference profiles for requested feature set
    mapping = MODEL_PATHS[args.feature_set]
    sizes = args.sizes if args.sizes else ["10000", "100000", "full"]
    
    print(f"[XGB-PROFILER] Running sizes: {sizes}")
    
    for size in sizes:
        path_pattern = mapping.get(size)
        if not path_pattern:
            print(f"⚠️  Skipping {args.feature_set} {size}: no path pattern defined")
            continue
        
        # Resolve glob pattern if present
        if '*' in path_pattern:
            import glob
            matches = glob.glob(path_pattern)
            if not matches:
                print(f"⚠️  Skipping {args.feature_set} {size}: no files matching {path_pattern}")
                continue
            path = max(matches, key=os.path.getmtime)
        else:
            path = path_pattern
        
        if not Path(path).exists():
            print(f"⚠️  Skipping {args.feature_set} {size}: model file missing → {path}")
            continue

        rec = measure_inference(
            model_path=path,
            df_path=args.df_path,
            indices_dir=args.indices_dir,
            sample_size=size,
            device=args.device,
            single_model=args.single_model,
            report_auc=args.report_auc,
            feature_set=args.feature_set,
        )
        if not rec:
            print(f"❌ No record for {args.feature_set} {size}")
            continue

        rows.append(rec)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json = Path(args.out_dir) / f"XGB-infer-{args.feature_set}-{size}_{stamp}.json"
        with open(out_json, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[XGB-PROFILER] Wrote: {out_json}")

    # Optional CSV summary (same schema you already use)
    if rows:
        csv_path = Path(args.out_dir) / CSV_NAME
        for r in rows:
            if r.get("model") == "XGBoost":
                csv_row = {
                    "run_label": f"XGB-{args.feature_set}-{r['sample_size']}",
                    "dataset": "android_security",
                    "feature_set": args.feature_set,
                    "sample_size": r["sample_size"],
                    "wall_train_s": None,
                    "peak_ram_mb": r["cpu_max_rss_mb"],
                    "peak_vram_mb": r["peak_vram_mb"],
                    "val_auc_last": None,
                    "test_auc_last": r["test_auc"],
                    "test_time_s": r["test_time_s"],
                    "test_size": r["test_size"],
                    "throughput_apps_per_s": r["throughput_apps_per_s"],
                    "checkpoint_bytes": r["model_bytes"],
                    "checkpoint_path": "",
                    "started_at": None,
                    "finished_at": None,
                }
            else:
                csv_row = r
            _append_csv(csv_row, csv_path)
        print(f"[XGB-PROFILER] Appended {len(rows)} row(s) to {csv_path}")
    else:
        print("Nothing to do. No records generated.")


if __name__ == "__main__":
    main()
