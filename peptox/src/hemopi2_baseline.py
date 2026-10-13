"""Retrain HemoPI2's Random Forest regressor on our training partition.

Reproduces the published HemoPI2 regression model with:
- HemoPI2's own featurizer (composition_calculate_hemopi2_2.py, 1167 features)
- HemoPI2's exact hyperparameters extracted from HemoPI2_reg.sav
- HemoPI2's target transform: model fits -ln(HC50), predicts exp(-y_hat) as HC50
Fits on our random-split training peptides with HemoPI2-overlap removed,
evaluates on the 135 real-HC50 test peptides.

Outputs:
  data/sota_eval/hemopi2_retrained_predictions.csv     (per-peptide predictions)
  analysis/phase1_reviewer_audit/f2_retrained.json     (metrics + Wilcoxon)
"""
from __future__ import annotations
import os, sys, json, subprocess, shutil, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from scipy.stats import wilcoxon

BASE = Path(".")
HEMOPI2 = Path("./env/lib/python3.10/site-packages/hemopi2")
FEATURIZER = HEMOPI2 / "Model" / "composition_calculate_hemopi2_2.py"
HP_TRAIN = BASE / "data" / "sota_eval" / "hemopi2_cross_val.csv"
OUT_PRED_CSV = BASE / "data" / "sota_eval" / "hemopi2_retrained_predictions.csv"
OUT_JSON = BASE / "analysis" / "phase1_reviewer_audit" / "f2_retrained.json"

# Exact hyperparameters from HemoPI2_reg.sav
RF_PARAMS = dict(
    n_estimators=800,
    max_depth=30,
    max_features=1.0,
    min_samples_split=2,
    min_samples_leaf=1,
    criterion="squared_error",
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)


def compute_features(sequences, tag):
    """Invoke HemoPI2's featurizer as a subprocess. Returns 1167-column DataFrame."""
    wd = Path(tempfile.mkdtemp(prefix=f"hp2feat_{tag}_"))
    seq_file = wd / "seq_input"
    out_file = wd / "features.csv"
    # Featurizer reads one sequence per line (no header)
    with open(seq_file, "w") as f:
        for s in sequences:
            f.write(s.upper().strip()[:40] + "\n")   # truncate at 40 (their lenchk does this)
    # Run their featurizer:  <seq_file>  <working_dir>  <output_csv>
    result = subprocess.run(
        ["python3", str(FEATURIZER), str(seq_file), str(wd), str(out_file)],
        capture_output=True, text=True, cwd=wd,
    )
    if result.returncode != 0:
        print("STDOUT:", result.stdout[-500:])
        print("STDERR:", result.stderr[-500:])
        raise RuntimeError(f"Featurizer failed for {tag}")
    if not out_file.exists():
        raise RuntimeError(f"Featurizer produced no output at {out_file}")
    X = pd.read_csv(out_file)
    shutil.rmtree(wd, ignore_errors=True)
    return X


def main():
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────
    # 1. Load our training partition (peptides with real HC50)
    # ─────────────────────────────────────────────────────────
    print("=" * 72)
    print("  Loading training partition")
    print("=" * 72)
    df = pd.read_csv(BASE / "data" / "master_v6.csv")
    seqs = df["sequence"].astype(str).str.upper().str.strip().values
    hc50 = pd.to_numeric(df["micromolar"], errors="coerce").values

    masks = torch.load(BASE / "data" / "split_masks_random.pt", weights_only=False)
    train_idx = np.asarray(masks["homology"]["train"]).astype(np.int64)
    test_idx  = np.asarray(masks["homology"]["test"]).astype(np.int64)

    # Real-HC50 training peptides
    tr_valid = (hc50 > 0) & (hc50 < 999)
    tr_mask = np.zeros(len(df), dtype=bool)
    tr_mask[train_idx] = True
    tr_mask &= tr_valid
    n_tr_all = int(tr_mask.sum())
    print(f"Training peptides with real HC50: {n_tr_all}")

    # Remove overlap with HemoPI2 training
    hp_train = pd.read_csv(HP_TRAIN)
    hp_seqs = set(hp_train["SEQUENCE"].astype(str).str.upper().str.strip())
    tr_overlap = np.array([s in hp_seqs for s in seqs])
    tr_final = tr_mask & ~tr_overlap
    n_overlap = int((tr_mask & tr_overlap).sum())
    n_tr_final = int(tr_final.sum())
    print(f"  ...of which in HemoPI2 training (removed): {n_overlap}")
    print(f"  ...remaining for retraining:               {n_tr_final}")

    # Real-HC50 test peptides (all 135)
    te_valid = (hc50 > 0) & (hc50 < 999)
    te_mask = np.zeros(len(df), dtype=bool)
    te_mask[test_idx] = True
    te_mask &= te_valid
    n_te = int(te_mask.sum())
    print(f"Test peptides (unchanged): {n_te}")

    tr_seqs = seqs[tr_final]
    tr_y_hc50 = hc50[tr_final].astype(float)
    te_seqs = seqs[te_mask]
    te_y_hc50 = hc50[te_mask].astype(float)

    # ─────────────────────────────────────────────────────────
    # 2. Featurize training + test peptides with HemoPI2 featurizer
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Computing HemoPI2 features (1167 columns)")
    print("=" * 72)
    print(f"  Featurizing {len(tr_seqs)} training peptides...")
    X_tr = compute_features(tr_seqs, "train")
    print(f"    -> {X_tr.shape}")
    print(f"  Featurizing {len(te_seqs)} test peptides...")
    X_te = compute_features(te_seqs, "test")
    print(f"    -> {X_te.shape}")

    if X_tr.shape[1] != 1167 or X_te.shape[1] != 1167:
        print(f"  WARNING: expected 1167 features, got tr={X_tr.shape[1]}, te={X_te.shape[1]}")

    # Ensure columns match
    if list(X_tr.columns) != list(X_te.columns):
        common = [c for c in X_tr.columns if c in X_te.columns]
        print(f"  Aligning column sets: {len(common)} common columns")
        X_tr = X_tr[common]; X_te = X_te[common]

    # HemoPI2 fits on -ln(HC50); predicts exp(-y_hat) as HC50
    y_tr_train = -np.log(tr_y_hc50)   # natural log
    print(f"\n  Training target range: -ln(HC50) ∈ [{y_tr_train.min():.3f}, {y_tr_train.max():.3f}]")

    # ─────────────────────────────────────────────────────────
    # 3. Fit fresh Random Forest with HemoPI2's exact hyperparams
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Training Random Forest (HemoPI2's exact hyperparameters)")
    print("=" * 72)
    for k, v in RF_PARAMS.items():
        print(f"  {k}: {v}")
    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_tr.values, y_tr_train)
    print(f"  Fitted on {X_tr.shape[0]} peptides x {X_tr.shape[1]} features")

    # ─────────────────────────────────────────────────────────
    # 4. Predict on test set
    # ─────────────────────────────────────────────────────────
    y_te_pred_raw = rf.predict(X_te.values)
    te_pred_hc50 = np.exp(-y_te_pred_raw)          # HemoPI2's inverse transform
    te_pred_log10 = np.log10(te_pred_hc50)
    te_true_log10 = np.log10(te_y_hc50)

    # Save per-peptide predictions
    pd.DataFrame({
        "sequence":       te_seqs,
        "true_hc50_uM":   te_y_hc50,
        "pred_hc50_uM":   te_pred_hc50,
        "true_log10":     te_true_log10,
        "pred_log10":     te_pred_log10,
    }).to_csv(OUT_PRED_CSV, index=False)
    print(f"\n  Saved per-peptide predictions to {OUT_PRED_CSV}")

    # ─────────────────────────────────────────────────────────
    # 5. Metrics vs PepTox-MTL and PepTox-STL-hc50
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Retrained HemoPI2 vs PepTox on identical n=135 test peptides")
    print("=" * 72)

    # Load PepTox MTL and STL potency predictions
    SEEDS = range(5)
    mtl_te = np.stack([np.load(BASE / "models" / "v9_random"    / f"test_preds_seed{s}.npz")["p_potency"]
                       for s in SEEDS])
    stl_te = np.stack([np.load(BASE / "models" / "v9_stl_hc50"  / f"test_preds_seed{s}.npz")["p_potency"]
                       for s in SEEDS])
    mtl_ens_all = mtl_te.mean(axis=0)
    stl_ens_all = stl_te.mean(axis=0)

    # PepTox test peptides are in test_idx order; subset to real-HC50
    real_mask = te_valid[test_idx]
    mtl_ens = mtl_ens_all[real_mask]
    stl_ens = stl_ens_all[real_mask]
    # Verify sequence alignment
    te_seqs_from_master = seqs[test_idx][real_mask]
    if not np.array_equal(te_seqs_from_master, te_seqs):
        raise RuntimeError("Sequence order mismatch between PepTox preds and HemoPI2 preds")

    y_log = te_true_log10
    r2_hp = r2_score(y_log, te_pred_log10)
    r2_mt = r2_score(y_log, mtl_ens)
    r2_st = r2_score(y_log, stl_ens)

    print(f"\nR^2 on {len(y_log)} test peptides:")
    print(f"  Retrained HemoPI2    : {r2_hp:.4f}")
    print(f"  PepTox MTL           : {r2_mt:.4f}")
    print(f"  PepTox STL-hc50      : {r2_st:.4f}")
    print(f"  Delta (MTL - HemoPI2): {r2_mt - r2_hp:+.4f}")
    print(f"  Delta (STL - HemoPI2): {r2_st - r2_hp:+.4f}")

    res_hp  = np.abs(te_pred_log10 - y_log)
    res_mtl = np.abs(mtl_ens        - y_log)
    res_stl = np.abs(stl_ens        - y_log)
    print(f"\nMean |log residual|:")
    print(f"  Retrained HemoPI2 : {res_hp.mean():.4f}")
    print(f"  PepTox MTL        : {res_mtl.mean():.4f}")
    print(f"  PepTox STL-hc50   : {res_stl.mean():.4f}")

    w_mtl_hp = wilcoxon(res_mtl, res_hp, alternative="two-sided")
    w_stl_hp = wilcoxon(res_stl, res_hp, alternative="two-sided")
    print(f"\nPaired Wilcoxon on |residuals| (two-sided):")
    print(f"  MTL vs HemoPI2 : W = {w_mtl_hp.statistic:.2f}, p = {w_mtl_hp.pvalue:.4g}")
    print(f"  STL vs HemoPI2 : W = {w_stl_hp.statistic:.2f}, p = {w_stl_hp.pvalue:.4g}")

    print("\n" + "=" * 72)
    print("  Interpretation")
    print("=" * 72)
    if r2_mt > r2_hp and w_mtl_hp.pvalue < 0.05:
        print(f"  PepTox-MTL BEATS retrained HemoPI2  (+{r2_mt-r2_hp:.3f} R^2, p={w_mtl_hp.pvalue:.3g})")
    elif r2_mt > r2_hp:
        print(f"  PepTox-MTL nominally beats HemoPI2 but not significantly (p={w_mtl_hp.pvalue:.3g})")
    else:
        print(f"  Retrained HemoPI2 BEATS PepTox-MTL ({r2_hp-r2_mt:+.3f} R^2, p={w_mtl_hp.pvalue:.3g})")

    if r2_st > r2_hp and w_stl_hp.pvalue < 0.05:
        print(f"  PepTox-STL BEATS retrained HemoPI2 => architecture alone beats HemoPI2")
    elif r2_st > r2_hp:
        print(f"  PepTox-STL nominally beats HemoPI2 (p={w_stl_hp.pvalue:.3g})")
    else:
        print(f"  Retrained HemoPI2 beats PepTox-STL => MTL is what closes the gap")

    with open(OUT_JSON, "w") as f:
        json.dump({
            "n_train_filtered": int(n_tr_final),
            "n_train_original": int(n_tr_all),
            "n_overlap_removed": int(n_overlap),
            "n_test": int(len(y_log)),
            "hyperparameters": RF_PARAMS,
            "r2": {"retrained_hemopi2": float(r2_hp),
                   "mtl": float(r2_mt), "stl": float(r2_st)},
            "mae_log10": {"retrained_hemopi2": float(res_hp.mean()),
                          "mtl": float(res_mtl.mean()),
                          "stl": float(res_stl.mean())},
            "wilcoxon": {
                "mtl_vs_hemopi2": {"W": float(w_mtl_hp.statistic), "p": float(w_mtl_hp.pvalue)},
                "stl_vs_hemopi2": {"W": float(w_stl_hp.statistic), "p": float(w_stl_hp.pvalue)},
            },
        }, f, indent=2)
    print(f"\nResults saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
