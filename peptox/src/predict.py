"""PepTox inference entry point.

Loads a trained PepTox checkpoint and produces per-peptide predictions for all
four endpoints: antimicrobial, cytotoxicity, hemolytic classification, and
hemolytic potency (HC50).

Expects pre-computed features saved as .npy arrays with aligned row ordering
to the input CSV. For predicting on new peptides, first run the feature
extractors (DA-35M, DA-650M, mined motifs, cyto-specific descriptors) in the
same order as training; see Methods for the full pipeline.

Usage:
    python predict.py \
        --checkpoint models/peptox_best.pt \
        --sequences data/example_peptides.csv \
        --features-dir data/features \
        --output predictions.csv
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model import OmniNetV9

AA = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX = {a: i + 1 for i, a in enumerate(AA)}
PAD_IDX, UNK_IDX = 0, len(AA) + 1
VOCAB = len(AA) + 2


def encode_seq(seq: str, max_len: int = 60) -> np.ndarray:
    """Encode an amino-acid sequence as integer indices with padding."""
    s = seq.upper().strip()[:max_len]
    out = np.zeros(max_len, dtype=np.int64)
    for i, a in enumerate(s):
        out[i] = AA2IDX.get(a, UNK_IDX)
    return out


def load_features(features_dir: Path, n_expected: int):
    """Load the five feature streams PepTox expects.

    All arrays must have the same leading dimension (one row per peptide,
    matching the row order of the input sequences CSV).
    """
    paths = {
        "deep":      features_dir / "X_da35m.npy",
        "extra":     features_dir / "X_extra.npy",
        "adv":       features_dir / "X_adv_phys.npy",
        "esm650":    features_dir / "X_da650m.npy",
        "motif_pre": features_dir / "X_motifs.npy",
        "cyto_phys": features_dir / "X_cyto_phys.npy",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing feature arrays in {features_dir}: {missing}\n"
            "Run the feature-extraction pipeline (see Methods) first."
        )
    arrays = {k: np.load(p).astype(np.float32) for k, p in paths.items()}
    for k, a in arrays.items():
        if a.shape[0] != n_expected:
            raise ValueError(
                f"Feature array '{k}' has {a.shape[0]} rows, "
                f"expected {n_expected} to match the sequences CSV."
            )
    # Wide features are extra + advanced physicochemical, concatenated.
    arrays["wide"] = np.concatenate([arrays.pop("extra"), arrays.pop("adv")],
                                    axis=1)
    return arrays


@torch.no_grad()
def predict_batch(model, seqs, features, device, batch_size=64, max_len=60):
    """Run the model over all peptides in batches, return a DataFrame."""
    n = len(seqs)
    results = {k: np.zeros(n, dtype=np.float32)
               for k in ("p_antimicrobial", "p_hemolytic", "p_cytotoxic",
                         "log10_hc50_pred")}

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        idx = slice(start, end)

        seq_batch = torch.from_numpy(
            np.stack([encode_seq(s, max_len) for s in seqs[idx]])
        ).to(device)
        deep = torch.from_numpy(features["deep"][idx]).to(device)
        wide = torch.from_numpy(features["wide"][idx]).to(device)
        esm  = torch.from_numpy(features["esm650"][idx]).to(device)
        mot  = torch.from_numpy(features["motif_pre"][idx]).to(device)
        phys = torch.from_numpy(features["cyto_phys"][idx]).to(device)

        out = model(seq_batch, deep, wide, esm, mot, phys)

        results["p_antimicrobial"][idx] = torch.sigmoid(out["microbe"]).cpu().numpy()
        results["p_hemolytic"][idx]     = torch.sigmoid(out["hemo"]).cpu().numpy()
        results["p_cytotoxic"][idx]     = torch.sigmoid(out["cyto"]).cpu().numpy()
        results["log10_hc50_pred"][idx] = out["potency"].cpu().numpy()

    df = pd.DataFrame(results)
    df["hc50_pred_uM"] = 10.0 ** df["log10_hc50_pred"]
    return df


def main():
    ap = argparse.ArgumentParser(description="PepTox inference")
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="Path to trained PepTox checkpoint (.pt).")
    ap.add_argument("--sequences", required=True, type=Path,
                    help="CSV with a 'sequence' column (one peptide per row).")
    ap.add_argument("--features-dir", required=True, type=Path,
                    help="Directory containing precomputed feature .npy arrays "
                         "(X_da35m.npy, X_da650m.npy, X_motifs.npy, "
                         "X_cyto_phys.npy, X_extra.npy, X_adv_phys.npy).")
    ap.add_argument("--output", default="predictions.csv", type=Path,
                    help="Where to write per-peptide predictions (CSV).")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=60)
    ap.add_argument("--device", default=None,
                    help="'cuda' or 'cpu'; auto-detected if unset.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load sequences
    df_in = pd.read_csv(args.sequences)
    if "sequence" not in df_in.columns:
        raise ValueError(f"{args.sequences} must contain a 'sequence' column.")
    seqs = df_in["sequence"].astype(str).values
    print(f"Loaded {len(seqs)} sequences from {args.sequences}")

    # Load features
    features = load_features(args.features_dir, n_expected=len(seqs))
    print(f"Loaded features from {args.features_dir}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    # Infer feature dims from the loaded arrays
    model = OmniNetV9(
        deep_dim=features["deep"].shape[1],
        wide_dim=features["wide"].shape[1],
        vocab_size=VOCAB,
        esm650_dim=features["esm650"].shape[1],
        motif_pre_dim=features["motif_pre"].shape[1],
        cyto_phys_dim=features["cyto_phys"].shape[1],
    ).to(device).eval()
    # Strip SWA 'module.' prefix if present.
    clean_sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                for k, v in state_dict.items()}
    model.load_state_dict(clean_sd, strict=False)
    print(f"Loaded checkpoint from {args.checkpoint} (device={device})")

    # Predict
    preds = predict_batch(model, seqs, features, device,
                          batch_size=args.batch_size, max_len=args.max_len)

    # Join with input and save
    out = pd.concat([df_in.reset_index(drop=True), preds], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} predictions to {args.output}")


if __name__ == "__main__":
    main()
