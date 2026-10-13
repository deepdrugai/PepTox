"""Train OmniNet V9 in single-task mode for the STL ablation study.

Definition-B ablation: keep the FULL V9 architecture (same params, same
input handling, all four heads and cross-attention), and set the losses of
every task except the selected one to zero. Only the shared trunk and the
selected task's head receive gradient signal; the unused heads' parameters
remain at initialisation.

This isolates the multi-task-learning contribution from the architectural
contribution when compared head-to-head with the V9-MTL baseline trained by
train.py.

Note on split file: the V9-MTL trainer loads split_masks_random.pt but reads
the "homology" key inside it (verified 2026-07-23: the file only contains
that key). The STL trainer uses the SAME split for an apples-to-apples
comparison with the V9-MTL numbers.

Note on epoch definition: one epoch = full iteration over all 36k peptides,
identical to the V9-MTL trainer. For STL, most mini-batches will contain
few or no labelled examples for the selected task, so per-epoch effective
gradient signal is much sparser than for MTL. This is deliberately not
compensated for -- the gap in effective supervision per epoch IS the
disadvantage MTL is claimed to overcome.

Usage:
    python train_stl.py --task cyto    --seed 0
    python train_stl.py --task potency --seed 0
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import matthews_corrcoef, r2_score

from model import OmniNetV9

BASE = Path(".")
CSV_PATH       = BASE / "data" / "master_v6.csv"
DEEP_PATH      = BASE / "data" / "X_master_v6.npy"
EXTRA_PATH     = BASE / "data" / "X_extra_v6.npy"
ADV_PATH       = BASE / "data" / "X_adv_phys_v6.npy"
SPLIT_PATH     = BASE / "data" / "split_masks_random.pt"
ESM650_PATH    = BASE / "data" / "X_master_v8_esm650m_DA.npy"
MOTIF_PATH     = BASE / "data" / "X_motifs_v9.npy"
CYTO_PHYS_PATH = BASE / "data" / "X_cyto_phys.npy"

SPLIT_KEY = "homology"
TASK_WEIGHTS = {"cyto": 2.0, "potency": 0.5}

AA = "ACDEFGHIKLMNPQRSTVWY"
AA2IDX = {a: i + 1 for i, a in enumerate(AA)}
PAD_IDX, UNK_IDX = 0, len(AA) + 1
VOCAB = len(AA) + 2


def encode_seq(seq, max_len):
    s = seq.upper().strip()[:max_len]
    out = np.zeros(max_len, dtype=np.int64)
    for i, a in enumerate(s):
        out[i] = AA2IDX.get(a, UNK_IDX)
    return out


class PeptideDS(Dataset):
    def __init__(self, seqs, y_dict, x_deep, x_wide,
                 x_esm650, x_motif_pre, x_cyto_phys, max_len=60):
        self.seqs = seqs; self.y = y_dict
        self.x_deep = x_deep; self.x_wide = x_wide
        self.x_esm650 = x_esm650
        self.x_motif_pre = x_motif_pre
        self.x_cyto_phys = x_cyto_phys
        self.max_len = max_len

    def __len__(self): return len(self.seqs)

    def __getitem__(self, i):
        return {
            "seq":         torch.from_numpy(encode_seq(self.seqs[i], self.max_len)),
            "x_deep":      torch.from_numpy(self.x_deep[i]).float(),
            "x_wide":      torch.from_numpy(self.x_wide[i]).float(),
            "x_esm650":    torch.from_numpy(self.x_esm650[i]).float(),
            "x_motif_pre": torch.from_numpy(self.x_motif_pre[i]).float(),
            "x_cyto_phys": torch.from_numpy(self.x_cyto_phys[i]).float(),
            "y_microbe":   torch.tensor(self.y["microbe"][i], dtype=torch.long),
            "y_hemo":      torch.tensor(self.y["hemo"][i],    dtype=torch.long),
            "y_cyto":      torch.tensor(self.y["cyto"][i],    dtype=torch.long),
            "y_pot":       torch.tensor(self.y["potency"][i], dtype=torch.float),
        }


def masked_bce(logits, targets, label_smooth=0.0, pos_weight=None,
               focal_gamma=0.0):
    valid = targets >= 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True), 0
    lg, tg = logits[valid], targets[valid].float()
    if label_smooth > 0:
        tg = tg * (1 - label_smooth) + 0.5 * label_smooth
    pw = pos_weight.to(lg.device) if pos_weight is not None else None
    bce = F.binary_cross_entropy_with_logits(lg, tg, pos_weight=pw,
                                             reduction="none")
    if focal_gamma > 0:
        p = torch.sigmoid(lg)
        pt = p * tg + (1 - p) * (1 - tg)
        bce = bce * (1 - pt) ** focal_gamma
    return bce.mean(), valid.sum().item()


def masked_huber(pred, target):
    valid = ~torch.isnan(target)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True), 0
    return F.smooth_l1_loss(pred[valid], target[valid]), valid.sum().item()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def best_threshold_mcc(probs, labels):
    valid = labels >= 0
    p, y = probs[valid], labels[valid]
    if len(np.unique(y)) < 2:
        return 0.5, 0.0
    best_t, best_m = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        m = matthews_corrcoef(y, (p > t).astype(int))
        if m > best_m:
            best_t, best_m = float(t), float(m)
    return best_t, best_m


_TASK_BATCH = {"microbe": "y_microbe", "hemo": "y_hemo",
               "cyto": "y_cyto", "potency": "y_pot"}


def evaluate(model, loader, device):
    model.eval()
    P = {k: [] for k in _TASK_BATCH}; Y = {k: [] for k in _TASK_BATCH}
    with torch.no_grad():
        for b in loader:
            out = model(b["seq"].to(device),
                        b["x_deep"].to(device),
                        b["x_wide"].to(device),
                        b["x_esm650"].to(device),
                        b["x_motif_pre"].to(device),
                        b["x_cyto_phys"].to(device))
            for k in ["microbe", "hemo", "cyto"]:
                P[k].append(torch.sigmoid(out[k]).cpu().numpy())
                Y[k].append(b[_TASK_BATCH[k]].numpy())
            P["potency"].append(out["potency"].cpu().numpy())
            Y["potency"].append(b["y_pot"].numpy())
    return ({k: np.concatenate(v) for k, v in P.items()},
            {k: np.concatenate(v) for k, v in Y.items()})


@torch.no_grad()
def update_bn_dict(loader, model, device):
    momenta, was_training = {}, model.training
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            momenta[m] = m.momentum
    if not momenta:
        return
    model.train()
    for m in momenta:
        m.momentum = None
    n = 0
    for b in loader:
        bs = b["seq"].size(0)
        for m in momenta:
            m.momentum = bs / float(n + bs)
        model(b["seq"].to(device),
              b["x_deep"].to(device),
              b["x_wide"].to(device),
              b["x_esm650"].to(device),
              b["x_motif_pre"].to(device),
              b["x_cyto_phys"].to(device))
        n += bs
    for m, mom in momenta.items():
        m.momentum = mom
    model.train(was_training)


def compute_stl_loss(task, out, y_cyto, y_pot, mixup_lam, mixup_perm, pw_cyto):
    """Return (loss, task_loss_value, n_valid) for the selected STL task.

    V9-MTL loss is:  lm + lh + 2.0*lc + 0.5*lp
    STL keeps only the selected task's term with its V9-MTL weight.
    """
    if task == "cyto":
        w = TASK_WEIGHTS["cyto"]
        lc, nc = masked_bce(out["cyto"], y_cyto, 0.05, pw_cyto, 2.0)
        if mixup_lam is not None:
            lc2, _ = masked_bce(out["cyto"], y_cyto[mixup_perm],
                                0.05, pw_cyto, 2.0)
            loss = mixup_lam * (w * lc) + (1 - mixup_lam) * (w * lc2)
        else:
            loss = w * lc
        return loss, lc.item(), nc
    elif task == "potency":
        w = TASK_WEIGHTS["potency"]
        lp, np_ = masked_huber(out["potency"], y_pot)
        if mixup_lam is not None:
            lp2, _ = masked_huber(out["potency"], y_pot[mixup_perm])
            loss = mixup_lam * (w * lp) + (1 - mixup_lam) * (w * lp2)
        else:
            loss = w * lp
        return loss, lp.item(), np_
    else:
        raise ValueError(f"Unknown task: {task}")


def train_one_seed(args):
    set_seed(args.seed)
    out_dir = BASE / "models" / f"v9_stl_{args.task_tag}"
    log_dir = BASE / "logs"   / f"v9_stl_{args.task_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[STL {args.task} / seed {args.seed}] device={device}")
    print(f"  out_dir={out_dir}")

    df = pd.read_csv(CSV_PATH)
    seqs = df["sequence"].astype(str).values
    y_dict = {
        "microbe": df["gen_label"].fillna(-1).astype(int).values,
        "hemo":    df["hemo_label"].fillna(-1).astype(int).values,
        "cyto":    df["cyto_label"].fillna(-1).astype(int).values,
        "potency": np.where((df["micromolar"] > 0) & (df["micromolar"] < 999),
                            np.log10(df["micromolar"].astype(float)), np.nan),
    }
    x_deep = np.load(DEEP_PATH).astype(np.float32)
    x_extra = np.load(EXTRA_PATH).astype(np.float32)
    x_adv = np.load(ADV_PATH).astype(np.float32)
    x_wide = np.concatenate([x_extra, x_adv], axis=1)
    x_esm650 = np.load(ESM650_PATH).astype(np.float32)
    x_motif_pre = np.load(MOTIF_PATH).astype(np.float32)
    x_cyto_phys = np.load(CYTO_PHYS_PATH).astype(np.float32)

    n = len(df)
    for name, arr in [("deep", x_deep), ("esm650", x_esm650),
                       ("motif", x_motif_pre), ("cyto_phys", x_cyto_phys)]:
        assert arr.shape[0] == n, f"{name} rows {arr.shape[0]} != master rows {n}"

    masks = torch.load(SPLIT_PATH, weights_only=False)
    splits = masks[SPLIT_KEY]
    train_idx = np.asarray(splits["train"]).astype(np.int64)
    val_idx   = np.asarray(splits["val"]).astype(np.int64)
    test_idx  = np.asarray(splits["test"]).astype(np.int64)

    def subset(idx):
        return PeptideDS(seqs[idx],
                         {k: v[idx] for k, v in y_dict.items()},
                         x_deep[idx], x_wide[idx],
                         x_esm650[idx], x_motif_pre[idx], x_cyto_phys[idx],
                         max_len=args.max_len)

    tr_ds, va_ds, te_ds = subset(train_idx), subset(val_idx), subset(test_idx)

    if args.task == "cyto":
        n_labels_tr = int((y_dict["cyto"][train_idx] >= 0).sum())
    else:
        n_labels_tr = int((~np.isnan(y_dict["potency"][train_idx])).sum())
    print(f"  train={len(tr_ds)}  val={len(va_ds)}  test={len(te_ds)}")
    print(f"  effective {args.task} labels in train: {n_labels_tr}")

    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers,
                           pin_memory=(device == "cuda"), drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=512, shuffle=False,
                           num_workers=max(1, args.num_workers // 2))
    te_loader = DataLoader(te_ds, batch_size=512, shuffle=False,
                           num_workers=max(1, args.num_workers // 2))

    def pw(arr):
        a = arr[arr >= 0]
        n_pos, n_neg = (a == 1).sum(), (a == 0).sum()
        return torch.tensor([max(n_neg / max(n_pos, 1), 1.0)], dtype=torch.float)
    pw_cyto = pw(y_dict["cyto"][train_idx])
    if args.task == "cyto":
        print(f"  pos_weight cyto: {pw_cyto.item():.2f}")

    model = OmniNetV9(deep_dim=x_deep.shape[1], wide_dim=x_wide.shape[1],
                      vocab_size=VOCAB,
                      esm650_dim=x_esm650.shape[1],
                      motif_pre_dim=x_motif_pre.shape[1],
                      cyto_phys_dim=x_cyto_phys.shape[1]).to(device)
    print(f"  params={sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(tr_loader), eta_min=args.lr * 0.01)

    swa_start = int(args.epochs * 0.60)
    swa_model = AveragedModel(model)
    swa_sched = SWALR(opt, swa_lr=args.lr * 0.1)

    best_score = -1e9
    best_state = None
    history = []
    patience_counter = 0
    PATIENCE = 15

    for epoch in range(args.epochs):
        model.train()
        ep_loss_sum = 0.0
        ep_n = 0

        for b in tr_loader:
            seq = b["seq"].to(device)
            xd, xw = b["x_deep"].to(device), b["x_wide"].to(device)
            xe = b["x_esm650"].to(device)
            xmp = b["x_motif_pre"].to(device)
            xcp = b["x_cyto_phys"].to(device)
            yc = b["y_cyto"].to(device)
            yp = b["y_pot"].to(device)

            if random.random() < args.mixup_p:
                lam = float(np.random.beta(0.4, 0.4))
                perm = torch.randperm(seq.size(0), device=device)
                out = model(seq, xd, xw, xe, xmp, xcp,
                            mixup_lam=lam, mixup_perm=perm)
                loss, l_val, n_val = compute_stl_loss(
                    args.task, out, yc, yp, lam, perm, pw_cyto)
            else:
                out = model(seq, xd, xw, xe, xmp, xcp)
                loss, l_val, n_val = compute_stl_loss(
                    args.task, out, yc, yp, None, None, pw_cyto)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if epoch < swa_start:
                sched.step()

            ep_loss_sum += l_val * max(n_val, 1)
            ep_n        += max(n_val, 1)

        if epoch >= swa_start:
            swa_model.update_parameters(model)
            swa_sched.step()

        if epoch >= swa_start:
            update_bn_dict(tr_loader, swa_model, device=device)
            eval_model = swa_model
        else:
            eval_model = model
        P_va, Y_va = evaluate(eval_model, va_loader, device)

        thr_m, mcc_m = best_threshold_mcc(P_va["microbe"], Y_va["microbe"])
        thr_h, mcc_h = best_threshold_mcc(P_va["hemo"],    Y_va["hemo"])
        thr_c, mcc_c = best_threshold_mcc(P_va["cyto"],    Y_va["cyto"])
        mp = ~np.isnan(Y_va["potency"])
        r2 = r2_score(Y_va["potency"][mp], P_va["potency"][mp]) if mp.sum() else 0.0

        score = mcc_c if args.task == "cyto" else r2
        tag = "  [SWA]" if epoch >= swa_start else ""
        print(f"  [ep {epoch+1:02d}] task_loss={ep_loss_sum/max(ep_n,1):.3f} | "
              f"VAL m={mcc_m:.4f}@{thr_m:.2f} h={mcc_h:.4f}@{thr_h:.2f} "
              f"c={mcc_c:.4f}@{thr_c:.2f} R2={r2:.4f} | "
              f"stl_score({args.task})={score:.4f}{tag}")

        history.append({"epoch": epoch + 1,
                        "mcc_microbe": mcc_m, "thr_microbe": thr_m,
                        "mcc_hemo": mcc_h, "thr_hemo": thr_h,
                        "mcc_cyto": mcc_c, "thr_cyto": thr_c,
                        "r2": r2, "stl_score": score})

        if score > best_score:
            best_score = score
            patience_counter = 0
            best_state = {
                "state_dict": (swa_model.state_dict() if epoch >= swa_start
                               else model.state_dict()),
                "is_swa": epoch >= swa_start,
                "task": args.task,
                "thresholds": {"microbe": thr_m, "hemo": thr_h, "cyto": thr_c},
                "val_mcc": {"microbe": mcc_m, "hemo": mcc_h, "cyto": mcc_c},
                "val_r2": r2, "epoch": epoch + 1, "seed": args.seed,
            }
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  early stop: no improvement for {PATIENCE} epochs "
                      f"(best at epoch {best_state['epoch']}, "
                      f"score={best_score:.4f})")
                break

    ckpt = out_dir / f"omni_v9_stl_{args.task_tag}_seed{args.seed}.pt"
    torch.save(best_state, ckpt)
    print(f"  saved {ckpt}  best_score={best_score:.4f}  "
          f"epoch={best_state['epoch']}")

    final = OmniNetV9(deep_dim=x_deep.shape[1], wide_dim=x_wide.shape[1],
                      vocab_size=VOCAB,
                      esm650_dim=x_esm650.shape[1],
                      motif_pre_dim=x_motif_pre.shape[1],
                      cyto_phys_dim=x_cyto_phys.shape[1]).to(device)
    if best_state["is_swa"]:
        swa_wrap = AveragedModel(final)
        swa_wrap.load_state_dict(best_state["state_dict"])
        eval_model = swa_wrap
    else:
        final.load_state_dict(best_state["state_dict"])
        eval_model = final

    P_va, Y_va = evaluate(eval_model, va_loader, device)
    P_te, Y_te = evaluate(eval_model, te_loader, device)
    np.savez(out_dir / f"val_preds_seed{args.seed}.npz",
             **{f"p_{k}": v for k, v in P_va.items()},
             **{f"y_{k}": v for k, v in Y_va.items()})
    np.savez(out_dir / f"test_preds_seed{args.seed}.npz",
             **{f"p_{k}": v for k, v in P_te.items()},
             **{f"y_{k}": v for k, v in Y_te.items()})
    with open(log_dir / f"history_seed{args.seed}.json", "w") as f:
        json.dump(history, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["cyto", "potency"], required=True,
                    help="Single task to train. cyto = binary cytotoxicity; "
                         "potency = log10(HC50) regression.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=60)
    ap.add_argument("--mixup_p", type=float, default=0.4)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()
    args.task_tag = "cyto" if args.task == "cyto" else "hc50"
    train_one_seed(args)


if __name__ == "__main__":
    main()
