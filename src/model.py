"""OmniNet V9: V2 backbone + cyto-specific auxiliary encoder.

Microbe / Hemo / Potency heads: byte-identical to V2 (same params, same data flow).
Cyto head: widened to accept an encoded representation of
    DA-650M ESM (2560-D) + pre-mined motifs (300-D) + cyto biophysics (12-D)
fed through a CytoAuxEncoder, then concatenated with the V2 cyto input.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MotifCNN(nn.Module):
    def __init__(self, vocab_size=22, emb_dim=32, channels=96,
                 kernels=(3, 4, 5), dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, channels, kernel_size=k, padding=k // 2)
            for k in kernels])
        self.bn = nn.ModuleList([nn.BatchNorm1d(channels) for _ in kernels])
        self.dropout = nn.Dropout(dropout)
        self.out_dim = channels * len(kernels) * 2

    def forward(self, seq_idx):
        x = self.emb(seq_idx).transpose(1, 2)
        feats = []
        for conv, bn in zip(self.convs, self.bn):
            h = F.gelu(bn(conv(x)))
            feats.append(torch.cat([h.amax(-1), h.mean(-1)], dim=-1))
        return self.dropout(torch.cat(feats, dim=-1))


class FiLMFusion(nn.Module):
    def __init__(self, deep_dim, wide_dim, hidden=256):
        super().__init__()
        self.wide_proj = nn.Sequential(
            nn.Linear(wide_dim, hidden), nn.GELU(),
            nn.Linear(hidden, deep_dim * 2))
        self.norm = nn.LayerNorm(deep_dim)

    def forward(self, deep, wide):
        gamma, beta = self.wide_proj(wide).chunk(2, dim=-1)
        return self.norm(deep * (1 + gamma) + beta)


class WideDeepTrunk(nn.Module):
    def __init__(self, deep_dim, wide_dim, motif_dim, hidden=384, depth=3,
                 msd_k=5, dropout=0.3):
        super().__init__()
        self.film = FiLMFusion(deep_dim, wide_dim)
        self.input_proj = nn.Linear(deep_dim + motif_dim + wide_dim, hidden)
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hidden * 2, hidden),
            ) for _ in range(depth)])
        self.msd_k = msd_k
        self.msd_drop = nn.Dropout(dropout)
        self.out_dim = hidden

    def forward(self, deep, wide, motif, mixup_lam=None, mixup_perm=None):
        deep_mod = self.film(deep, wide)
        h = torch.cat([deep_mod, motif, wide], dim=-1)
        h = self.input_proj(h)
        if mixup_lam is not None and mixup_perm is not None:
            h = mixup_lam * h + (1 - mixup_lam) * h[mixup_perm]
        for block in self.blocks:
            h = h + block(h)
        return h

    def msd_forward(self, h, head):
        if self.training and self.msd_k > 1:
            return torch.stack(
                [head(self.msd_drop(h)) for _ in range(self.msd_k)]).mean(0)
        return head(h)


class CytoAuxEncoder(nn.Module):
    """Encode (DA-650M ESM + pre-mined motifs + cyto biophysics) → cyto_aux_dim."""
    def __init__(self, esm_dim=2560, motif_pre_dim=300, phys_dim=12,
                 hidden=512, out_dim=192, dropout=0.3):
        super().__init__()
        in_dim = esm_dim + motif_pre_dim + phys_dim
        self.norm_in = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim), nn.LayerNorm(out_dim))
        self.out_dim = out_dim

    def forward(self, esm, motif_pre, phys):
        return self.proj(self.norm_in(torch.cat([esm, motif_pre, phys], dim=-1)))


class PotencyGatedToxicityTowerV9(nn.Module):
    """Same as V2's tower except cyto_head receives cyto_aux."""
    def __init__(self, deep_dim, wide_dim, motif_dim, hidden=384,
                 n_heads=4, dropout=0.3, cyto_aux_dim=192):
        super().__init__()
        self.trunk = WideDeepTrunk(deep_dim, wide_dim, motif_dim,
                                   hidden=hidden, depth=3, msd_k=5,
                                   dropout=dropout)
        self.potency_emb = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout))
        self.potency_reg = nn.Linear(hidden, 1)
        self.hemo_query = nn.Parameter(torch.randn(1, hidden) * 0.02)
        self.cyto_query = nn.Parameter(torch.randn(1, hidden) * 0.02)
        self.attn = nn.MultiheadAttention(hidden, n_heads, dropout=dropout,
                                          batch_first=True)
        self.hemo_head = nn.Sequential(
            nn.LayerNorm(hidden * 2), nn.Linear(hidden * 2, hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        cyto_in = hidden * 2 + cyto_aux_dim
        self.cyto_head = nn.Sequential(
            nn.LayerNorm(cyto_in), nn.Linear(cyto_in, hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, deep, wide, motif, cyto_aux,
                mixup_lam=None, mixup_perm=None):
        h = self.trunk(deep, wide, motif, mixup_lam, mixup_perm)
        p_emb = self.potency_emb(h)
        potency = self.potency_reg(p_emb).squeeze(-1)
        kv = p_emb.unsqueeze(1)
        B = h.size(0)
        hemo_q = self.hemo_query.expand(B, -1).unsqueeze(1)
        cyto_q = self.cyto_query.expand(B, -1).unsqueeze(1)
        hemo_ctx, _ = self.attn(hemo_q, kv, kv)
        cyto_ctx, _ = self.attn(cyto_q, kv, kv)
        hemo_logit = self.hemo_head(
            torch.cat([h, hemo_ctx.squeeze(1)], dim=-1)).squeeze(-1)
        cyto_logit = self.cyto_head(
            torch.cat([h, cyto_ctx.squeeze(1), cyto_aux], dim=-1)).squeeze(-1)
        return hemo_logit, cyto_logit, potency


class MicrobeTower(nn.Module):
    def __init__(self, deep_dim, wide_dim, motif_dim, hidden=384, dropout=0.3):
        super().__init__()
        self.trunk = WideDeepTrunk(deep_dim, wide_dim, motif_dim,
                                   hidden=hidden, depth=3, msd_k=5,
                                   dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden // 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, deep, wide, motif, mixup_lam=None, mixup_perm=None):
        h = self.trunk(deep, wide, motif, mixup_lam, mixup_perm)
        return self.trunk.msd_forward(h, self.head).squeeze(-1)


class OmniNetV9(nn.Module):
    def __init__(self, deep_dim=480, wide_dim=32, vocab_size=22,
                 esm650_dim=2560, motif_pre_dim=300, cyto_phys_dim=12,
                 cyto_aux_dim=192, cnn_channels=96, hidden=384, dropout=0.3):
        super().__init__()
        self.cnn = MotifCNN(vocab_size=vocab_size, channels=cnn_channels,
                            dropout=dropout)
        self.cyto_aux_encoder = CytoAuxEncoder(
            esm_dim=esm650_dim, motif_pre_dim=motif_pre_dim,
            phys_dim=cyto_phys_dim, out_dim=cyto_aux_dim, dropout=dropout)
        self.microbe = MicrobeTower(deep_dim, wide_dim, self.cnn.out_dim,
                                    hidden=hidden, dropout=dropout)
        self.toxicity = PotencyGatedToxicityTowerV9(
            deep_dim, wide_dim, self.cnn.out_dim, hidden=hidden,
            dropout=dropout, cyto_aux_dim=cyto_aux_dim)

    def forward(self, seq_idx, x_deep, x_wide,
                x_esm650, x_motif_pre, x_cyto_phys,
                mixup_lam=None, mixup_perm=None):
        motif = self.cnn(seq_idx)
        if mixup_lam is not None and mixup_perm is not None:
            motif = mixup_lam * motif + (1-mixup_lam) * motif[mixup_perm]
            x_deep = mixup_lam * x_deep + (1-mixup_lam) * x_deep[mixup_perm]
            x_wide = mixup_lam * x_wide + (1-mixup_lam) * x_wide[mixup_perm]
            x_esm650 = mixup_lam * x_esm650 + (1-mixup_lam) * x_esm650[mixup_perm]
            x_motif_pre = mixup_lam * x_motif_pre + (1-mixup_lam) * x_motif_pre[mixup_perm]
            x_cyto_phys = mixup_lam * x_cyto_phys + (1-mixup_lam) * x_cyto_phys[mixup_perm]
        cyto_aux = self.cyto_aux_encoder(x_esm650, x_motif_pre, x_cyto_phys)
        microbe_logit = self.microbe(x_deep, x_wide, motif)
        hemo_logit, cyto_logit, potency = self.toxicity(
            x_deep, x_wide, motif, cyto_aux)
        return {"microbe": microbe_logit, "hemo": hemo_logit,
                "cyto": cyto_logit, "potency": potency}
