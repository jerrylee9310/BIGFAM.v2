"""Shared simulation parameters. Truth is fixed (V_G=0.5, V_S=0.2); methods are
scored by bias = V_G_hat - 0.5.
"""
from __future__ import annotations

from pathlib import Path

# ── truth ─────────────────────────────────────────────────────────────────────
V_G_TRUE = 0.5
V_S_TRUE = 0.2
V_E_TRUE = 1.0 - V_G_TRUE - V_S_TRUE      # 0.3
W_G = 0.5                                  # Mendelian genetic decay per DOR (fixed)

# ── sweep / grid ──────────────────────────────────────────────────────────────
WS_GRID = [0.2, 0.5, 0.8]                  # 0.2 fast (main) · 0.5 degenerate · 0.8 slow (main)
DORS = [1, 2, 3]
N_D = 10_000                               # pairs per DOR
K_PREV = 0.3                               # binary prevalence
SCALES = ["continuous", "binary"]

# ── replicate count (CLI --reps overrides) ──────────────────────────────────
R = 10

# ── BIGFAM covariate contract ────────────────────────────────────────────────
# no real covariates in this simulation -> empty list, intercept-only design.
COV_COLS = []

# ── reproducibility ───────────────────────────────────────────────────────────
SEED_BASE = 20_260_622

# ── paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
RESULTS_DIR = _HERE / "results"


def true_rho(w_s: float, d: int) -> float:
    """True relative correlation rho_d = w_G^d * V_G + w_S^{d-1} * V_S."""
    return W_G ** d * V_G_TRUE + w_s ** (d - 1) * V_S_TRUE


def replicate_seed(scale_idx: int, ws_idx: int, rep: int) -> int:
    """Deterministic per-(scale, w_S, rep) seed -- reproducible."""
    return SEED_BASE + scale_idx * 1_000_000 + ws_idx * 100_000 + rep
