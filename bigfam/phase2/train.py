"""Offline training: wide-DGP simulation -> ridge calibration artifact.

NOT part of the inference path. Step 2 is ridge-only: the reliability GBR/gate
was removed (the reporting tier does not use it -- scratch v2-fieller-refactor
step5). The downstream-loss helpers stay as a scoring metric for research.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from ..config import CLIP, ALPHAS_CV, SEED
from ..types import CalibrationCoef
from ..core.design import design_matrix_batch
from ..phase3.refit import refit_batch
from .features import FEAT_ALL


RHO_OBS_COLS = [f"rho_obs_{d}" for d in range(1, 4)]
SIGMA_OBS_COLS = [[f"Sigma_{r}{c}" for c in range(1, 4)] for r in range(1, 4)]


def _safe_std(x):
    s = float(np.std(np.asarray(x, dtype=float)))
    return s if np.isfinite(s) and s > 0 else 1.0


def _downstream_arrays(train_df):
    required = ["V_G", "V_S", "w_S_true", *RHO_OBS_COLS]
    required += [col for row in SIGMA_OBS_COLS for col in row]
    missing = [col for col in required if col not in train_df.columns]
    if missing:
        raise ValueError(
            "downstream loss requires columns: " + ", ".join(missing)
        )

    rho = train_df[RHO_OBS_COLS].values.astype(float)
    Sigmas = np.empty((len(train_df), 3, 3), dtype=float)
    for r, row in enumerate(SIGMA_OBS_COLS):
        for c, col in enumerate(row):
            Sigmas[:, r, c] = train_df[col].values.astype(float)

    return (
        rho,
        Sigmas,
        train_df["V_G"].values.astype(float),
        train_df["V_S"].values.astype(float),
        train_df["w_S_true"].values.astype(float),
    )


def _downstream_scales(V_G, V_S, w_S):
    return (_safe_std(V_G), _safe_std(V_S), _safe_std(w_S))


def _downstream_fisher(w_true, Sigmas):
    """Per-row Fisher information A = X(w_true)^T Sigma^-1 X(w_true) at the TRUE w_S.

    This is the metric that makes the downstream loss proper near the w_S=0.5
    singularity. At w_true=0.5 the two design columns are collinear, so A has a
    zero eigenvalue along the unidentified (V_G,V_S) split direction -- weighting
    the refit error by A therefore penalises ONLY the recoverable linear
    combination, never the part of (V_G,V_S) the data cannot resolve. Frozen at
    truth (not w_hat) so the gate cannot game it by inflating its own SE.
    """
    Xt = design_matrix_batch(np.asarray(w_true, dtype=float))   # (N, 3, 2)
    Sinv = np.linalg.inv(np.asarray(Sigmas, dtype=float))
    return np.einsum("nia,nij,njb->nab", Xt, Sinv, Xt)          # (N, 2, 2)


def _downstream_hybrid_loss_arrays(w_hat, rho, Sigmas, V_G, V_S, w_S,
                                   lambda_w=0.05, scales=None, fisher=None,
                                   quad_cap=None):
    """Fisher-weighted Phase3 refit error + weak w_S anchor (mean over rows).

    loss_n = (beta_hat_n - beta_true_n)^T A_n (beta_hat_n - beta_true_n)
             + lambda_w * ((w_hat_n - w_S_n) / scale_w)^2

    beta_hat is the Phase3 refit at the PREDICTED w_hat; A_n = Fisher info at the
    TRUE w_S (see _downstream_fisher). The quadratic form is ~chi^2 with 2 dof at
    the oracle (w_hat=w_S): E[r^T A r] = tr(A A^-1) = 2 regardless of Sigma scale
    or conditioning, so no row -- including the near-singular w_S~=0.5 cases --
    can dominate the mean. The earlier (beta-true)^2 / pop-std form was
    heteroscedastic and let ~5% near-0.5 rows hijack the gate (a,b).
    """
    w_hat = np.clip(np.asarray(w_hat, dtype=float), *CLIP)
    if scales is None:
        scales = _downstream_scales(V_G, V_S, w_S)
    scale_w = scales[2]
    if fisher is None:
        fisher = _downstream_fisher(w_S, Sigmas)

    beta, *_ = refit_batch(w_hat, rho, Sigmas)
    r = beta - np.column_stack([np.asarray(V_G, float), np.asarray(V_S, float)])
    quad = np.einsum("na,nab,nb->n", r, fisher, r)             # r^T A r >= 0
    quad = np.where(np.isfinite(quad), quad, 0.0)
    quad = np.maximum(quad, 0.0)                                # guard fp negatives
    if quad_cap is not None:
        quad = np.minimum(quad, float(quad_cap))               # optional outlier guard

    anchor = float(lambda_w) * ((w_hat - w_S) / scale_w) ** 2
    return float(np.mean(quad + anchor))


def downstream_hybrid_loss(w_hat, train_df, lambda_w=0.05, scales=None,
                           quad_cap=None):
    """Simulation-only loss: Fisher-weighted Phase3 refit error + weak w_S anchor."""
    if len(w_hat) != len(train_df):
        raise ValueError("w_hat length must match train_df length")

    rho, Sigmas, V_G, V_S, w_S = _downstream_arrays(train_df)
    return _downstream_hybrid_loss_arrays(
        w_hat, rho, Sigmas, V_G, V_S, w_S,
        lambda_w=lambda_w, scales=scales, quad_cap=quad_cap,
    )


def _fit_ridge_ws(X, y, n_splits=5, seed=SEED, scoring=None):
    """StandardScaler + RidgeCV(alpha) for the w_S predictor.

    scoring=None reproduces the original RidgeCV alpha selection (R^2), so the
    shipped 'ws' baseline artifact is byte-for-byte unchanged by this refactor.
    Pass scoring='neg_root_mean_squared_error' only where RMSE selection is
    explicitly wanted.
    """
    sc = StandardScaler().fit(X)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    ridge = RidgeCV(alphas=ALPHAS_CV, cv=cv, scoring=scoring).fit(sc.transform(X), y)
    return sc, ridge


# ── Step 2: calibration ──────────────────────────────────────────────────────
def train_calibration(train_df, xfit=True, n_splits=5, seed=SEED) -> CalibrationCoef:
    """Fit StandardScaler + RidgeCV for w_S. Step 2 is ridge-only.

    The w_map confidence gate was removed (see calibrate.py history); `xfit`,
    `n_splits`, `seed` are retained for signature compatibility but no longer
    affect the result -- they only ever tuned the removed gate.

    Returns a serialisable CalibrationCoef (no closures).
    """
    y_tr = train_df["w_S_true"].values
    X_tr = train_df[FEAT_ALL].values

    sc, ridge = _fit_ridge_ws(X_tr, y_tr)

    return CalibrationCoef(
        feature_order=list(FEAT_ALL),
        scaler_mean=np.asarray(sc.mean_, dtype=float),
        scaler_scale=np.asarray(sc.scale_, dtype=float),
        ridge_coef=np.asarray(ridge.coef_, dtype=float),
        ridge_intercept=float(ridge.intercept_),
        ridge_alpha=float(ridge.alpha_),
        clip=tuple(CLIP),
    )


def train_calibration_downstream(train_df, lambda_w=0.05, n_splits=5,
                                 seed=SEED, xfit=True) -> CalibrationCoef:
    """Ridge calibration; retained for API/experiment compatibility.

    This used to tune the w_map blend gate (a,b) against a Phase-3 Fisher-weighted
    downstream loss. With the gate removed (see calibrate.py history) there is
    nothing downstream-specific left to fit, so it returns the same ridge-only
    calibrator as train_calibration; `lambda_w` is ignored. downstream_hybrid_loss
    stays available as a scoring metric (used by per-case validation (research)).
    """
    return train_calibration(train_df, xfit=xfit, n_splits=n_splits, seed=seed)


def calibrate_predict(df, calib: CalibrationCoef) -> np.ndarray:
    """Vectorised calibrated w_S over a DataFrame (matches calibrate_ws row-wise)."""
    lo, hi = calib.clip
    x_std = (df[FEAT_ALL].values - calib.scaler_mean) / calib.scaler_scale
    return np.clip(calib.ridge_intercept + x_std @ calib.ridge_coef, lo, hi)


# ── full offline pipeline ────────────────────────────────────────────────────
def train_all(out_dir, objective="ws", lambda_w=0.05):
    """wide-DGP dgm -> features -> ridge calibration -> save ws_calibration.json.

    Returns the CalibrationCoef.
    """
    from .dgm import generate_training_frame_wide
    from ..io.save import save_artifacts

    train = generate_training_frame_wide()               # all rows are 'train'

    if objective == "ws":
        calib = train_calibration(train)
    elif objective == "downstream_hybrid":
        calib = train_calibration_downstream(train, lambda_w=lambda_w)
    else:
        raise ValueError("objective must be 'ws' or 'downstream_hybrid'")

    save_artifacts(calib, out_dir)
    return calib
