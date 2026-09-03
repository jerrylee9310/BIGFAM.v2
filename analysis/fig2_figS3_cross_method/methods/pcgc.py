"""PCGC regression -- binary liability-scale heritability (Golan, Lander, Rosset 2014, PNAS).

The standard binary generalization of HE ("Haseman-Elston regression is a
special case of PCGC"). Under the liability-threshold model, the product of
normalized phenotypes regresses on genetic relatedness with slope c*h2_l.
zero-class (s_d=0, additive genetics only) -- V_S>0 gets absorbed the same way
as Falconer/HE.

Golan eq[4]:  E[z_i z_j] = c * h2_l * G_ij        (z=normalized phenotype, G=relatedness=w_G^d)
     eq[10]:  h2_l = slope( z_i z_j  ~  G_ij )  / c   (origin OLS = HE regression)
     eq[13]:  c = P(1-P) phi(t)^2 / (K^2(1-K)^2),   t=Phi^-1(1-K), phi=standard normal density
                  P=sample case fraction, K=population prevalence
Unascertained sampling (P=K) gives c = phi(t)^2/(K(1-K)). This uses the sample
P directly (self-calibrating).

Covariate-adjusted (if cov given): Golan supplement's fixed-effect PCGC. probit
y~[1,cov] gives a per-individual liability mean u_i=W_i*beta, per-individual
P_i=Phi(u_i), normalized z_i=(y_i-P_i)/sqrt(P_i(1-P_i)), density factor
q_i=phi(u_i)/sqrt(P_i(1-P_i)) (t_i=Phi^-1(1-P_i)=-u_i so phi(t_i)=phi(u_i)).
E[z_i z_j]=q_i q_j h2_l G_ij -> regressing (z1 z2) on (q1 q2 G) through the
origin gives slope=h2_l. cov_cols=[] reduces to the unadjusted branch (K=P).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from relatedness import W_G

_WGD = W_G ** np.arange(1, 4)              # [0.5, 0.25, 0.125]
_REL = {1: _WGD[0], 2: _WGD[1], 3: _WGD[2]}


def estimate(rho, pairs, pheno, K, cov=None, cov_cols=None):
    """K = population prevalence (simulation: fixed at config.K_PREV; real
    data: per-trait). cov=None -> unadjusted (byte-identical to the sim).
    cov given -> Golan per-individual covariate adjustment."""
    y = pheno["phenotype"]
    G = pairs["dor"].map(_REL).to_numpy()          # relative genetic correlation G_ij = w_G^d

    if cov is None:
        P = float(y.mean())                        # sample case fraction
        t = norm.ppf(1.0 - K)                      # liability threshold
        phi = float(norm.pdf(t))                   # standard normal density phi(t)
        z = (y - P) / np.sqrt(P * (1.0 - P))       # normalized phenotype (mean 0, var 1)
        z1 = z.loc[pairs["id1"].values].to_numpy()
        z2 = z.loc[pairs["id2"].values].to_numpy()
        slope = float((z1 * z2 * G).sum() / (G * G).sum())   # origin OLS (HE regression)
        c = P * (1.0 - P) * phi ** 2 / (K ** 2 * (1.0 - K) ** 2)
        return {"V_G": slope / c}                  # h2_l = slope / c

    # covariate-adjusted -- Golan per-individual. Not exercised by this
    # simulation (which has no covariates); kept for parity with the source.
    import pandas as pd
    import statsmodels.api as sm
    cols = cov_cols or []
    yv = y.to_numpy(float)
    W = np.column_stack([np.ones(len(yv)), cov.loc[y.index, cols].to_numpy(float)])
    beta = sm.Probit(yv, W).fit(disp=0).params
    u = W @ beta                                   # per-individual liability mean u_i
    P_i = np.clip(norm.cdf(u), 1e-6, 1 - 1e-6)
    denom = np.sqrt(P_i * (1.0 - P_i))
    z = pd.Series((yv - P_i) / denom, index=y.index)
    q = pd.Series(norm.pdf(u) / denom, index=y.index)        # q_i = phi(t_i)/sqrt(P_i(1-P_i))
    z1 = z.loc[pairs["id1"].values].to_numpy(); z2 = z.loc[pairs["id2"].values].to_numpy()
    q1 = q.loc[pairs["id1"].values].to_numpy(); q2 = q.loc[pairs["id2"].values].to_numpy()
    x = q1 * q2 * G                                # per-individual c absorbed into the regressor
    slope = float((z1 * z2 * x).sum() / (x * x).sum())       # origin OLS -> h2_l
    return {"V_G": slope}


def _demo():
    """Self-check: with cov_cols=[], the covariate branch matches the unadjusted branch (K=P)."""
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 4000
    g = rng.normal(size=n)
    ids = np.arange(n)
    pheno = pd.DataFrame({"phenotype": (g + rng.normal(size=n) > 0.5).astype(float)},
                         index=ids)
    K = float(pheno["phenotype"].mean())                     # K=P -> both branches on the same scale
    pairs = pd.DataFrame({"id1": ids[:-1], "id2": ids[1:],
                          "dor": rng.integers(1, 4, n - 1)})
    h_plain = estimate(None, pairs, pheno, K)["V_G"]
    h_cov0 = estimate(None, pairs, pheno, K,
                      cov=pd.DataFrame(index=ids), cov_cols=[])["V_G"]
    assert abs(h_plain - h_cov0) < 1e-6, (h_plain, h_cov0)
    print(f"_demo ok: plain={h_plain:.4f}  cov[]={h_cov0:.4f}")


if __name__ == "__main__":
    _demo()
