# Figure S2 — source of the Fig. 2 bias

Fig. 2 shows BIGFAM.v2 underestimating heritability under slow ($w_C=0.8$) and
fast ($w_C=0.2$) common-environmental decay. This isolates *where* that bias
comes from by comparing three variants of the Phase 2 -> Phase 3 pipeline on
the same simulated data (1,000 replicates x 3 decay rates, DOR-1/2/3 pairs,
$N_d=10{,}000$, true $V_A=0.5$, $V_C=0.2$):

| variant | what it does | isolates |
|---|---|---|
| `pipeline` | $\hat w_C$ (Phase 2 ridge) -> NNLS refit | the reported BIGFAM.v2 estimator |
| `truew_nnls` | true $w_C$ -> NNLS refit | removes the decay-rate *estimation* error |
| `truew_gls` | true $w_C$ -> unconstrained GLS | also removes the NNLS non-negativity constraint |

Result: under slow/fast decay, only `pipeline` is biased — the two true-$w$
variants sit on the true $V_A$ almost exactly, so the NNLS step contributes
~0 of the bias; the bias is essentially all decay-rate estimation error. At
the degenerate point ($w_C=0.5$) the design matrix is exactly singular, so
`truew_gls` is undefined (plotted as "singular") and even `truew_nnls` does
not recover the truth — knowing the true decay rate does not save this point,
because the scenario itself has no unique answer.

This is paper **Figure S2**.

## Files

- `generate.py` — simulates the pairs, runs Phase 1 -> 2 -> 3 for all three
  variants, writes `figS2.parquet` (9,000 rows: 1,000 reps x 3 $w_C$ x 3
  variants).
- `plot.py` — reads `figS2.parquet`, writes `figS2.png` + `figS2.pdf`.
- `_style.py` — shared Nature Genetics-style matplotlib settings (fonts,
  colors, panel labels). Copied as-is from the paper's figure style module;
  falls back to DejaVu Sans if Arial isn't installed.
- `figS2.parquet` — checked in so `plot.py` works right after cloning, without
  waiting on `generate.py`.

## Run

```bash
.venv/bin/pip install -e ".[analysis]"     # from the repo root
.venv/bin/python analysis/figS2_bias_decomposition/generate.py   # ~3 min
.venv/bin/python analysis/figS2_bias_decomposition/plot.py
```

## External dependencies

None — only the installed `bigfam` package (+ matplotlib/seaborn/pandas from
the `analysis` extra).

## Validation

`generate.py` was run in this repo's own `.venv` and its output compared to
`paper/materials/figS2.parquet` from the private working repo that produced
the submitted draft (same seeds, same math, reimplemented self-contained here
instead of importing across folders). All 9,000 rows matched to floating-point
precision (max abs diff 2.5e-8, all in the `truew_nnls` row at the degenerate
point; every other row diff'd to <5e-12), and the `truew_gls` NaNs at $w_C=0.5$
matched exactly (1,000/1,000 both sides). `plot.py`'s output reproduces the
submitted Figure S2 layout and pattern.
