# Numerical illustration for "Statistical Complexity of Soft Bellman Residual Minimization"

Coupled minibatch SGDA on the bi-conjugate soft-BRM objective with two-layer networks,
measuring argument stability and the exact excess population MSBE as functions of n.
Pure numpy; gradients are hand-derived and checked against finite differences (max relative error ~1e-9).

## Files

| File | What it is |
|---|---|
| `brm_stability.py` | MDP, data, networks, coupled SGDA, metrics; command-line runner |
| `constants.py` | Estimates mu_PL, rho, mu_QG, G, L along SGDA trajectories |
| `analyze.py` | Summaries, slopes with bootstrap CIs, figure (`stability_vs_n.pdf/png`), `summary.json` |
| `make_appendix.py` | Writes `appendix_E.tex` and `main_text_pointer.tex` from the result files |
| `results_iid.jsonl`, `results_traj.jsonl` | Raw per-repetition results (one JSON object per (width, n)) |
| `constants_H16_n800.json` | Raw constant estimates |
| `appendix_E.tex`, `main_text_pointer.tex`, `stability_vs_n.pdf` | Paste-ready LaTeX and figure |

## Reproduce (about 1 hour on 2 CPU cores)

```bash
export OMP_NUM_THREADS=1
python3 brm_stability.py --widths 8 16 32 --data iid        --out results_iid.jsonl    # ~45 min
python3 brm_stability.py --widths 16      --data trajectory --out results_traj.jsonl   # ~11 min
python3 constants.py 16 800 constants_H16_n800.json                                     # ~5 min
python3 analyze.py results_iid.jsonl results_traj.jsonl
python3 make_appendix.py
```

## Protocol

- MDP: |S|=5, |A|=3, beta=0.9, eta=1; P(.|s,a) = 0.5 delta_s + 0.5 Dirichlet(1); rewards Unif[0,1]; seed 0;
  uniform behaviour policy; mu = stationary state-action distribution (min mass 0.046).
- Data: n i.i.d. transitions from mu x P, or one stationary trajectory of length n.
  Neighbour: replace a uniformly chosen sample by an independent copy from mu x P.
- Networks: one-hot inputs, H softplus units, NTK parameterization (output layer scaled by H^-1/2),
  standard Gaussian init, output biases at the constant value scale. H in {8, 16, 32}.
- SGDA: B=32, eta_t = 400/(2000+t), dual multiplier kappa_v=2, T=40,000; same init and same index sequence
  on D and D^(i); 50 repetitions per (n, H); n in {100, ..., 6400}.
- Metrics at T: ||w_T - w'_T||, ||v_T - v'_T||, ||Q_w - Q_w'||_mu, exact population MSBE of the run on D,
  generalization gap R - R_n, and the population MSBE of the exact empirical minimizer (tabular fixed point).
- Slopes: OLS of log(mean) on log n over n >= 200, 95% bootstrap CI over repetitions.

## Differences from the experiment described in the rebuttal

Softplus instead of ReLU (keeps F_D smooth, condition (C1)); NTK parameterization (step size independent of
width; with standard parameterization and a shared step size, width 32 diverged); B, step sizes, T and the
MDP are now fully specified; n spans 100-6400 instead of 50-800; the primal stability (Theorem 3's quantity)
is reported separately from the dual block; the excess MSBE is computed exactly. The width-8 plateau
reported in the rebuttal does not appear here.
