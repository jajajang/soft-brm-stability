"""Numerical illustration for "Statistical Complexity of Soft Bellman Residual Minimization".

Measures the on-average argument stability of minibatch SGDA on the bi-conjugate soft-BRM
objective (Eq. (5) of the paper) with small two-layer networks for Q_w and zeta_v.

For each repetition we draw a dataset D, an index i ~ Unif[n] and an independent copy
z~_i, form the neighbour D^(i), and run SGDA on D and D^(i) from the same initialization
with the same minibatch index sequence. We record ||w_T - w'_T|| and ||v_T - v'_T||.

Everything is written in numpy with hand-derived gradients (checked against finite
differences in `gradient_check`). All runs of one configuration (repetitions x {D, D^(i)})
are vectorized along a leading axis R = 2 * reps.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict

import numpy as np


# --------------------------------------------------------------------------------------
# MDP and data
# --------------------------------------------------------------------------------------
@dataclass
class MDP:
    S: int
    A: int
    beta: float
    eta: float
    P: np.ndarray        # (S, A, S) transition kernel
    r: np.ndarray        # (S, A) rewards in [0, 1]
    pi_b: np.ndarray     # (S, A) behaviour policy
    mu: np.ndarray       # (S, A) stationary state-action distribution under pi_b


def make_mdp(seed: int = 0, S: int = 5, A: int = 3, beta: float = 0.9, eta: float = 1.0,
             stick: float = 0.5) -> MDP:
    """Random MDP with a uniform behaviour policy. With prob. `stick` the state stays put,
    which makes trajectories positively autocorrelated (single-trajectory variant)."""
    rng = np.random.default_rng(seed)
    P = rng.dirichlet(np.ones(S), size=(S, A))
    P = (1 - stick) * P + stick * np.eye(S)[:, None, :]
    r = rng.uniform(0.0, 1.0, size=(S, A))
    pi_b = np.full((S, A), 1.0 / A)
    # stationary distribution of the state chain under pi_b
    Ps = np.einsum("sa,sat->st", pi_b, P)
    evals, evecs = np.linalg.eig(Ps.T)
    d = np.real(evecs[:, np.argmin(np.abs(evals - 1))])
    d = d / d.sum()
    mu = d[:, None] * pi_b
    return MDP(S, A, beta, eta, P, r, pi_b, mu)


def sample_iid(mdp: MDP, n: int, rng: np.random.Generator):
    """n i.i.d. transitions with (s, a) ~ mu and s' ~ P(.|s, a)."""
    idx = rng.choice(mdp.S * mdp.A, size=n, p=mdp.mu.ravel())
    s, a = np.divmod(idx, mdp.A)
    u = rng.random(n)[:, None]
    sp = (u > np.cumsum(mdp.P[s, a], axis=1)).sum(axis=1)
    sp = np.minimum(sp, mdp.S - 1)
    return s, a, sp


def sample_trajectory(mdp: MDP, n: int, rng: np.random.Generator):
    """One stationary trajectory of length n under pi_b; z_t = (s_t, a_t, s_{t+1})."""
    d = mdp.mu.sum(axis=1)
    s = np.empty(n + 1, dtype=int)
    a = np.empty(n, dtype=int)
    s[0] = rng.choice(mdp.S, p=d)
    for t in range(n):
        a[t] = rng.choice(mdp.A, p=mdp.pi_b[s[t]])
        s[t + 1] = rng.choice(mdp.S, p=mdp.P[s[t], a[t]])
    return s[:-1], a, s[1:]


# --------------------------------------------------------------------------------------
# Networks (NTK parameterization, H hidden units):
#           Q_w(s, .)    = W2 sigma(W1 e_s + b1) / sqrt(H) + b2     (one output per action)
#           zeta_v(s, a) = u2 . sigma(U1 e_{(s,a)} + c1) / sqrt(H) + c2
# Parameters carry a leading axis R (independent runs).
# --------------------------------------------------------------------------------------
W_KEYS = ("W1", "b1", "W2", "b2")
V_KEYS = ("U1", "c1", "u2", "c2")


def act(x, kind):
    if kind == "relu":
        return np.maximum(x, 0.0)
    if kind == "softplus":
        return np.logaddexp(0.0, x)
    raise ValueError(kind)


def dact(x, kind):
    if kind == "relu":
        return (x > 0).astype(x.dtype)
    if kind == "softplus":
        return 1.0 / (1.0 + np.exp(-x))
    raise ValueError(kind)


def init_params(R, S, A, H, rng, q_bias=0.0, z_bias=0.0):
    """Standard Gaussian init (NTK parameterization); output biases start at the value
    scale (q_bias, z_bias)."""
    return {
        "W1": rng.normal(0.0, 1.0, size=(R, H, S)),
        "b1": np.zeros((R, H)),
        "W2": rng.normal(0.0, 1.0, size=(R, A, H)),
        "b2": np.full((R, A), q_bias),
        "U1": rng.normal(0.0, 1.0, size=(R, H, S * A)),
        "c1": np.zeros((R, H)),
        "u2": rng.normal(0.0, 1.0, size=(R, H)),
        "c2": np.full(R, z_bias),
    }


def value_scale(mdp: MDP):
    """Constant initial guesses: Q ~ (rbar + eta log A) / (1 - beta) - eta log A, zeta ~ V."""
    v0 = (mdp.r.mean() + mdp.eta * np.log(mdp.A)) / (1 - mdp.beta)
    return v0 - mdp.eta * np.log(mdp.A), v0


_EYES = {}


def _eye(m):
    if m not in _EYES:
        _EYES[m] = np.eye(m)
    return _EYES[m]


def _q_forward(p, s, kind):
    """s: (R, B) state indices -> pre-activation, hidden, q-values (R, B, A)."""
    R = s.shape[0]
    ridx = np.arange(R)[:, None]
    pre = p["W1"].transpose(0, 2, 1)[ridx, s] + p["b1"][:, None, :]
    h = act(pre, kind) / np.sqrt(p["W1"].shape[1])          # 1/sqrt(H) folded into h
    q = np.matmul(h, p["W2"].transpose(0, 2, 1)) + p["b2"][:, None, :]
    return pre, h, q


def _q_backward(p, s, pre, h, dq, kind, S, B, g):
    g["W2"] += np.matmul(dq.transpose(0, 2, 1), h) / B
    g["b2"] += dq.sum(axis=1) / B
    dh = np.matmul(dq, p["W2"])
    dpre = dh * dact(pre, kind) / np.sqrt(p["W1"].shape[1])
    onehot = _eye(S)[s]                                     # (R, B, S)
    g["W1"] += np.matmul(dpre.transpose(0, 2, 1), onehot) / B
    g["b1"] += dpre.sum(axis=1) / B


def loss_and_grads(p, s, a, sp, rew, mdp: MDP, kind: str, need_grad: bool = True,
                   zeta_override=None):
    """Per-sample objective f(w, v; s, a, s') = delta^2 - beta^2 (V(s') - zeta(s, a))^2,
    averaged over the B samples of each run. Returns (F, grads) with F of shape (R,).
    `zeta_override` (R, B) replaces zeta_v(s, a); it is used only for diagnostics."""
    R, B = s.shape
    S, A, beta, eta = mdp.S, mdp.A, mdp.beta, mdp.eta
    ridx = np.arange(R)[:, None]
    bidx = np.arange(B)[None, :]

    pre_s, h_s, q_s = _q_forward(p, s, kind)
    pre_p, h_p, q_p = _q_forward(p, sp, kind)
    z = q_p / eta
    zmax = z.max(axis=-1, keepdims=True)
    ez = np.exp(z - zmax)
    V = eta * (np.log(ez.sum(axis=-1)) + zmax[..., 0])      # (R, B)
    pi_p = ez / ez.sum(axis=-1, keepdims=True)
    Qsa = q_s[ridx, bidx, a]

    k = s * A + a
    pre_z = p["U1"].transpose(0, 2, 1)[ridx, k] + p["c1"][:, None, :]
    g_z = act(pre_z, kind) / np.sqrt(p["U1"].shape[1])       # 1/sqrt(H) folded into g_z
    zeta = np.matmul(g_z, p["u2"][:, :, None])[..., 0] + p["c2"][:, None]
    if zeta_override is not None:
        zeta = zeta_override

    delta = rew + beta * V - Qsa
    resid = V - zeta
    f = delta ** 2 - beta ** 2 * resid ** 2
    F = f.mean(axis=1)
    if not need_grad:
        return F, None

    g = {key: np.zeros_like(val) for key, val in p.items()}
    # primal block
    dq_s = np.zeros_like(q_s)
    dq_s[ridx, bidx, a] = -2.0 * delta
    dV = 2.0 * beta * delta - 2.0 * beta ** 2 * resid
    dq_p = dV[..., None] * pi_p
    _q_backward(p, s, pre_s, h_s, dq_s, kind, S, B, g)
    _q_backward(p, sp, pre_p, h_p, dq_p, kind, S, B, g)
    # dual block
    dzeta = 2.0 * beta ** 2 * resid                         # (R, B)
    g["u2"] += np.matmul(dzeta[:, None, :], g_z)[:, 0, :] / B
    g["c2"] += dzeta.sum(axis=1) / B
    dg = dzeta[..., None] * p["u2"][:, None, :]
    dpre_z = dg * dact(pre_z, kind) / np.sqrt(p["U1"].shape[1])
    onehot = _eye(S * A)[k]
    g["U1"] += np.matmul(dpre_z.transpose(0, 2, 1), onehot) / B
    g["c1"] += dpre_z.sum(axis=1) / B
    return F, g


def gradient_check(kind="softplus", seed=1):
    mdp = make_mdp(0)
    rng = np.random.default_rng(seed)
    R, B, H = 2, 6, 5
    p = init_params(R, mdp.S, mdp.A, H, rng)
    for key in p:
        p[key] = p[key] + 0.1 * rng.normal(size=p[key].shape)
    s, a, sp = (rng.integers(0, m, size=(R, B)) for m in (mdp.S, mdp.A, mdp.S))
    rew = mdp.r[s, a]
    _, g = loss_and_grads(p, s, a, sp, rew, mdp, kind)
    worst = 0.0
    for key in p:
        flat = p[key].reshape(-1)
        num = np.zeros_like(flat)
        for j in range(flat.size):
            old = flat[j]
            flat[j] = old + 1e-6
            Fp, _ = loss_and_grads(p, s, a, sp, rew, mdp, kind, need_grad=False)
            flat[j] = old - 1e-6
            Fm, _ = loss_and_grads(p, s, a, sp, rew, mdp, kind, need_grad=False)
            flat[j] = old
            num[j] = (Fp - Fm).sum() / 2e-6
        rel = np.max(np.abs(num - g[key].reshape(-1))) / (np.max(np.abs(num)) + 1e-12)
        worst = max(worst, rel)
    return worst


# --------------------------------------------------------------------------------------
# Coupled SGDA
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    width: int = 16
    n: int = 400
    reps: int = 50
    T: int = 40000
    checkpoints: tuple = (5000, 10000, 20000, 40000)
    B: int = 32
    c1: float = 400.0          # eta_t = c1 / (c2 + t): about 0.2 for t << c2, then ~ c1 / t
    c2: float = 2000.0
    kappa_v: float = 2.0       # dual stepsize multiplier
    act: str = "softplus"
    data: str = "iid"          # "iid" or "trajectory"
    mdp_seed: int = 0
    seed: int = 0


def block_norm(p, q, keys):
    return np.sqrt(sum(((p[k] - q[k]) ** 2).reshape(p[k].shape[0], -1).sum(axis=1) for k in keys))


def q_table(p, mdp: MDP, kind: str):
    """Q_w(s, a) for every state-action pair: (R, S, A)."""
    R = p["W1"].shape[0]
    states = np.tile(np.arange(mdp.S), (R, 1))
    _, _, q = _q_forward(p, states, kind)
    return q


def soft_v(q, eta):
    m = q.max(axis=-1, keepdims=True)
    return eta * (np.log(np.exp((q - m) / eta).sum(axis=-1)) + m[..., 0] / eta)


def population_msbe(p, mdp: MDP, kind: str):
    """Exact population MSBE E_mu[(TQ - Q)^2]; equals the population primal risk R(w)
    when the auxiliary class realizes E[V_Q(s') | s, a] (true here: one-hot inputs)."""
    q = q_table(p, mdp, kind)
    V = soft_v(q, mdp.eta)                                       # (R, S)
    TQ = mdp.r[None] + mdp.beta * np.einsum("sat,rt->rsa", mdp.P, V)
    return (mdp.mu[None] * (TQ - q) ** 2).sum(axis=(1, 2)), q


def oracle_msbe(S_, A_, SP, mdp: MDP, iters: int = 3000):
    """Population MSBE of the exact empirical minimizer for a tabular Q (fixed point of the
    empirical soft Bellman operator). NaN if some (s, a) pair is absent from the dataset."""
    R = S_.shape[0]
    out = np.full(R, np.nan)
    for j in range(R):
        cnt = np.zeros((mdp.S, mdp.A, mdp.S))
        np.add.at(cnt, (S_[j], A_[j], SP[j]), 1.0)
        tot = cnt.sum(axis=2, keepdims=True)
        if (tot == 0).any():
            continue
        Ph = cnt / tot
        Q = np.zeros((mdp.S, mdp.A))
        for _ in range(iters):
            Q = mdp.r + mdp.beta * Ph @ soft_v(Q, mdp.eta)
        TQ = mdp.r + mdp.beta * mdp.P @ soft_v(Q, mdp.eta)
        out[j] = float((mdp.mu * (TQ - Q) ** 2).sum())
    return out


def empirical_primal_risk(q, S_, A_, SP, mdp: MDP):
    """R_n(w) = max_zeta F_D(w, zeta) over all functions zeta of (s, a): the maximizer is the
    empirical conditional mean of V_Q(s') given (s, a)."""
    R, n = S_.shape
    V = soft_v(q, mdp.eta)                                       # (R, S)
    ridx = np.arange(R)[:, None]
    Vp = V[ridx, SP]
    delta = mdp.r[S_, A_] + mdp.beta * Vp - q[ridx, S_, A_]
    k = S_ * mdp.A + A_
    K = mdp.S * mdp.A
    flat = (ridx * K + k).ravel()
    cnt = np.bincount(flat, minlength=R * K).reshape(R, K)
    sm = np.bincount(flat, weights=Vp.ravel(), minlength=R * K).reshape(R, K)
    mhat = sm / np.maximum(cnt, 1)
    resid = Vp - mhat[ridx, k]
    return (delta ** 2).mean(axis=1) - mdp.beta ** 2 * (resid ** 2).mean(axis=1)


def run_config(cfg: Config, mdp: MDP | None = None, return_params: bool = False):
    mdp = mdp or make_mdp(cfg.mdp_seed)
    rng = np.random.default_rng(cfg.seed)
    reps, n, R = cfg.reps, cfg.n, 2 * cfg.reps
    sampler = sample_iid if cfg.data == "iid" else sample_trajectory

    S_ = np.empty((R, n), dtype=int)
    A_ = np.empty((R, n), dtype=int)
    SP = np.empty((R, n), dtype=int)
    for j in range(reps):
        s, a, sp = sampler(mdp, n, rng)
        i = rng.integers(n)
        st, at, spt = sample_iid(mdp, 1, rng)              # independent copy z~_i ~ mu x P
        S_[2 * j], A_[2 * j], SP[2 * j] = s, a, sp
        s2, a2, sp2 = s.copy(), a.copy(), sp.copy()
        s2[i], a2[i], sp2[i] = st[0], at[0], spt[0]
        S_[2 * j + 1], A_[2 * j + 1], SP[2 * j + 1] = s2, a2, sp2
    REW = mdp.r[S_, A_]

    qb, zb = value_scale(mdp)
    p0 = init_params(reps, mdp.S, mdp.A, cfg.width, rng, q_bias=qb, z_bias=zb)
    p = {k: np.repeat(v, 2, axis=0) for k, v in p0.items()}   # shared initialization

    ridx = np.arange(R)[:, None]
    out = {"eps_w": {}, "eps_v": {}, "eps_q": {}, "msbe": {}, "gen_gap": {},
           "gradnorm_w": {}, "gradnorm_v": {}}
    cps = set(cfg.checkpoints)
    for t in range(cfg.T):
        I = rng.integers(0, n, size=(reps, cfg.B))
        I = np.repeat(I, 2, axis=0)                            # shared index sequence
        s, a, sp, rw = S_[ridx, I], A_[ridx, I], SP[ridx, I], REW[ridx, I]
        _, g = loss_and_grads(p, s, a, sp, rw, mdp, cfg.act)
        lr = cfg.c1 / (cfg.c2 + t)
        for k in W_KEYS:
            p[k] -= lr * g[k]
        for k in V_KEYS:
            p[k] += cfg.kappa_v * lr * g[k]
        if (t + 1) in cps:
            pa = {k: v[0::2] for k, v in p.items()}
            pb = {k: v[1::2] for k, v in p.items()}
            out["eps_w"][t + 1] = block_norm(pa, pb, W_KEYS).tolist()
            out["eps_v"][t + 1] = block_norm(pa, pb, V_KEYS).tolist()
            msbe, q = population_msbe(p, mdp, cfg.act)
            dq = q[0::2] - q[1::2]
            out["eps_q"][t + 1] = np.sqrt((mdp.mu[None] * dq ** 2).sum(axis=(1, 2))).tolist()
            out["msbe"][t + 1] = msbe[0::2].tolist()                       # runs on D
            Rn = empirical_primal_risk(q, S_, A_, SP, mdp)
            out["gen_gap"][t + 1] = (msbe[0::2] - Rn[0::2]).tolist()
            # full-batch gradient norms on the run's own dataset (optimization diagnostic)
            _, gf = loss_and_grads(p, S_, A_, SP, REW, mdp, cfg.act)
            gw = np.sqrt(sum((gf[k] ** 2).reshape(R, -1).sum(1) for k in W_KEYS))
            gv = np.sqrt(sum((gf[k] ** 2).reshape(R, -1).sum(1) for k in V_KEYS))
            out["gradnorm_w"][t + 1] = float(np.median(gw))
            out["gradnorm_v"][t + 1] = float(np.median(gv))
    out["oracle_msbe"] = oracle_msbe(S_[0::2], A_[0::2], SP[0::2], mdp).tolist()
    if return_params:
        return out, p, (S_, A_, SP, REW), mdp
    return out


def _run_one(args):
    cfg_dict = args
    cfg = Config(**cfg_dict)
    t0 = time.time()
    res = run_config(cfg)
    res["config"] = asdict(cfg)
    res["seconds"] = time.time() - t0
    return res


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--ns", type=int, nargs="+", default=[100, 200, 400, 800, 1600, 3200, 6400])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--T", type=int, default=40000)
    ap.add_argument("--act", default="softplus")
    ap.add_argument("--data", default="iid")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=2)
    args = ap.parse_args()

    print("gradient check (max rel. error):", gradient_check(args.act), flush=True)
    jobs = []
    for H in args.widths:
        for n in args.ns:
            cfg = Config(width=H, n=n, reps=args.reps, T=args.T, act=args.act, data=args.data,
                         seed=args.seed + 1000 * H + n)
            cfg.checkpoints = tuple(c for c in (5000, 10000, 20000, 40000, 80000) if c <= args.T)
            jobs.append(asdict(cfg))
    jobs.sort(key=lambda c: -c["width"])          # longest jobs first
    with mp.Pool(args.procs) as pool, open(args.out, "a") as fh:
        for res in pool.imap_unordered(_run_one, jobs):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            c, T = res["config"], str(res["config"]["T"])
            T = res["config"]["checkpoints"][-1]
            print(f"H={c['width']:3d} n={c['n']:5d} data={c['data']}  "
                  f"eps_w={np.mean(res['eps_w'][T]):.3e} eps_q={np.mean(res['eps_q'][T]):.3e} "
                  f"msbe={np.mean(res['msbe'][T]):.3e} oracle={np.nanmean(res['oracle_msbe']):.3e} "
                  f"({res['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
