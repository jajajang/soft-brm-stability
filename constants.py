"""Empirical estimates of the constants in Theorems 3 and 7 along SGDA trajectories.

For a few datasets we run the same SGDA protocol as brm_stability.py, keep parameter
snapshots, and estimate on the run's own dataset D:
  mu_PL  : min over snapshots of  0.5 ||grad Phi_D(w)||^2 / (Phi_D(w) - Phi_D^*)
  rho    : min over snapshots of  0.5 ||grad_v F_D(w, v)||^2 / (Phi_D(w) - F_D(w, v))
  mu_QG  : min over snapshots of  (Phi_D(w) - Phi_D^*) / (0.5 ||w - w_ref||^2)
  G      : max per-sample gradient norm (both blocks) over snapshots
  L      : max |eigenvalue| of the joint Hessian of F_D over snapshots (power iteration)
Phi_D(w) = max over all functions zeta of F_D(w, zeta) is attained at the empirical
conditional mean of V_Q(s'); its minimum over Q is 0 (empirical fixed point), and the
networks used here can represent it, so Phi_D^* = 0. w_ref is the final iterate of a run
four times longer, used as a proxy for the (non-unique) empirical minimizer.
"""

import json
import sys

import numpy as np

import brm_stability as b


def grad_phi(p, S_, A_, SP, REW, mdp, kind):
    """grad Phi_D(w) = grad_w F_D(w, zeta*) with zeta* the empirical conditional mean."""
    R, n = S_.shape
    q = b.q_table(p, mdp, kind)
    V = b.soft_v(q, mdp.eta)
    ridx = np.arange(R)[:, None]
    Vp = V[ridx, SP]
    k = S_ * mdp.A + A_
    K = mdp.S * mdp.A
    flat = (ridx * K + k).ravel()
    cnt = np.bincount(flat, minlength=R * K).reshape(R, K)
    sm = np.bincount(flat, weights=Vp.ravel(), minlength=R * K).reshape(R, K)
    zstar = (sm / np.maximum(cnt, 1))[ridx, k]
    # same algebra as loss_and_grads with zeta replaced by zstar (primal block only)
    pre_s, h_s, q_s = b._q_forward(p, S_, kind)
    pre_p, h_p, q_p = b._q_forward(p, SP, kind)
    z = q_p / mdp.eta
    ez = np.exp(z - z.max(axis=-1, keepdims=True))
    pi_p = ez / ez.sum(axis=-1, keepdims=True)
    bidx = np.arange(n)[None, :]
    delta = REW + mdp.beta * Vp - q_s[ridx, bidx, A_]
    g = {key: np.zeros_like(val) for key, val in p.items()}
    dq_s = np.zeros_like(q_s)
    dq_s[ridx, bidx, A_] = -2.0 * delta
    dV = 2.0 * mdp.beta * delta - 2.0 * mdp.beta ** 2 * (Vp - zstar)
    b._q_backward(p, S_, pre_s, h_s, dq_s, kind, mdp.S, n, g)
    b._q_backward(p, SP, pre_p, h_p, dV[..., None] * pi_p, kind, mdp.S, n, g)
    return {k_: g[k_] for k_ in b.W_KEYS}


def sqnorm(d, keys):
    return sum((d[k] ** 2).reshape(d[k].shape[0], -1).sum(axis=1) for k in keys)


def per_sample_gmax(p, S_, A_, SP, REW, mdp, kind):
    """max over samples of ||grad_w f|| and ||grad_v f|| for each run (R,)."""
    R, n = S_.shape
    gw_max = np.zeros(R)
    gv_max = np.zeros(R)
    for r in range(R):
        pr = {k: np.repeat(v[r:r + 1], n, axis=0) for k, v in p.items()}
        _, g = b.loss_and_grads(pr, S_[r][:, None], A_[r][:, None], SP[r][:, None],
                                REW[r][:, None], mdp, kind)
        gw_max[r] = np.sqrt(sqnorm(g, b.W_KEYS)).max()
        gv_max[r] = np.sqrt(sqnorm(g, b.V_KEYS)).max()
    return gw_max, gv_max


def hessian_norm(p, S_, A_, SP, REW, mdp, kind, iters=30, eps=1e-5, seed=0):
    """Largest |eigenvalue| of the joint Hessian of F_D via finite-difference HVPs."""
    rng = np.random.default_rng(seed)
    keys = list(p.keys())
    sign = {k: (1.0 if k in b.W_KEYS else 1.0) for k in keys}

    def full_grad(pp):
        _, g = b.loss_and_grads(pp, S_, A_, SP, REW, mdp, kind)
        return g

    vec = {k: rng.normal(size=p[k].shape) for k in keys}
    nrm = np.sqrt(sqnorm(vec, keys))
    lam = np.zeros(p["W1"].shape[0])
    for _ in range(iters):
        vec = {k: vec[k] / nrm.reshape((-1,) + (1,) * (vec[k].ndim - 1)) for k in keys}
        pp = {k: p[k] + eps * vec[k] for k in keys}
        pm = {k: p[k] - eps * vec[k] for k in keys}
        gp, gm = full_grad(pp), full_grad(pm)
        hv = {k: sign[k] * (gp[k] - gm[k]) / (2 * eps) for k in keys}
        nrm = np.sqrt(sqnorm(hv, keys))
        lam = nrm
        vec = hv
    return lam


def main(width=16, n=800, reps=5, T=40000, n_snap=20, seed=7):
    mdp = b.make_mdp(0)
    kind = "softplus"
    rng = np.random.default_rng(seed)
    R = reps
    S_ = np.empty((R, n), dtype=int)
    A_ = np.empty((R, n), dtype=int)
    SP = np.empty((R, n), dtype=int)
    for j in range(R):
        S_[j], A_[j], SP[j] = b.sample_iid(mdp, n, rng)
    REW = mdp.r[S_, A_]
    qb, zb = b.value_scale(mdp)
    p = b.init_params(R, mdp.S, mdp.A, width, rng, q_bias=qb, z_bias=zb)
    cfg = b.Config(width=width, n=n, reps=reps, T=T)
    ridx = np.arange(R)[:, None]
    snap_at = set(np.unique(np.geomspace(200, T, n_snap).astype(int)))
    snaps = []
    T_ref = 4 * T
    for t in range(T_ref):
        I = rng.integers(0, n, size=(R, cfg.B))
        _, g = b.loss_and_grads(p, S_[ridx, I], A_[ridx, I], SP[ridx, I], REW[ridx, I], mdp, kind)
        lr = cfg.c1 / (cfg.c2 + t)
        for k in b.W_KEYS:
            p[k] -= lr * g[k]
        for k in b.V_KEYS:
            p[k] += cfg.kappa_v * lr * g[k]
        if (t + 1) in snap_at:
            snaps.append((t + 1, {k: v.copy() for k, v in p.items()}))
    w_ref = {k: p[k].copy() for k in b.W_KEYS}

    rows = []
    for t, ps in snaps:
        q = b.q_table(ps, mdp, kind)
        phi = b.empirical_primal_risk(q, S_, A_, SP, mdp)            # Phi_D(w) - Phi_D^*
        F, gfull = b.loss_and_grads(ps, S_, A_, SP, REW, mdp, kind)
        gphi = grad_phi(ps, S_, A_, SP, REW, mdp, kind)
        gap_in = np.maximum(phi - F, 1e-300)
        mu_pl = 0.5 * sqnorm(gphi, b.W_KEYS) / np.maximum(phi, 1e-300)
        rho = 0.5 * sqnorm(gfull, b.V_KEYS) / gap_in
        dist2 = sqnorm({k: ps[k] - w_ref[k] for k in b.W_KEYS}, b.W_KEYS)
        mu_qg = phi / np.maximum(0.5 * dist2, 1e-300)
        gw, gv = per_sample_gmax(ps, S_, A_, SP, REW, mdp, kind)
        L = hessian_norm(ps, S_, A_, SP, REW, mdp, kind)
        rows.append(dict(t=int(t), phi=phi.tolist(), inner_gap=(phi - F).tolist(),
                         mu_pl=mu_pl.tolist(), rho=rho.tolist(), mu_qg=mu_qg.tolist(),
                         G=np.maximum(gw, gv).tolist(), L=L.tolist()))
        print(f"t={t:6d} phi={np.median(phi):.2e} mu_PL={np.median(mu_pl):.2e} "
              f"rho={np.median(rho):.2e} mu_QG={np.median(mu_qg):.2e} "
              f"G={np.median(np.maximum(gw, gv)):.2f} L={np.median(L):.2f}", flush=True)
    return rows


if __name__ == "__main__":
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 800
    out = sys.argv[3] if len(sys.argv) > 3 else "constants.json"
    rows = main(width=width, n=n)
    with open(out, "w") as fh:
        json.dump(dict(width=width, n=n, rows=rows), fh)
