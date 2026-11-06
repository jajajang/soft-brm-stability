"""Summaries, slopes with bootstrap CIs, LaTeX tables and the figure for Appendix E."""

import json
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# validated categorical slots 1-3 (light surface), text and neutral tokens
SERIES = {8: "#2a78d6", 16: "#eb6834", 32: "#1baf7a"}
MARKERS = {8: "o", 16: "s", 32: "^"}
TEXT, TEXT2, GRID, REF = "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def final(res, key):
    T = str(res["config"]["checkpoints"][-1])
    return np.asarray(res[key][T], dtype=float)


def at(res, key, T):
    return np.asarray(res[key][str(T)], dtype=float)


def slope_ci(ns, samples, n_boot=4000, seed=0):
    """OLS slope of log(mean) on log(n); 95% CI by resampling repetitions within each n."""
    rng = np.random.default_rng(seed)
    x = np.log(ns)
    y = np.log([np.nanmean(s) for s in samples])
    slope = np.polyfit(x, y, 1)[0]
    boots = []
    for _ in range(n_boot):
        yb = [np.log(np.nanmean(rng.choice(s[~np.isnan(s)], size=(~np.isnan(s)).sum()))) for s in samples]
        boots.append(np.polyfit(x, yb, 1)[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return slope, lo, hi


def summarize(rows, data="iid", n_min=200):
    table = {}
    for r in rows:
        c = r["config"]
        if c["data"] != data:
            continue
        table[(c["width"], c["n"])] = r
    widths = sorted({k[0] for k in table})
    out = {}
    for H in widths:
        ns = sorted(n for (h, n) in table if h == H)
        entry = {"n": ns}
        for key in ("eps_w", "eps_q", "eps_v", "msbe", "gen_gap"):
            vals = [final(table[(H, n)], key) for n in ns]
            entry[key + "_mean"] = [float(np.mean(v)) for v in vals]
            entry[key + "_se"] = [float(np.std(v, ddof=1) / np.sqrt(len(v))) for v in vals]
            sel = [i for i, n in enumerate(ns) if n >= n_min]
            if key != "gen_gap" or all(entry[key + "_mean"][i] > 0 for i in sel):
                s, lo, hi = slope_ci(np.array(ns)[sel], [vals[i] for i in sel])
                entry[key + "_slope"] = [float(s), float(lo), float(hi)]
        orc = [np.asarray(table[(H, n)]["oracle_msbe"], dtype=float) for n in ns]
        entry["oracle_mean"] = [float(np.nanmean(o)) for o in orc]
        entry["oracle_nan"] = [int(np.isnan(o).sum()) for o in orc]
        sel = [i for i, n in enumerate(ns) if n >= n_min]
        s, lo, hi = slope_ci(np.array(ns)[sel], [orc[i] for i in sel])
        entry["oracle_slope"] = [float(s), float(lo), float(hi)]
        # optimization check: change of eps_w between the last two checkpoints
        cps = table[(H, ns[0])]["config"]["checkpoints"]
        entry["eps_w_T_ratio"] = [float(np.mean(at(table[(H, n)], "eps_w", cps[-1])) /
                                        np.mean(at(table[(H, n)], "eps_w", cps[-2]))) for n in ns]
        entry["msbe_T_ratio"] = [float(np.mean(at(table[(H, n)], "msbe", cps[-1])) /
                                       np.mean(at(table[(H, n)], "msbe", cps[-2]))) for n in ns]
        out[H] = entry
    return out


def fmt(x):
    m, e = f"{x:.2e}".split("e")
    return f"{m}\\text{{e}}{int(e)}"


def latex_tables(summ, path):
    widths = sorted(summ)
    ns = summ[widths[0]]["n"]
    lines = []
    lines.append("% Table E.1: stability (primal parameters) and n * eps")
    lines.append("\\begin{tabular}{@{}r" + "cc" * len(widths) + "@{}}")
    lines.append("\\toprule")
    lines.append(" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{width $H={H}$}}" for H in widths) + " \\\\")
    lines.append("$n$ & " + " & ".join("$\\varepsilon^w_T$ & $n\\,\\varepsilon^w_T$" for _ in widths) + " \\\\")
    lines.append("\\midrule")
    for i, n in enumerate(ns):
        cells = []
        for H in widths:
            m, se = summ[H]["eps_w_mean"][i], summ[H]["eps_w_se"][i]
            cells.append(f"${fmt(m)}$ & ${n * m:.2f}$")
        lines.append(f"{n} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    lines.append("slope & " + " & ".join(
        f"\\multicolumn{{2}}{{c}}{{${summ[H]['eps_w_slope'][0]:.2f}$ "
        f"$[{summ[H]['eps_w_slope'][1]:.2f}, {summ[H]['eps_w_slope'][2]:.2f}]$}}" for H in widths) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")
    lines.append("% Table E.2: excess population MSBE of the SGDA output vs the exact empirical minimizer")
    lines.append("\\begin{tabular}{@{}r" + "c" * len(widths) + "c@{}}")
    lines.append("\\toprule")
    lines.append("$n$ & " + " & ".join(f"SGDA, $H={H}$" for H in widths) + " & oracle \\\\")
    lines.append("\\midrule")
    for i, n in enumerate(ns):
        cells = [f"${fmt(summ[H]['msbe_mean'][i])}$" for H in widths]
        lines.append(f"{n} & " + " & ".join(cells) + f" & ${fmt(summ[widths[0]]['oracle_mean'][i])}$ \\\\")
    lines.append("\\midrule")
    lines.append("slope & " + " & ".join(
        f"${summ[H]['msbe_slope'][0]:.2f}$" for H in widths) + f" & ${summ[widths[0]]['oracle_slope'][0]:.2f}$ \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def figure(summ, summ_traj, path):
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": TEXT2, "axes.labelcolor": TEXT,
                         "xtick.color": TEXT2, "ytick.color": TEXT2, "font.family": "serif"})
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3), constrained_layout=True)
    panels = [("eps_w", r"$\varepsilon^w_T$ (parameters)"),
              ("eps_q", r"$\|Q_{w_T}-Q_{w'_T}\|_{\mu}$ (function)"),
              ("msbe", "excess population MSBE")]
    for ax, (key, title) in zip(axes, panels):
        for H in sorted(summ):
            e = summ[H]
            ax.plot(e["n"], e[key + "_mean"], color=SERIES[H], lw=1.5, marker=MARKERS[H], ms=4,
                    mec="white", mew=0.6, label=f"H={H}, i.i.d.")
        if summ_traj and key != "eps_v":
            for H in sorted(summ_traj):
                e = summ_traj[H]
                ax.plot(e["n"], e[key + "_mean"], color=SERIES[H], lw=1.2, ls=(0, (4, 2)),
                        marker=MARKERS[H], ms=4, mfc="white", mew=0.9,
                        label=f"H={H}, single trajectory")
        if key == "msbe":
            H0 = sorted(summ)[0]
            ax.plot(summ[H0]["n"], summ[H0]["oracle_mean"], color=TEXT2, lw=1.0, ls=":",
                    marker="x", ms=4, label="exact empirical minimizer")
        # reference slopes anchored at the largest-width curve's first point
        e = summ[sorted(summ)[-1]]
        n0, y0 = e["n"][0], e[key + "_mean"][0]
        ns = np.array([e["n"][0], e["n"][-1]], dtype=float)
        y1 = 0.45 * y0                       # n^{-1} guide sits below the data
        ax.plot(ns, y1 * (ns / n0) ** -1.0, color=REF, lw=0.8, ls="--")
        ax.plot(ns, y0 * (ns / n0) ** -0.5, color=REF, lw=0.8, ls="-.")
        ax.text(ns[-1], y1 * (ns[-1] / n0) ** -1.0, r" $n^{-1}$", color=TEXT2, va="center", fontsize=7)
        ax.text(ns[-1], y0 * (ns[-1] / n0) ** -0.5, r" $n^{-1/2}$", color=TEXT2, va="center", fontsize=7)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$n$")
        ax.set_title(title, fontsize=8, color=TEXT)
        ax.grid(True, which="major", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(ns[0] / 1.3, ns[-1] * 2.2)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, fontsize=7,
               columnspacing=1.2, handlelength=2.6)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    iid = load(sys.argv[1])
    traj = load(sys.argv[2]) if len(sys.argv) > 2 else []
    summ = summarize(iid, "iid")
    summ_t = summarize(traj, "trajectory") if traj else {}
    json.dump({"iid": summ, "trajectory": summ_t}, open("summary.json", "w"), indent=1)
    latex_tables(summ, "tables.tex")
    figure(summ, summ_t, "stability_vs_n.pdf")
    for name, S in (("iid", summ), ("trajectory", summ_t)):
        for H, e in S.items():
            print(name, H, "eps_w slope", np.round(e["eps_w_slope"], 2), "eps_q slope", np.round(e["eps_q_slope"], 2),
                  "msbe slope", np.round(e["msbe_slope"], 2), "oracle slope", np.round(e["oracle_slope"], 2))
            print("   n*eps_w", np.round(np.array(e["n"]) * np.array(e["eps_w_mean"]), 3))
            print("   n*msbe ", np.round(np.array(e["n"]) * np.array(e["msbe_mean"]), 3),
                  " oracle n*", np.round(np.array(e["n"]) * np.array(e["oracle_mean"]), 3), "nan", e["oracle_nan"])
            print("   T-ratio eps_w", np.round(e["eps_w_T_ratio"], 3), " msbe", np.round(e["msbe_T_ratio"], 3))
