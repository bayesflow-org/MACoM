"""
Prior predictive equivalence check for the turbocharged meta-simulator.

The speedup (commit 90a61a5, "Turbocharge meta-simulator") replaced the per-step
scalar ``np.random.normal()`` draws in the DDM / RDM / CDM diffusion kernels with
block-pregenerated standard-normal noise, and split each model into a fast *serial*
kernel (used when numba runs single-threaded — the training sim-worker config) and
the original *parallel* ``prange`` kernel (the reference). This module is the
regression test that the serial kernels still sample from the *same* distribution
as the reference kernels.

For each model it:
  1. Draws N independent parameter sets from the prior and pushes them through the
     link functions to natural scale — the prior predictive over parameters.
  2. Simulates one trial per parameter set with BOTH kernels on identical prior
     draws (contamination excluded so the diffusion kernel is tested in isolation).
  3. Compares the resulting DATA distributions: no-decision (timeout) rate, the RT
     distribution (2-sample KS + a Q-Q plot), and the response variable (choice
     proportion for DDM/RDM, response angle for CDM).

If the speedup is bug-free the two kernels are just different RNG realizations of
one generative distribution, so every comparison agrees within Monte-Carlo error
and the script exits 0. A real discrepancy trips a threshold and exits 1.

Usage:
    python -m experiments.check_simulator                       # all models, save figure
    python -m experiments.check_simulator --models ddm rdm      # subset
    python -m experiments.check_simulator --n 500000 --seed 7   # tighter check
    python -m experiments.check_simulator --no-plot             # text report only
"""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cogformer.simulator.benchmarks.ssms.priors import ddm_priors, rdm_priors, cdm_priors
from cogformer.simulator.benchmarks.ssms.link_funs import ddm_link_fun, rdm_link_fun, cdm_link_fun
from cogformer.simulator.benchmarks.ssms.ddm import _simulate_ddm_serial, _simulate_ddm_parallel
from cogformer.simulator.benchmarks.ssms.rdm import _simulate_rdm_serial_m2, _simulate_rdm_parallel
from cogformer.simulator.benchmarks.ssms.cdm import _simulate_cdm_serial, _simulate_cdm_parallel

# Defaults mirror the production model classes (DDM/RDM/CDM __init__).
DT = 5e-4
MAX_STEPS = 20_000

# Pass thresholds. The KS D floor tracks the alpha=0.05 two-sample critical value,
# which at N=200k is ~0.0043; 0.02 leaves comfortable headroom for MC noise while
# still catching any real distributional shift. The no-decision-rate and response
# gaps are set well outside their sampling error at the default N.
KS_D_MAX = 0.02
MISS_RATE_MAX = 0.01
RESP_GAP_MAX = 0.01

# Reference (parallel) = blue, New (serial) = orange. CVD-validated pair
# (ΔE ~97 across protan/deutan/tritan); linestyle is the secondary encoding.
C_REF = "#2a78d6"
C_NEW = "#eb6834"


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
def _draw_natural(priors, link, n, rng):
    """Draw n unconstrained param sets from the prior and map to natural scale."""
    out = {}
    for k, dist in priors.items():
        if k == "pi":
            continue  # contamination is applied outside the kernel; excluded here
        out[k] = link[k](dist.rvs(size=n, random_state=rng))
    return out


def _simulate_ddm(n, rng):
    p = _draw_natural(ddm_priors(), ddm_link_fun(), n, rng)
    args = (p["v"].astype(np.float64), p["a"].astype(np.float64), p["z"].astype(np.float64),
            p["tau"].astype(np.float64), p["s_v"].astype(np.float64), p["s_tau"].astype(np.float64),
            1.0, DT, MAX_STEPS)
    return _simulate_ddm_parallel(*args), _simulate_ddm_serial(*args)


def _simulate_rdm(n, rng):
    p = _draw_natural(rdm_priors(), rdm_link_fun(), n, rng)
    v = np.zeros((n, 2))
    v[:, 0] = p["v"]
    v[:, 1] = p["v"] + p["v_diff"]
    a, tau = p["a"].astype(np.float64), p["tau"].astype(np.float64)
    s_v, s_tau = p["s_v"].astype(np.float64), p["s_tau"].astype(np.float64)
    ref = _simulate_rdm_parallel(v, a, tau, s_v, s_tau, 0.0, 1.0, DT, MAX_STEPS)
    new = _simulate_rdm_serial_m2(v, a, tau, s_v, s_tau, 1.0, DT, MAX_STEPS)
    return ref, new


def _simulate_cdm(n, rng):
    p = _draw_natural(cdm_priors(), cdm_link_fun(), n, rng)
    v = np.stack([p["v"] * np.cos(p["v_theta"]), p["v"] * np.sin(p["v_theta"])], axis=1)
    a, tau = p["a"].astype(np.float64), p["tau"].astype(np.float64)
    s_v, s_tau = p["s_v"].astype(np.float64), p["s_tau"].astype(np.float64)
    ref = _simulate_cdm_parallel(v, a, tau, s_v, s_tau, 0.0, 1.0, DT, MAX_STEPS)
    new = _simulate_cdm_serial(v, a, tau, s_v, s_tau, 1.0, DT, MAX_STEPS)
    return ref, new


MODELS = {
    "ddm": {"name": "DDM",              "sim": _simulate_ddm, "resp": "choice", "circular": False},
    "rdm": {"name": "RDM (2-alt)",      "sim": _simulate_rdm, "resp": "choice", "circular": False},
    "cdm": {"name": "CDM",              "sim": _simulate_cdm, "resp": "angle",  "circular": True},
}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(model_name, n, rng):
    """Simulate with both kernels and return arrays + comparison metrics."""
    cfg = MODELS[model_name]
    ref, new = cfg["sim"](n, rng)
    miss = -4.0 if cfg["circular"] else -1.0  # sentinel differs for the circular model

    ref_valid = ref[:, 0] > 0
    new_valid = new[:, 0] > 0
    rt_ref, rt_new = ref[ref_valid, 0], new[new_valid, 0]
    resp_ref, resp_new = ref[ref_valid, 1], new[new_valid, 1]

    D, p = ks_2samp(rt_ref, rt_new)
    miss_ref, miss_new = 1.0 - ref_valid.mean(), 1.0 - new_valid.mean()

    if cfg["circular"]:
        Dr, _ = ks_2samp(resp_ref, resp_new)
        resp_gap = Dr  # circular response compared by its own KS statistic
    else:
        resp_gap = abs(resp_new.mean() - resp_ref.mean())

    passed = (D < KS_D_MAX) and (abs(miss_new - miss_ref) < MISS_RATE_MAX) and (resp_gap < RESP_GAP_MAX)

    return {
        "model": cfg["name"], "circular": cfg["circular"], "resp": cfg["resp"],
        "rt_ref": rt_ref, "rt_new": rt_new, "resp_ref": resp_ref, "resp_new": resp_new,
        "ks_D": D, "ks_p": p, "miss_ref": miss_ref, "miss_new": miss_new,
        "resp_gap": resp_gap, "n": n, "passed": passed,
    }


def print_report(m):
    tag = "PASS" if m["passed"] else "FAIL"
    print(f"\n{'='*66}\n{m['model']}   [{tag}]\n{'='*66}")
    print(f"no-decision rate:  ref={m['miss_ref']:.4%}  new={m['miss_new']:.4%}  "
          f"Δ={m['miss_new']-m['miss_ref']:+.4%}")
    print(f"RT KS 2-sample:    D={m['ks_D']:.4f}  p={m['ks_p']:.3f}   (D<{KS_D_MAX} to pass)")
    qs = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    qr, qn = np.quantile(m["rt_ref"], qs), np.quantile(m["rt_new"], qs)
    print("RT quantiles (s):  " + "  ".join(f"q{int(q*100):02d}" for q in qs))
    print("          ref:     " + "  ".join(f"{x:.3f}" for x in qr))
    print("          new:     " + "  ".join(f"{x:.3f}" for x in qn))
    if m["circular"]:
        print(f"angle θ:           KS D={m['resp_gap']:.4f}  "
              f"mean ref={m['resp_ref'].mean():+.4f}  new={m['resp_new'].mean():+.4f}")
    else:
        print(f"P(response=1):     ref={m['resp_ref'].mean():.4%}  new={m['resp_new'].mean():.4%}  "
              f"Δ={m['resp_new'].mean()-m['resp_ref'].mean():+.4%}")


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def plot(results, outpath):
    """One row per model: RT overlay | RT Q-Q (ref vs new) | response comparison.

    The Q-Q panel is the headline: identically-distributed samples land on the
    y=x line. The RT overlay and response panel show the same conclusion in the
    reader's native units.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 130,
    })

    nrows = len(results)
    fig, axes = plt.subplots(nrows, 3, figsize=(11, 2.9 * nrows), squeeze=False)

    for r, m in enumerate(results):
        ok = m["passed"]
        rt_ref, rt_new = m["rt_ref"], m["rt_new"]

        # --- Col 0: RT distribution overlay -------------------------------- #
        ax = axes[r][0]
        hi = np.quantile(np.concatenate([rt_ref, rt_new]), 0.995)
        bins = np.linspace(0, hi, 80)
        ax.hist(rt_ref, bins=bins, density=True, histtype="step", lw=1.6,
                color=C_REF, label="reference (parallel)")
        ax.hist(rt_new, bins=bins, density=True, histtype="step", lw=1.4,
                color=C_NEW, ls="--", label="new (serial)")
        ax.set_xlim(0, hi)
        ax.set_xlabel("RT (s)")
        ax.set_ylabel("density")
        ax.set_title(f"{m['model']} — RT distribution")
        ax.spines[["top", "right"]].set_visible(False)
        if r == 0:
            ax.legend(frameon=False, fontsize=7)

        # --- Col 1: RT Q-Q plot (the headline) ----------------------------- #
        ax = axes[r][1]
        qs = np.linspace(0.005, 0.995, 220)
        qr, qn = np.quantile(rt_ref, qs), np.quantile(rt_new, qs)
        lim = [0, max(qr.max(), qn.max())]
        ax.plot(lim, lim, color="#8a8a86", lw=1.0, zorder=1)  # y = x
        ax.scatter(qr, qn, s=6, color=C_NEW, alpha=0.7, edgecolor="none", zorder=2)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("reference RT quantile (s)")
        ax.set_ylabel("new RT quantile (s)")
        ax.set_title(f"RT Q-Q   ·   KS D={m['ks_D']:.4f} (p={m['ks_p']:.2f})")
        ax.spines[["top", "right"]].set_visible(False)

        # --- Col 2: response comparison ------------------------------------ #
        ax = axes[r][2]
        if m["circular"]:
            b = np.linspace(-np.pi, np.pi, 60)
            ax.hist(m["resp_ref"], bins=b, density=True, histtype="step", lw=1.6, color=C_REF)
            ax.hist(m["resp_new"], bins=b, density=True, histtype="step", lw=1.4, color=C_NEW, ls="--")
            ax.set_xlim(-np.pi, np.pi)
            ax.set_xlabel("response angle θ (rad)")
            ax.set_ylabel("density")
            ax.set_title(f"θ distribution   ·   KS D={m['resp_gap']:.4f}")
        else:
            levels = np.array([0, 1])
            pr = np.array([(m["resp_ref"] == lv).mean() for lv in levels])
            pn = np.array([(m["resp_new"] == lv).mean() for lv in levels])
            w = 0.38
            ax.bar(levels - w / 2, pr, w, color=C_REF, label="reference")
            ax.bar(levels + w / 2, pn, w, color=C_NEW, label="new", hatch="///",
                   edgecolor="white", linewidth=0)
            ax.set_xticks(levels)
            ax.set_xlabel("response")
            ax.set_ylabel("proportion")
            signed = m["resp_new"].mean() - m["resp_ref"].mean()  # new − ref, P(response=1)
            ax.set_title(f"response mix   ·   ΔP(1)={signed:+.4%}")
            for lv, a_, b_ in zip(levels, pr, pn):
                ax.text(lv - w / 2, a_ + 0.01, f"{a_:.3f}", ha="center", fontsize=6, color=C_REF)
                ax.text(lv + w / 2, b_ + 0.01, f"{b_:.3f}", ha="center", fontsize=6, color=C_NEW)
        ax.spines[["top", "right"]].set_visible(False)

        # Per-row PASS/FAIL tag on the left axis
        axes[r][0].text(-0.18, 0.5, "PASS" if ok else "FAIL", transform=axes[r][0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=10, fontweight="bold",
                        color="#1baf7a" if ok else "#e34948")

    n = results[0]["n"]
    allok = all(m["passed"] for m in results)
    fig.suptitle(
        f"Prior predictive equivalence: serial (turbocharged) vs parallel (reference) kernels"
        f"   ·   N={n:,} trials/model   ·   overall {'PASS' if allok else 'FAIL'}",
        fontsize=10.5, y=0.998,
    )
    # One shared legend line for the two series (identity never color-alone).
    handles = [Line2D([0], [0], color=C_REF, lw=1.8, label="reference (parallel kernel)"),
               Line2D([0], [0], color=C_NEW, lw=1.6, ls="--", label="new (serial kernel)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0.01, 0.02, 1, 0.98])

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {outpath}")


# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["ddm", "rdm", "cdm"], choices=list(MODELS))
    p.add_argument("--n", type=int, default=200_000, help="Trials (prior draws) per model.")
    p.add_argument("--seed", type=int, default=20260710)
    p.add_argument("--outdir", type=str, default="./outputs/figures/simulator_check")
    p.add_argument("--no-plot", action="store_true", help="Text report only, skip the figure.")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Prior predictive equivalence check — N={args.n:,}/model, dt={DT}, "
          f"max_steps={MAX_STEPS}, seed={args.seed}")
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)  # numba kernels draw from the legacy global RNG

    results = []
    for name in args.models:
        m = compute_metrics(name, args.n, rng)
        print_report(m)
        results.append(m)

    allok = all(m["passed"] for m in results)
    print(f"\n{'='*66}\nOVERALL: {'ALL PASS' if allok else 'REVIEW NEEDED'}   "
          f"{ {m['model']: ('pass' if m['passed'] else 'FAIL') for m in results} }")

    if not args.no_plot:
        plot(results, Path(args.outdir) / "prior_predictive_equivalence.pdf")

    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
