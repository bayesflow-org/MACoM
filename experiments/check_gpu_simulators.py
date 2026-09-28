"""Check the Triton GPU simulators against the numba reference kernels.

For each of DDM / RDM / CDM:
  1. capture the per-trial parameters the training sampler actually feeds the kernel
     (r4c4 random designs, RDM with 2-4 accumulators);
  2. speed: numba serial kernel (1 thread, the sim-worker configuration) vs GPU on the
     pooled trials;
  3. distribution: at N real parameter sets plus extreme corners, simulate M trials with
     both and test choice counts (chi-square, unfinished trials as their own category)
     and RTs per response (KS; CDM also the response angle). Under equivalence the
     p-values are uniform: ~5% below 0.05, ~1% below 0.01.

Usage:  python -m experiments.check_gpu_simulators [--sets 200] [--corners 40] [--trials 20000]
"""
import argparse
import time

import numpy as np
from scipy import stats

import numba
numba.set_num_threads(1)

import cogformer.simulator.benchmarks.ssms.cdm as C
import cogformer.simulator.benchmarks.ssms.ddm as D
import cogformer.simulator.benchmarks.ssms.rdm as R
from cogformer.simulator import ModelClass, ModelFamily
from cogformer.simulator.benchmarks.ssms.gpu_kernels import (
    simulate_cdm_gpu, simulate_ddm_gpu, simulate_rdm_gpu)
from experiments.experiment_1.model_class_cf_validate import _MODELS_WITH_PI
from cogformer.simulator.benchmarks.ssms.priors import cdm_priors, ddm_priors, rdm_priors
from cogformer.simulator.benchmarks.ssms.link_funs import cdm_link_fun, ddm_link_fun, rdm_link_fun

SCALARS = ("a", "tau", "s_v", "s_tau")


def capture(num_batches):
    calls = {"DDM": [], "RDM": [], "CDM": []}
    orig = {"DDM": D.simulate_ddm, "RDM": R.simulate_rdm, "CDM": C.simulate_cdm}

    def recorder(name):
        def rec(**kw):
            calls[name].append({k: np.array(v) for k, v in kw.items()})
            return orig[name](**kw)
        return rec
    D.simulate_ddm, R.simulate_rdm, C.simulate_cdm = recorder("DDM"), recorder("RDM"), recorder("CDM")
    fams = {
        "DDM": ModelFamily(model=D.DDM(), name="DDM", prior=ddm_priors(), num_workers=1,
                           intrinsic_params=_MODELS_WITH_PI["ddm"]["intrinsic_params"]),
        "RDM": ModelFamily(model=R.RDM(), name="RDM", prior=rdm_priors(), num_workers=1,
                           intrinsic_params=_MODELS_WITH_PI["rdm"]["intrinsic_params"]),
        "CDM": ModelFamily(model=C.CDM(), name="CDM", prior=cdm_priors(), num_workers=1,
                           intrinsic_params=_MODELS_WITH_PI["cdm"]["intrinsic_params"]),
    }
    mc = ModelClass(model_families=fams, num_workers=1,
                    link_funs={"DDM": ddm_link_fun(), "RDM": rdm_link_fun(), "CDM": cdm_link_fun()},
                    train_contexts={"RDM": R.RDM.build_random_context})
    for _ in range(num_batches):
        mc.batch_sample(batch_size=64, min_num_obs=200, max_num_obs=1000, max_num_regressors=4,
                        max_num_categories=4, add_interaction=True)
    D.simulate_ddm, R.simulate_rdm, C.simulate_cdm = orig["DDM"], orig["RDM"], orig["CDM"]
    return calls


def rows_of(fam, calls):
    """One dict per captured trial (a parameter set)."""
    out = []
    for c in calls:
        n = len(c["a"])
        for i in range(n):
            r = {k: float(c[k][i]) for k in SCALARS}
            if fam == "DDM":
                r.update(v=float(c["v"][i]), z=float(c["z"][i]))
            else:
                r["v"] = np.asarray(c["v"][i], dtype=np.float64)
            out.append(r)
    return out


def run(fam, p, M, dt, max_steps, gpu):
    full = {k: np.full(M, p[k]) for k in SCALARS}
    if fam == "DDM":
        v, z = np.full(M, p["v"]), np.full(M, p["z"])
        if gpu:
            return simulate_ddm_gpu(v, full["a"], z, full["tau"], full["s_v"], full["s_tau"], dt=dt, max_steps=max_steps)
        return D._simulate_ddm_serial(v, full["a"], z, full["tau"], full["s_v"], full["s_tau"], 1.0, dt, max_steps)
    v = np.tile(p["v"], (M, 1))
    if fam == "RDM":
        if gpu:
            return simulate_rdm_gpu(v, full["a"], full["tau"], full["s_v"], full["s_tau"], dt=dt, max_steps=max_steps)
        return R._rdm_serial_kernel(v.shape[1])(v, full["a"], full["tau"], full["s_v"], full["s_tau"], 1.0, dt, max_steps)
    if gpu:
        return simulate_cdm_gpu(v, full["a"], full["tau"], full["s_v"], full["s_tau"], dt=dt, max_steps=max_steps)
    return C._simulate_cdm_serial(v, full["a"], full["tau"], full["s_v"], full["s_tau"], 1.0, dt, max_steps)


def compare(fam, r, q):
    """p-values: choice-count chi-square (DDM/RDM), KS on RT per response, CDM angle KS."""
    ps = []
    unfinished = -4.0 if fam == "CDM" else -1.0
    fin_r, fin_q = r[:, 0] != unfinished, q[:, 0] != unfinished
    if fam == "CDM":
        cats_r = np.where(fin_r, 0, 1); cats_q = np.where(fin_q, 0, 1)
    else:
        cats_r, cats_q = r[:, 1].astype(int), q[:, 1].astype(int)
    labels = np.union1d(cats_r, cats_q)
    table = np.array([[np.sum(cats_r == k), np.sum(cats_q == k)] for k in labels])
    if len(labels) > 1:
        ps.append(("choice", stats.chi2_contingency(table)[1]))
    if fam == "CDM":
        if fin_r.sum() > 50 and fin_q.sum() > 50:
            ps.append(("rt", stats.ks_2samp(r[fin_r, 0], q[fin_q, 0]).pvalue))
            ps.append(("angle", stats.ks_2samp(r[fin_r, 1], q[fin_q, 1]).pvalue))
    else:
        for k in labels:
            if k < 0:
                continue
            a_, b_ = r[cats_r == k, 0], q[cats_q == k, 0]
            if len(a_) > 50 and len(b_) > 50:
                ps.append((f"rt|{k}", stats.ks_2samp(a_, b_).pvalue))
    return ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", type=int, default=200)
    ap.add_argument("--corners", type=int, default=40)
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--families", default="DDM,RDM,CDM")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    calls = capture(args.batches)
    dt, max_steps = 5e-4, 20000
    verdict = {}
    for fam in args.families.split(","):
        c = calls[fam]
        dt, max_steps = float(c[0]["dt"]), int(c[0]["max_steps"])
        rows = rows_of(fam, c)
        print(f"\n=== {fam}: {len(c)} datasets, {len(rows)} trials captured (dt={dt}, max_steps={max_steps})")

        # ── speed on the pooled real trials ─────────────────────────────────────────
        if fam == "RDM":
            groups = {}
            for cc in c:
                groups.setdefault(cc["v"].shape[1], []).append(cc)
            t_cpu = 0.0
            for m, cs in groups.items():
                cat = {k: np.concatenate([x[k] for x in cs]) for k in ("v",) + SCALARS}
                ker = R._rdm_serial_kernel(m)
                ker(*(cat[k][:50] for k in ("v",) + SCALARS), 1.0, dt, max_steps)
                t0 = time.perf_counter(); ker(*(cat[k] for k in ("v",) + SCALARS), 1.0, dt, max_steps)
                t_cpu += time.perf_counter() - t0
            maxk = max(groups)
            vpad = np.concatenate([np.pad(x["v"], ((0, 0), (0, maxk - x["v"].shape[1]))) for x in c])
            mm = np.concatenate([np.full(len(x["a"]), x["v"].shape[1]) for x in c])
            sc = {k: np.concatenate([x[k] for x in c]) for k in SCALARS}
            gpu_call = lambda: simulate_rdm_gpu(vpad, sc["a"], sc["tau"], sc["s_v"], sc["s_tau"],
                                                dt=dt, max_steps=max_steps, num_alternatives=mm)
        else:
            keys = ("v", "a", "z", "tau", "s_v", "s_tau") if fam == "DDM" else ("v",) + SCALARS
            cat = {k: np.concatenate([x[k] for x in c]) for k in keys}
            ref = D._simulate_ddm_serial if fam == "DDM" else C._simulate_cdm_serial
            ref(*(cat[k][:50] for k in keys), 1.0, dt, max_steps)
            t0 = time.perf_counter(); ref(*(cat[k] for k in keys), 1.0, dt, max_steps)
            t_cpu = time.perf_counter() - t0
            g = simulate_ddm_gpu if fam == "DDM" else simulate_cdm_gpu
            gpu_call = lambda: g(*(cat[k] for k in keys), dt=dt, max_steps=max_steps)
        gpu_call(); gpu_call()
        import torch; torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            gpu_call()
        t_gpu = (time.perf_counter() - t0) / 5
        n = len(rows)
        print(f"speed: numba 1 thread {t_cpu / n * 1e6:.2f} us/trial | GPU {t_gpu / n * 1e9:.1f} ns/trial "
              f"-> {t_cpu / t_gpu:.0f}x one core")

        # ── distribution tests ──────────────────────────────────────────────────────
        sets = [rows[i] for i in rng.choice(n, min(args.sets, n), replace=False)]
        lo_hi = {k: (np.percentile([r[k] for r in rows], 0.5), np.percentile([r[k] for r in rows], 99.5))
                 for k in SCALARS + (("v", "z") if fam == "DDM" else ())}
        for _ in range(args.corners):
            base = dict(rows[rng.integers(n)])
            for k, (lo, hi) in lo_hi.items():
                if rng.random() < 0.6:
                    base[k] = lo if rng.random() < 0.5 else hi
            if fam != "DDM" and rng.random() < 0.5:
                vlo, vhi = np.percentile(np.concatenate([r["v"] for r in rows]), [0.5, 99.5])
                base["v"] = rng.choice([vlo, vhi], size=len(base["v"]))
            sets.append(base)

        allp, outliers = [], []
        for p in sets:
            r = run(fam, p, args.trials, dt, max_steps, gpu=False)
            q = run(fam, p, args.trials, dt, max_steps, gpu=True)
            for name, pv in compare(fam, r, q):
                allp.append(pv)
                if pv < 1e-3:
                    outliers.append((name, pv, {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else round(v, 4))
                                                for k, v in p.items()}))
        allp = np.array(allp)
        print(f"{len(sets)} parameter sets ({args.sets} real + {args.corners} corners) x {args.trials} trials, "
              f"{len(allp)} tests: p<0.05 {np.mean(allp < 0.05):.3f} (expect 0.05) | p<0.01 "
              f"{np.mean(allp < 0.01):.3f} (expect 0.01) | min p {allp.min():.2g} "
              f"(expect ~{1 / len(allp):.1g}) | uniformity KS p={stats.kstest(allp, 'uniform').pvalue:.3f}")
        for o in outliers:
            print("   p<1e-3:", o)
        verdict[fam] = (np.mean(allp < 0.05), np.mean(allp < 0.01))
    return verdict


if __name__ == "__main__":
    main()
