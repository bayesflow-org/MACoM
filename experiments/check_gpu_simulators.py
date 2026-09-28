"""Check the Triton GPU simulators against the numba reference kernels.

For each of DDM / RDM / CDM / LBA / aDDM:
  1. capture the per-trial parameters the training samplers actually feed the kernel
     (r4c4 random designs; RDM and LBA with 2-4 accumulators; aDDM as in Exp 4);
  2. speed: numba (1 thread, the sim-worker configuration) vs GPU on the pooled trials;
  3. distribution: at N real parameter sets plus extreme corners, simulate M trials with
     both and test choice counts (chi-square, unfinished trials as their own category)
     and RTs per response (KS; CDM also the response angle). Under equivalence the
     p-values are uniform: ~5% below 0.05, ~1% below 0.01.

Usage:  python -m experiments.check_gpu_simulators [--families DDM,RDM,CDM,LBA,ADDM]
                                                   [--sets 200] [--corners 40] [--trials 20000]
"""
import argparse
import time

import numpy as np
from scipy import stats

import numba
numba.set_num_threads(1)

import cogformer.simulator.benchmarks.ssms.addm as AD
import cogformer.simulator.benchmarks.ssms.cdm as C
import cogformer.simulator.benchmarks.ssms.ddm as D
import cogformer.simulator.benchmarks.ssms.lba as L
import cogformer.simulator.benchmarks.ssms.rdm as R
import cogformer.simulator.benchmarks.ssms.gpu_kernels as G
from cogformer.simulator import ModelFamily
from cogformer.simulator.benchmarks.ssms.link_funs import (addm_link_fun, cdm_link_fun, ddm_link_fun,
                                                           lba_link_fun, rdm_link_fun)
from cogformer.simulator.benchmarks.ssms.priors import (addm_priors, cdm_priors, ddm_priors,
                                                        lba_priors, rdm_priors)
from experiments.experiment_1.model_class_cf_validate import _MODELS_WITH_PI
from experiments.experiment_3.model_class_lba_finetune_validate import _LBA_WITH_PI
from experiments.experiment_4.addm_bf_baseline import INTRINSIC_PARAMS as ADDM_PARAMS


def _rdm_ref(v, a, tau, s_v, s_tau, sigma=1.0, dt=5e-4, max_steps=20000, **_):
    return R._rdm_serial_kernel(v.shape[1])(v, a, tau, s_v, s_tau, sigma, dt, max_steps)


def _drop(fn, *names):
    return lambda **kw: fn(**{k: v for k, v in kw.items() if k not in names})


# name -> (module, captured function name, numba reference, GPU twin, unfinished code,
#          model + family builder)
FAMILIES = {
    "DDM": dict(mod=D, fn="simulate_ddm", ref=D._simulate_ddm_serial, gpu=G.simulate_ddm_gpu,
                unfinished=-1.0),
    "RDM": dict(mod=R, fn="simulate_rdm", ref=_drop(_rdm_ref, "decay"),
                gpu=_drop(G.simulate_rdm_gpu, "decay"), unfinished=-1.0, accumulators=True),
    "CDM": dict(mod=C, fn="simulate_cdm",
                ref=_drop(C._simulate_cdm_serial, "decay"), gpu=_drop(G.simulate_cdm_gpu, "decay"),
                unfinished=-4.0),
    "LBA": dict(mod=L, fn="simulate_lba", ref=L.simulate_lba, gpu=G.simulate_lba_gpu, unfinished=-1.0,
                accumulators=True),
    "ADDM": dict(mod=AD, fn="simulate_addm", ref=AD.simulate_addm, gpu=G.simulate_addm_gpu,
                 unfinished=-1.0),
}
CALL_LEVEL = {"sigma", "dt", "max_steps", "decay", "max_time", "max_iter", "first_fix_mode"}


def families():
    mk = lambda model, prior, params: ModelFamily(model=model, prior=prior, intrinsic_params=params,
                                                  num_workers=1)
    return {
        "DDM": (mk(D.DDM(), ddm_priors(), _MODELS_WITH_PI["ddm"]["intrinsic_params"]), ddm_link_fun(), None),
        "RDM": (mk(R.RDM(), rdm_priors(), _MODELS_WITH_PI["rdm"]["intrinsic_params"]), rdm_link_fun(),
                R.RDM.build_random_context),
        "CDM": (mk(C.CDM(), cdm_priors(), _MODELS_WITH_PI["cdm"]["intrinsic_params"]), cdm_link_fun(), None),
        "LBA": (mk(L.LBA(fixed_A=0.5), lba_priors(), _LBA_WITH_PI["intrinsic_params"]), lba_link_fun(),
                L.LBA.build_random_context),
        # Both first-fixation rules: Exp 4 pins item 0, the standard aDDM draws it.
        "ADDM": (mk(AD.ADDM(), addm_priors(), ADDM_PARAMS),
                 {k: f for k, f in addm_link_fun().items() if k in ADDM_PARAMS},
                 lambda n: {"first_fixation": np.random.choice(["random", "item0", "item1"])}),
    }


def capture(names, num_batches):
    """Per family: the kwargs of every kernel call made while sampling training batches."""
    calls = {n: [] for n in names}
    fams = families()
    for name in names:
        spec = FAMILIES[name]
        orig = getattr(spec["mod"], spec["fn"])

        def rec(*a, _orig=orig, _name=name, **kw):
            calls[_name].append({k: (np.array(v) if isinstance(v, (np.ndarray, list)) else v)
                                 for k, v in kw.items()})
            return _orig(*a, **kw)
        setattr(spec["mod"], spec["fn"], rec)
        try:
            mf, link, ctx = fams[name]
            for _ in range(num_batches):
                mf.batch_sample(batch_size=24, min_num_obs=200, max_num_obs=1000, max_num_regressors=4,
                                max_num_categories=4, add_interaction=True, link_fun=link, context=ctx)
        finally:
            setattr(spec["mod"], spec["fn"], orig)
    return calls


def rows_of(calls):
    rows = []
    for c in calls:
        per_trial = [k for k in c if k not in CALL_LEVEL]
        n = len(c[per_trial[0]])
        for i in range(n):
            r = {k: (np.asarray(c[k][i], dtype=np.float64) if np.ndim(c[k]) == 2 else float(c[k][i]))
                 for k in per_trial}
            r.update({k: c[k] for k in c if k in CALL_LEVEL})
            rows.append(r)
    return rows


def expand(row, M):
    return {k: (v if k in CALL_LEVEL else np.tile(v, (M, 1)) if np.ndim(v) == 1 else np.full(M, v))
            for k, v in row.items()}


def compare(unfinished, circular, r, q):
    ps = []
    fin_r, fin_q = r[:, 0] != unfinished, q[:, 0] != unfinished
    if circular:
        cats_r, cats_q = np.where(fin_r, 0, 1), np.where(fin_q, 0, 1)
    else:
        cats_r, cats_q = r[:, 1].astype(int), q[:, 1].astype(int)
    labels = np.union1d(cats_r, cats_q)
    if len(labels) > 1:
        table = np.array([[np.sum(cats_r == k), np.sum(cats_q == k)] for k in labels])
        ps.append(("choice", stats.chi2_contingency(table)[1]))
    if circular:
        if fin_r.sum() > 50 and fin_q.sum() > 50:
            ps.append(("rt", stats.ks_2samp(r[fin_r, 0], q[fin_q, 0]).pvalue))
            ps.append(("angle", stats.ks_2samp(r[fin_r, 1], q[fin_q, 1]).pvalue))
        return ps
    for k in labels:
        if k < 0:
            continue
        a_, b_ = r[cats_r == k, 0], q[cats_q == k, 0]
        if len(a_) > 50 and len(b_) > 50:
            ps.append((f"rt|{k}", stats.ks_2samp(a_, b_).pvalue))
    return ps


def speed(spec, calls):
    """numba (1 thread) vs GPU seconds over all captured trials."""
    import torch
    groups = {}
    for c in calls:   # the numba kernels take one accumulator count per call
        # v is (n, K) accumulator drifts for RDM/LBA; CDM's (n, 2) is a drift VECTOR
        k = c["v"].shape[1] if spec.get("accumulators") else 0
        groups.setdefault(k, []).append(c)
    t_cpu, n = 0.0, 0
    for cs in groups.values():
        cat = {k: (cs[0][k] if k in CALL_LEVEL else np.concatenate([x[k] for x in cs])) for k in cs[0]}
        spec["ref"](**{k: (v if k in CALL_LEVEL else v[:50]) for k, v in cat.items()})   # JIT
        t0 = time.perf_counter(); spec["ref"](**cat); t_cpu += time.perf_counter() - t0
        n += len(next(v for k, v in cat.items() if k not in CALL_LEVEL))
    # GPU: one launch over everything (pad accumulators like the model wrappers do)
    kmax = max(groups)
    cat = {}
    for key in calls[0]:
        if key in CALL_LEVEL:
            cat[key] = calls[0][key]
        elif key == "v" and kmax:
            cat[key] = np.concatenate([np.pad(c["v"], ((0, 0), (0, kmax - c["v"].shape[1]))) for c in calls])
        else:
            cat[key] = np.concatenate([np.asarray(c[key]) for c in calls])
    if kmax:
        cat["num_alternatives"] = np.concatenate([np.full(len(c["v"]), c["v"].shape[1]) for c in calls])
    if "first_fix_mode" in cat:
        cat["first_fix_mode"] = np.concatenate([np.full(len(c["v_0"]), c["first_fix_mode"]) for c in calls])
    spec["gpu"](**cat); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(5):
        spec["gpu"](**cat)
    torch.cuda.synchronize()
    return t_cpu / n, (time.perf_counter() - t0) / 5 / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="DDM,RDM,CDM,LBA,ADDM")
    ap.add_argument("--sets", type=int, default=200)
    ap.add_argument("--corners", type=int, default=40)
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--batches", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dump", default=None,
                    help="Write every (family, test, p, parameters) row to this .npz for diagnosis.")
    args = ap.parse_args()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    names = args.families.split(",")

    calls = capture(names, args.batches)
    dump = []
    for name in names:
        spec, c = FAMILIES[name], calls[name]
        rows = rows_of(c)
        cpu, gpu, n = speed(spec, c)
        print(f"\n=== {name}: {len(c)} datasets, {n} trials | numba 1 thread {cpu * 1e6:.2f} us/trial, "
              f"GPU {gpu * 1e9:.1f} ns/trial -> {cpu / gpu:.0f}x one core")

        sets = [rows[i] for i in rng.choice(len(rows), min(args.sets, len(rows)), replace=False)]
        scalar_keys = [k for k in rows[0] if k not in CALL_LEVEL and np.ndim(rows[0][k]) == 0]
        lo_hi = {k: np.percentile([r[k] for r in rows], [0.5, 99.5]) for k in scalar_keys}
        vec = [k for k in rows[0] if k not in CALL_LEVEL and np.ndim(rows[0][k]) == 1]
        vlo_hi = {k: np.percentile(np.concatenate([r[k] for r in rows]), [0.5, 99.5]) for k in vec}
        for _ in range(args.corners):
            base = dict(rows[rng.integers(len(rows))])
            for k, (lo, hi) in lo_hi.items():
                if rng.random() < 0.6:
                    base[k] = lo if rng.random() < 0.5 else hi
            for k, (lo, hi) in vlo_hi.items():
                if rng.random() < 0.5:
                    base[k] = rng.choice([lo, hi], size=len(base[k]))
            sets.append(base)

        circular = name == "CDM"
        allp, outliers = [], []
        for row in sets:
            kw = expand(row, args.trials)
            r = spec["ref"](**kw)
            q = spec["gpu"](**kw)
            for test, pv in compare(spec["unfinished"], circular, r, q):
                allp.append(pv)
                dump.append((name, test, pv, repr({k: (np.round(v, 5).tolist() if np.ndim(v) else v)
                                                   for k, v in row.items()})))
                if pv < 1e-3:
                    outliers.append((test, round(float(pv), 6),
                                     {k: (np.round(v, 4).tolist() if np.ndim(v) else round(float(v), 4))
                                      for k, v in row.items() if k not in CALL_LEVEL}))
        allp = np.array(allp)
        print(f"{len(sets)} parameter sets ({min(args.sets, len(rows))} real + {args.corners} corners) x "
              f"{args.trials} trials, {len(allp)} tests: p<0.05 {np.mean(allp < 0.05):.3f} (expect 0.05) | "
              f"p<0.01 {np.mean(allp < 0.01):.3f} (expect 0.01) | min p {allp.min():.2g} "
              f"(expect ~{1 / len(allp):.1g}) | uniformity KS p={stats.kstest(allp, 'uniform').pvalue:.3f}")
        for o in outliers:
            print("   p<1e-3:", o)
    if args.dump:
        np.savez(args.dump, family=[d[0] for d in dump], test=[d[1] for d in dump],
                 p=np.array([d[2] for d in dump]), params=[d[3] for d in dump])


if __name__ == "__main__":
    main()
