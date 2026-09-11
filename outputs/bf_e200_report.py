"""Report BF diagnostics for the __e200 runs straight from the saved npz.

model_family_bf_validate already writes true_set / pred_set / param_masks, so every
number here is CPU-only -- no checkpoint load, no GPU contention with a run still
training on the other card.

Adds the two columns that actually distinguish "honestly wide" from "collapsed",
which the bf.diagnostics table does not carry:
    sd_ratio = mean posterior sd / prior(target) sd
    corr     = corr(posterior mean, truth) across datasets
A collapsed posterior has sd_ratio -> 0 with corr still highish; an honest one keeps
sd_ratio large exactly where the parameter is weakly identified (watch s_v).

Usage: .venv/bin/python outputs/bf_e200_report.py [tag] [model/case ...]
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.configs import MODELS as _BASE          # noqa: E402
from experiments.utils import add_global_params          # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "e200"
TARGETS = sys.argv[2:] or ["ddm/intercept_only", "ddm/regressed",
                           "cdm/intercept_only", "cdm/regressed"]
DATA = Path("./outputs/data/model_family")


def names_for(model, case):
    cfg = add_global_params(_BASE[model], ["pi"], [r"$\pi$"])
    return cfg["bf_case_configs"][case]["param_names"]


def report(model, case):
    stem = f"{case}__{TAG}" if TAG else case
    npz = DATA / f"{model}_{stem}_data.npz"
    if not npz.exists():
        print(f"  [{model}/{case}] not ready ({npz.name} missing)")
        return None

    d = np.load(npz)
    true_full, pred_full, masks = d["true_set"], d["pred_set"], d["param_masks"]
    act = masks[0].astype(bool)
    true = true_full[:, act]                 # (N, P)
    pred = pred_full[:, :, act]              # (N, draws, P)
    names = names_for(model, case)

    post_sd = pred.std(axis=1).mean(0)       # mean posterior sd per param
    true_sd = true.std(0)
    pm = pred.mean(1)
    corr = np.array([np.corrcoef(true[:, j], pm[:, j])[0, 1] for j in range(true.shape[1])])
    # contraction as bayesflow defines it: 1 - E[var_post] / var_prior
    contr = 1.0 - (pred.var(axis=1).mean(0) / true.var(0))
    nrmse = np.sqrt(((pm - true) ** 2).mean(0)) / (true.max(0) - true.min(0))

    print(f"\n  {model}/{case}   N={true.shape[0]} datasets, {true.shape[1]} active params")
    print(f"  {'param':16s} {'NRMSE':>7s} {'contr':>7s} {'sd_ratio':>9s} {'corr':>7s} "
          f"{'post_sd':>9s} {'prior_sd':>9s}")
    for j, nm in enumerate(names):
        print(f"  {nm:16s} {nrmse[j]:7.3f} {contr[j]:7.3f} {post_sd[j]/true_sd[j]:9.3f} "
              f"{corr[j]:7.3f} {post_sd[j]:9.4f} {true_sd[j]:9.4f}")
    print(f"  {'MEAN':16s} {nrmse.mean():7.3f} {contr.mean():7.3f} "
          f"{(post_sd/true_sd).mean():9.3f} {corr.mean():7.3f}")
    return dict(model=model, case=case, nrmse=nrmse.mean(), contr=contr.mean(),
                sd_ratio=(post_sd / true_sd).mean(), corr=corr.mean())


print(f"=== BF tag={TAG!r}: posterior width vs identifiability ===")
rows = [r for t in TARGETS for r in [report(*t.split("/"))] if r]
if rows:
    print(f"\n=== summary ===")
    print(f"{'run':22s} {'NRMSE':>7s} {'contr':>7s} {'sd_ratio':>9s} {'corr':>7s}")
    for r in rows:
        print(f"{r['model']+'/'+r['case']:22s} {r['nrmse']:7.3f} {r['contr']:7.3f} "
              f"{r['sd_ratio']:9.3f} {r['corr']:7.3f}")
