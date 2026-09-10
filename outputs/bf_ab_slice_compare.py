"""Score the slicing A/B on ONE shared validation set (ddm/regressed)."""
import os, sys
os.environ.setdefault("KERAS_BACKEND", "jax")
import numpy as np, keras, bayesflow as bf
from experiments.model_family.model_family_bf_validate import ModelFamilyBF

MODEL, CASE, N, DRAWS = "ddm", "regressed", 200, 200
ARMS = {
    "fullgrid (28 dims, 15 dead)": f"./outputs/checkpoints/model_family/bf/{MODEL}/{MODEL}_families_bf_{CASE}__ab_fullgrid/model.keras",
    "sliced   (13 active dims)":   f"./outputs/checkpoints/model_family/bf/{MODEL}/{MODEL}_families_bf_{CASE}__ab_sliced/model.keras",
}

np.random.seed(20260910)
sim = ModelFamilyBF(model_key=MODEL, case=CASE)   # NOT sliced: full grid truth
val = sim.sample(N)
act = val["masks"][0].astype(bool)
true = val["params"][:, act]
n_grid, n_act = int(act.size), int(act.sum())
names = [f"p{i}" for i in range(n_act)]
print(f"shared validation set: {N} datasets, grid={n_grid}, active={n_act}\n")

rows = {}
for label, path in ARMS.items():
    if not os.path.exists(path):
        print(f"[missing] {label}"); continue
    ap = keras.saving.load_model(path)
    d = ap.sample(conditions={"rts": val["rts"], "choices": val["choices"],
                              "design_matrices": val["design_matrices"]}, num_samples=DRAWS)
    e = np.asarray(d["params"])
    e = e if e.shape[-1] == n_act else e[:, :, act]     # width tells the era apart
    ce = bf.diagnostics.metrics.calibration_error(estimates=e, targets=true, variable_names=names)["values"]
    pc = bf.diagnostics.metrics.posterior_contraction(estimates=e, targets=true, variable_names=names)["values"]
    rm = bf.diagnostics.metrics.root_mean_squared_error(estimates=e, targets=true, variable_names=names)["values"]
    r  = np.array([np.corrcoef(true[:, j], e[:, :, j].mean(1))[0, 1] for j in range(n_act)])
    rows[label] = dict(ce=ce, pc=pc, rm=rm, r=r, sd=e.std(axis=1).mean(axis=0))

print(f"{'arm':30s} {'calib_err':>10s} {'contraction':>12s} {'NRMSE':>8s} {'corr':>7s} {'post_sd':>8s}")
for k, v in rows.items():
    print(f"{k:30s} {v['ce'].mean():10.3f} {v['pc'].mean():12.4f} {v['rm'].mean():8.3f} "
          f"{v['r'].mean():7.3f} {v['sd'].mean():8.4f}")
if len(rows) == 2:
    a, b = list(rows.values())
    print(f"\ncalibration error: {a['ce'].mean():.3f} -> {b['ce'].mean():.3f} "
          f"({100*(b['ce'].mean()-a['ce'].mean())/a['ce'].mean():+.0f}%)")
print("\nReference: production 2000-epoch ddm/regressed = 0.405 calib, 0.9905 contraction.")
print("CogFormer on the same simulations reaches 0.02-0.04.")
