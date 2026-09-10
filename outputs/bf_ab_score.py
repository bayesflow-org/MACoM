"""Score any set of tagged BF checkpoints on ONE shared validation set.

Usage: python outputs/bf_ab_score.py ddm intercept_only tag1 tag2 ...
All arms are scored on identical simulated datasets, so differences are the
approximator's, not the simulator's.
"""
import os, sys
os.environ.setdefault("KERAS_BACKEND", "jax")
import numpy as np, keras, bayesflow as bf
from experiments.model_family.model_family_bf_validate import ModelFamilyBF

MODEL, CASE = sys.argv[1], sys.argv[2]
TAGS = sys.argv[3:]
N, DRAWS = 200, 200

np.random.seed(20260910)
sim = ModelFamilyBF(model_key=MODEL, case=CASE)          # full grid, unsliced
val = sim.sample(N)
act = val["masks"][0].astype(bool)
true = val["params"][:, act]
n_act = int(act.sum())
names = [f"p{i}" for i in range(n_act)]
print(f"{MODEL}/{CASE}: {N} datasets, {n_act} active params\n")

print(f"{'arm':34s} {'calib_err':>10s} {'contract':>10s} {'NRMSE':>8s} {'corr':>7s} {'post_sd':>8s}")
for tag in TAGS:
    path = f"./outputs/checkpoints/model_family/bf/{MODEL}/{MODEL}_families_bf_{CASE}__{tag}/model.keras"
    if not os.path.exists(path):
        print(f"{tag:34s} [missing]"); continue
    ap = keras.saving.load_model(path)
    d = ap.sample(conditions={"rts": val["rts"], "choices": val["choices"],
                              "design_matrices": val["design_matrices"]}, num_samples=DRAWS)
    e = np.asarray(d["params"])
    e = e if e.shape[-1] == n_act else e[:, :, act]
    ce = bf.diagnostics.metrics.calibration_error(estimates=e, targets=true, variable_names=names)["values"]
    pc = bf.diagnostics.metrics.posterior_contraction(estimates=e, targets=true, variable_names=names)["values"]
    rm = bf.diagnostics.metrics.root_mean_squared_error(estimates=e, targets=true, variable_names=names)["values"]
    r = np.mean([np.corrcoef(true[:, j], e[:, :, j].mean(1))[0, 1] for j in range(n_act)])
    print(f"{tag:34s} {ce.mean():10.3f} {pc.mean():10.4f} {rm.mean():8.3f} {r:7.3f} "
          f"{e.std(axis=1).mean():8.4f}")
print("\nTARGET (predecessor, ../CogFormer, ddm/intercept_only): calib 0.011, NRMSE 0.094")
print("Ours in production (2000 ep):                            calib 0.422, NRMSE 0.337")
