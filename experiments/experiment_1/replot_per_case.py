"""Redraw the per-case validation figures from the saved predictions — no GPU.

``model_class_cf_validate.py`` spends roughly 20 minutes per case on ODE
integration, 18 cases in a run. Restyling its figures must not cost that again,
and the cost is not even the worst of it: sampling again draws a DIFFERENT
posterior, so a rerun to change how a figure looks would quietly change what it
shows.

Everything the figures draw is in ``<pred_dir>/<family>_<case>_cf_pred.npz``,
which phase 2 writes before any plotting happens. The plotting itself is
``model_class_cf_validate.plot_case``, imported rather than copied, so a replot
and a real run cannot drift apart.

    python -m experiments.experiment_1.replot_per_case \\
        --pred_dir outputs/data/model_class/r4c4_b256_e10000_validate \\
        --outdir   outputs/figures/model_class/cf/r4c4_b256_e10000_validate

Posterior pairplots are never redrawn: they are per-dataset, the finalized tree
excludes them by allowlist, and there are 10 per case.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

from cogformer.simulator import ModelFamily, ModelClass
from cogformer.simulator.benchmarks.ssms.ddm import DDM
from cogformer.simulator.benchmarks.ssms.rdm import RDM
from cogformer.simulator.benchmarks.ssms.cdm import CDM
from cogformer.simulator.benchmarks.ssms.priors import ddm_priors, rdm_priors, cdm_priors
from cogformer.simulator.benchmarks.ssms.link_funs import (ddm_link_fun, rdm_link_fun,
                                                           cdm_link_fun)
from cogformer.utils.plot_utils import cogformer_mc_colors

from .model_class_cf_validate import (get_benchmark_design_configs, plot_case,
                                      _MODELS_WITH_PI)
from ..utils import resolve_val_scenario


def build_model_class():
    """Parameter space only — no checkpoint, no simulation workers."""
    families = {
        "DDM": ModelFamily(model=DDM(), name="DDM", prior=ddm_priors(),
                           intrinsic_params=_MODELS_WITH_PI["ddm"]["intrinsic_params"]),
        "RDM": ModelFamily(model=RDM(), name="RDM", prior=rdm_priors(),
                           intrinsic_params=_MODELS_WITH_PI["rdm"]["intrinsic_params"]),
        "CDM": ModelFamily(model=CDM(), name="CDM", prior=cdm_priors(),
                           intrinsic_params=_MODELS_WITH_PI["cdm"]["intrinsic_params"]),
    }
    link_funs = {"DDM": ddm_link_fun(), "RDM": rdm_link_fun(), "CDM": cdm_link_fun()}
    return ModelClass(model_families=families, link_funs=link_funs)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--pred_dir", type=str, required=True,
                   help="Directory of <family>_<case>_cf_pred.npz. Point this at the "
                        "VALIDATE RUN's own directory, not the mixed-vintage default "
                        "tree, or the figures come from stale predictions silently.")
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--max_num_regressors", type=int, default=4)
    p.add_argument("--max_num_categories", type=int, default=4)
    # Mirrors the validation script: the metric is slow and off by default there,
    # so it has to be off by default here too or the replot writes an extra column.
    p.add_argument("--skip_log_gamma", action="store_true", default=True)
    p.add_argument("--include_log_gamma", dest="skip_log_gamma",
                   action="store_false")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated <family>_<case> tags; default is all present.")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    a = parse_args()
    pred_dir, outroot = Path(a.pred_dir), Path(a.outdir)
    colors = cogformer_mc_colors()
    model_class = build_model_class()
    only = {t.strip() for t in a.only.split(",")} if a.only else None

    drawn, missing = 0, []
    for model_name, model_cfg in get_benchmark_design_configs().items():
        global_indices = model_class.local_to_global[model_name]
        for cfg_name, design_config, scenario_opts in model_cfg["val_scenarios"]:
            # The same two skips the validation run applies. Without them this would
            # look for npz files that a run at this geometry never produced.
            _, _, mix_opts, skip = resolve_val_scenario(scenario_opts,
                                                        a.max_num_categories)
            n_main = sum(1 for k in design_config if k != "1" and ":" not in k)
            if skip or n_main > a.max_num_regressors:
                continue
            tag = f"{model_name.lower()}_{cfg_name}"
            if only is not None and tag not in only:
                continue
            npz = pred_dir / f"{tag}_cf_pred.npz"
            if not npz.exists():
                missing.append(tag)
                continue

            d = np.load(npz)
            outdir = outroot / model_name.lower() / cfg_name
            outdir.mkdir(parents=True, exist_ok=True)
            case = {
                "model_name": model_name, "cfg_name": cfg_name,
                "design_config": design_config,
                "intrinsic_params": model_cfg["intrinsic_params"],
                "variable_names": model_cfg["variable_names"],
                "global_indices": global_indices,
                "true_set": d["true_set"], "pred_set": d["pred_set"],
                "params_mask": d["params_mask"],
                "n_rows": d["true_set"].shape[1], "outdir": outdir,
                "categorical_regressors": mix_opts.get("categorical_regressors"),
            }
            # The metadata is rebuilt from config while the arrays come off disk, so
            # the two could in principle describe different runs. A row/column
            # mismatch is the symptom, and it is worth failing on rather than
            # drawing a grid whose labels belong to something else.
            n_par = len(case["intrinsic_params"])
            if case["params_mask"].shape != (case["n_rows"], n_par):
                raise SystemExit(
                    f"{tag}: saved mask is {case['params_mask'].shape} but the config "
                    f"implies {(case['n_rows'], n_par)}. The npz and this geometry "
                    "disagree -- check --max_num_regressors/--max_num_categories.")

            logging.info(f"drawing {tag} -> {outdir}")
            plot_case(case, colors, a.max_num_categories, a.max_num_regressors,
                      skip_posteriors=True, skip_log_gamma=a.skip_log_gamma)
            drawn += 1

    logging.info(f"redrew {drawn} cases")
    if missing:
        logging.warning(f"no predictions for {len(missing)} case(s): {missing}")


if __name__ == "__main__":
    main()
