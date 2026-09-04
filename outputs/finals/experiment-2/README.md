# Experiment 2 — real data (posterior predictive checks)

Frozen snapshot, organised by model family. Assembled
2026-09-03 16:04:59Z by
`experiments/finalize_experiment_2.py` — 29 files. Git commit `224b31bac1a886a808514f705a23d550a0adfcfe`.
Per-file sources and original mtimes: `MANIFEST.csv`.

## Contents

Each family folder holds, per grouping variable:

| file | what it checks |
|---|---|
| `rt_ppc_<cond>.pdf` | RT distribution, observed vs posterior predictive |
| `choice_ppc_<cond>.pdf` | response dimension: proportions (DDM/RDM) or distribution (CDM) |
| `rt_quantiles_<cond>.pdf` | RT quantiles with predictive intervals |
| `rt_qq_<cond>.pdf` | RT quantile–quantile |
| `choice_quantiles_<cond>.pdf` | **CDM only** — see below |

`posteriors/{ddm,rdm,cdm}.npz` holds the posterior draws every figure was simulated
from, so the plots can be regenerated without re-running inference.

Grouping variables: DDM `stim_type`/`truth`, RDM `congruency`/`session`,
CDM `difficulty`/`speed`.

## Why choice quantiles are CDM-only

Not an omission. CDM's response is a continuous angle, so quantiles of theta are
meaningful the way RT quantiles are. DDM and RDM respond with a binary choice, where
quantiles carry no information — for those the response-dimension check IS the
proportion comparison in `choice_ppc_*`. `plot_choice_quantiles` in
`experiments/real_data/ppc.py` gates on this explicitly and logs a skip.

## Naming

The source calls the RT-distribution plot just `ppc_`, which says nothing about which
dimension it checks when it sits beside `choice_ppc_`. Renamed to `rt_ppc_` here; the
`_mc_` infix is dropped since the whole tree is model-class. `MANIFEST.csv` records
every rename.

## What is included

An allowlist of plot types -- `rt_ppc`, `choice_ppc`, `rt_quantiles`, `rt_qq`,
`choice_quantiles` -- not "whatever was in the source directory", so the tree's
contents do not depend on which flags a given PPC run happened to pass.

## Excluded

Model-FAMILY checkpoint figures and their posteriors — a different lineage from
everything else here (and one that does not exist for RDM at all, since
`RDM_MF_CKPT` was never retrained after the 2026-07-09 factorisation). The mf
posteriors are additionally stale, dated 2026-06-02:

- `outputs/figures/real_data/ppc/ddm/ddm_mf_ppc_stim_type.pdf`
- `outputs/figures/real_data/ppc/ddm/ddm_mf_ppc_truth.pdf`
- `outputs/data/real_data/ddm/mf/ddm_mf_posteriors.npz`
- `outputs/figures/real_data/ppc/cdm/cdm_mf_ppc_difficulty.pdf`
- `outputs/figures/real_data/ppc/cdm/cdm_mf_ppc_speed.pdf`
- `outputs/data/real_data/cdm/mf/cdm_mf_posteriors.npz`

## Caveats

- **RDM predates the Stroop data.** Every RDM figure here was produced before the
  Stroop dataset landed and will need regenerating once that analysis is settled.
- **These figures come from uncommitted code.** As of this snapshot
  `experiments/real_data/ppc.py` carries an uncommitted restyle: per-family
  identity colours from `cogformer.utils.plot_utils.FAMILY_PALETTES` (DDM
  magenta, RDM purple-blue, CDM green), one `PPC_CI` interval level replacing a
  mix of 90% and 95% bands, condition names routed through `CONDITION_DISPLAY`,
  and titles no longer carrying the `(mc)` variant tag. The git commit recorded
  above is therefore the parent of the code that drew these, not that code
  itself. A clean checkout would reproduce the same NUMBERS -- the posterior
  draws in `posteriors/` are unchanged -- but not the same appearance.
- Adjacent analyses NOT included here: `ppc/r4c4_fulldraws/` (contamination
  sensitivity, contam vs nocontam per family, 20 files) and the DDM-only
  `ppc/ddm_combined/` and `ppc/ddm_original/` sets.
