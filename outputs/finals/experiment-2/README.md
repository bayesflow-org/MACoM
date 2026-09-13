# Experiment 2 — real data (posterior predictive checks)

Frozen snapshot, organised by model family. Assembled
2026-09-13 04:13:38Z by
`experiments/finalize_experiment_2.py` — 71 files. Git commit `1705ab025dd1553a92ea2d6c79101b67d58de0e6`.
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
| `combined_ppc.pdf` | every grouping variable in ONE figure — see below |

`posteriors/{ddm,rdm,cdm}.npz` holds the posterior draws every figure was simulated
from, so the plots can be regenerated without re-running inference.
`posteriors/ddm_sv_fixed.npz` is the separate fit behind `ddm/sv_fixed/`; the
simulation-only variants have no npz of their own because they reuse their family's.

### combined_ppc.pdf

The family's whole check on one page, for the per-condition figures above are the same
data split across files. Condition groups are separated by a dashed vertical divider.

- **DDM / RDM** — one row: an RT-quantile panel per level of each grouping variable,
  then a solid divider and the choice-proportion check as a 2x2 of per-level panels
  folded into the footprint ONE quantile panel would take. The solid rule marks a
  change of MEASURE, not of grouping. Inside the 2x2 the grouping variables are the
  rows, separated by a dashed rule; the four share a y scale, and only the top-left
  panel carries a legend.
- **CDM** — two rows of five: RT quantiles on top, response-angle quantiles beneath,
  difficulty then speed across.

No titles: each panel is labelled only by its condition level, and one legend on the
first panel covers every series in the figure.

Grouping variables: DDM `stim_type`/`truth`, RDM `congruency`/`session`,
CDM `difficulty`/`speed`.

## Constrained variants

A family folder's ROOT is always the model's own fit, simulated as-is. Subfolders hold
PPCs run from the **same posterior draws** with something pinned during the forward
simulation. They are not separate fits, which is why there is deliberately no second
`posteriors/*.npz` for them.

| folder | pinned | produced by |
|---|---|---|
| folder | `s_v` | kind |
|---|---|---|
| `ddm/sv0/` | 0 at simulation | constrained **PPC** (`--fix_natural s_v=0.0`) |
| `ddm/sv_fixed/` | 0 at fitting | constrained **REFIT** (`--fix_s_v`) |
| `cdm/sv0/` | 0 at simulation | constrained PPC (`--fix_natural s_v=0.0`) |
| `rdm/nocontam/` | `pi`=0 at simulation | constrained PPC (`--no_contamination`) |

### A constrained PPC is not a constrained fit

`ddm/sv0/` and `ddm/sv_fixed/` both set `s_v` to zero, and they are NOT the same thing.

- **`sv0/`** overrides at SIMULATION time. The posterior draws still come from a fit
  where `s_v` was free, so `v`, `a` and `tau` carry the compensation they acquired
  while it was estimated. This answers "what does this fit predict with drift
  variability switched off".
- **`sv_fixed/`** is a genuine REFIT. `s_v` is dropped from the design, so it is never
  an estimation target and the simulator injects 0.0 directly, bypassing the link.
  Every other parameter is re-estimated under that constraint. This is the one that
  matches the original specification (Stump et al., 2026).

`s_v` is a substantial component -- natural-scale median 0.456 against a median |v| of
1.243, a ratio of 0.37 -- so neither is a formality. Measured over RT quantiles
0.1/0.3/0.5/0.7/0.9, both conditions, 72 subjects:

| run | mean abs quantile error | predictive SD (truth=True) |
|---|---|---|
| observed | — | 1.173 |
| canonical | 0.0869 | 1.503 |
| `sv0/` (simulation-only) | 0.0695 | 1.581 |
| `sv_fixed/` (refit) | 0.0784 | 1.533 |

Read the aggregate with care: it is sensitive to which quantiles you score. On a coarse
0.1/0.5/0.9 grid the canonical run wins; adding 0.3 and 0.7 reverses it, because the
canonical fit's largest miss sits at q0.7 (-0.146). What is STABLE across grids is the
shape, not the ranking:

- `sv0/` fits the BODY better but overshoots the right tail badly (q0.9: +0.112 and
  +0.161 against observed), and is the most over-dispersed of the three. Switching off
  drift variability without re-estimating leaves `v`, `a` and `tau` compensating for a
  component that is no longer there.
- `sv_fixed/` tracks the canonical fit closely everywhere and slightly improves the
  right tail (q0.9 +0.019 vs -0.049 on truth=True). Re-estimating under the constraint
  re-absorbs what `s_v` was doing, so the predictive barely moves. **That similarity is
  the result**: for this dataset, dropping across-trial drift variability properly costs
  almost nothing.

All three under-predict RT through the body and are over-dispersed relative to observed
(SD ~1.5 against 1.17), which is a pre-existing property of the lapse model, not
something these variants introduce.

`cdm/sv0/` asks the simulation-time question of CDM;

Note that `sv_fixed/`'s PPC must still pass `--fix_natural s_v=0.0`: a parameter absent
from the design has its coefficients masked to zero, and the link is still applied --
`shifted_softplus(0) = 0.693`, not 0, because that function never implemented the
-log(2) shift its name implies.

`cdm/sv0/` asks the simulation-time question of CDM;
`rdm/nocontam/` isolates the distributional fit from lapse contamination. Which one
belongs in the paper is a modelling decision and has NOT been made here -- both are
kept labelled so the constraint stays recoverable from the tree, the same rule
`experiment-3/ppc/` follows.

`rdm/nocontam/` is on the CORRECTED lineage: it was produced after the 2026-09-11
`favored_idx` refit, not before it.

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
