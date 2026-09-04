# Experiment 4 — encoder representation

Frozen snapshot. Assembled 2026-09-04 19:19:45Z
by `experiments/finalize_experiment_4.py` — 94 files. Commit `224b31bac1a886a808514f705a23d550a0adfcfe`.

## Layout

| folder | level | what |
|---|---|---|
| `probe/<variant>/frozen` | posterior | frozen CogFormer encoder + FlowMatching |
| `probe/<variant>/baseline` | posterior | learned SetTransformer + FlowMatching |
| `probe/<variant>/compare` | posterior | PAIRED shared-validation C2ST / log-density / CRPS |
| `representation/<case>` | representation | RSA (crossnobis) + CKA |

Variants: `intercept_only_7d` (original 7-cell target), `vfixed` (6 cells, v pinned),
`varying_values` (9 cells, trial-varying item values).

## Headline results

**The arms differ at BOTH levels.** Posteriors: C2ST excess over an empirical null of
+0.128 / +0.118 / +0.205 across the three variants (z = +25 to +38), learned summary
better on log-density and CRPS. Geometry: RSA 0.80-0.81 against a
dimensionality-matched null of 0.85-0.87 — at or BELOW the null. Not "same
representation, different decoder".

**Model families occupy orthogonal subspaces.** Between-family CKA is -0.011..0.000 for
every pair, including DDM-RDM which were trained together, against split-half noise
ceilings of 0.96-0.98. This argues against a single universal code.

**LBA finetuning did not change the representation.** original vs finetuned encoder on
SHARED datasets scores 0.98-0.996 on lba/ddm/rdm/cdm, at or above every noise ceiling —
including LBA itself. The -48.2% LBA error improvement came from the DECODER; the
zero-shot encoder already carried the information (weight drift: encoder 3.3%,
decoder 6.8%). This answers experiment 3's forgetting question at a level aggregate
NRMSE cannot reach.

**Held-out aDDM transfers fine.** encoder-vs-shared-parameter RSA: ddm 0.217, rdm 0.478,
cdm 0.377, aDDM 0.223 — above trained DDM, and the spread across trained families
dwarfs the trained/held-out gap.

## Reading the numbers

- **Always against the null.** A random 128->16 projection scores RSA ~0.87, so an
  unqualified "RSA 0.80" reads as strong evidence and is evidence of nothing.
- **Always with the ceiling.** Split-half scores (same conditions, re-simulated) run
  0.96-0.98. Without them a near-zero off-diagonal cannot be distinguished from a
  metric that simply fails across independent stimulus sets.
- **Joint regression R^2 is only 0.08-0.35**, so most of what the encoder represents is
  NOT parameter structure.
- Marginal per-parameter RSA is DILUTED (`a`: contraction 0.983, marginal RSA 0.166);
  the joint-regression betas are the unique contributions. Both are plotted.

## Excluded

30 files: five superseded exploratory baseline runs under
`addm_baseline/intercept_only/` (item0_bd_7d, item0_best_defaults, item0_best_defaults_a0, item0_main_7d, item0_main_do0) from the 469-dim and
BayesFlow-defaults debugging — only `item0_bd_7d_e500` is the final 7-dim arm — plus
non-artifact files.
