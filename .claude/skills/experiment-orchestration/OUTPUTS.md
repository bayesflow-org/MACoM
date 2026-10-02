# Pipeline outputs: stage → script → artifact → working location → finals

Reference for `SKILL.md`. Verified against the code on 2026-09-28. The authoritative
source→finals mapping is `experiments/finalize_experiment_N.py`; if this file and that
code disagree, the code wins and this file needs updating.

## Notation

| Symbol | Meaning (current value) |
|---|---|
| `TAG` | `$MC_TAG` = `r4c4_b128_e15000` |
| `FIG` | `outputs/figures/model_class/cf` |
| `DATA` | `outputs/data/model_class` |
| `LBA` | `FIG/TAG_lba` (the Exp 3 arm tree) |
| `BENCH` | `outputs/data/benchmark/seed0` |
| `MC_CKPT` | `outputs/checkpoints/model_class/cf/TAG/cogformer_model_class_fm_l8_h8_p256_s32_d128_b128_e15000_t100.pt` |
| `LBA_FT_CKPT` | `outputs/checkpoints/model_class/cf/lba_finetune/cogformer_lba_finetune_s2_A0.5_l8_h8_p256_s32_d128_b128_e1000_t100.pt` |
| `<fam>` / `<case>` | `ddm rdm cdm` (+`lba`) / bench5: `intercept_only fixed regressed fixed_regressed interaction` (+`mixed_predictor` in per-case validation) |
| `<plots>` | `recovery coverage ecdf metrics` as PDF, each with a `_global` twin (the π panel) + `metrics.csv`, `metrics_global.csv` |

**Finals renaming rules** (all assemblers): the run prefix (`<fam>_<case>_fm_`, `_ft_`, `_mid1_`,
`lba_mc_`, …) is stripped because the path already carries it; `_global` becomes `_pi`;
the bare RT plot `ppc_*` becomes `rt_ppc_*`.

## Dependency graph

```
e1_data ──► BENCH ──┬──────────────► e1_validate ──► DATA/TAG_validate ─┬─► e1_benchmark
                    │                     ▲                            ├─► e1a_c2st ◄── e1a_bf
                    ├─► e1a_bf ───────────┼──► outputs/data/model_family ┘       (BF npz)
                    ├─► e3_bf ────────────┼──► outputs/data/model_family ──► e3_gap_c2st
                    ├─► e3_zeroshot ◄─────┤ MC_CKPT
                    └─► e3_ft_validate ◄──┼── LBA_FT_CKPT ◄── e3_finetune ◄── MC_CKPT
e1_train ──► MC_CKPT ─┬─► e1_validate, e1_ensemble, e1_design_eig, e2_*, e3_zeroshot,
                      │   e3_finetune, e4_probe, e4_compare, e4_representation
e3_original = links into e1_validate ─► e3_paired_c2st ◄── e3_ft_validate
e3_zeroshot + e3_ft_validate + e1_validate ─► e3_compare
LBA_FT_CKPT ─► e3_ppc_apply ─► e3_ppc ;  LBA_FT_CKPT ─► e3_realdata_validate, e4_representation
e4_probe ─► e4_compare ;  e4_representation needs e3_finetune too
all source trees ─► e{1,1a,2,3,4}_finalize ─► outputs/finals/experiment-*
```

---

## Experiment 1 — generalising across design space → `outputs/finals/experiment-1/`

| Stage | Script (`experiments/…`) | Artifact | Working location | Finals |
|---|---|---|---|---|
| `e1_data` | `experiment_1/benchmark_data.py` | data: shared bench5 datasets × ddm/rdm/cdm/lba (200 × 500 trials; `rts choices design_matrices true_set param_masks [favored_idx] num_alternatives`) | `BENCH/<fam>_<case>_data.npz` + `BENCH/.done` | — (never copied; the test set itself) |
| `e1_train` | `experiment_1/model_class_cf_train.py` | checkpoint: model class (DDM+RDM+CDM, r4c4) | `MC_CKPT` (+ `MC_CKPT.done`) | — (sha256 in README) |
| | | figures: in-training validation (epoch 1000, every 2500) | `FIG/TAG_train/recovery/` | — |
| | | table: wall-clock | `outputs/timing/mc_cf.json` | — |
| `e1_validate` | `experiment_1/model_class_cf_validate.py` | figures + tables: per-case `<plots>` (6 cases × 3 fam) | `FIG/TAG_validate/<fam>/<case>/<fam>_<case>_fm_<plot>.pdf/.csv` | `per_case/<fam>/<case>/<plot>[_pi].pdf`, `metrics[_pi].csv` |
| | | data: posterior draws (`pred_set true_set params_mask max_num_categories`) | `DATA/TAG_validate/<fam>_<case>_cf_pred.npz` | `predictions/per_case/<fam>_<case>.npz` |
| `e1_ensemble` | `experiment_1/model_class_cf_ensemble.py` | figures: random-design ensemble `<plots>` (12 configs) | `FIG/TAG_ensemble/model_class_ensemble_<plot>[_global].pdf` | `random_ensemble/<plot>[_pi].pdf` |
| | | table: sampled design configs | `FIG/TAG_ensemble/model_class_ensemble_configs.csv` | `random_ensemble/design_configs.csv` |
| | | data: ensemble draws | `FIG/TAG_ensemble/model_class_ensemble_data.npz` | `predictions/random_ensemble.npz` |
| `e1_benchmark` | `experiment_1/model_class_cf_benchmark_ensemble.py` (reads e1_validate preds) | figures: bench5 / bench6 pooled `<plots>` per family | `FIG/TAG_ensemble_benchmark/<fam>_<bench>_ensemble_<plot>[_global].pdf` | `benchmark_ensemble/<bench>/<fam>/<plot>[_pi].pdf` |
| `e1_design_eig` | `experiment_1/ddm_design_eig.py` | figure + table: information gain vs trial count (DDM) | `outputs/data/design_opt/TAG/ddm/design_eig.{pdf,csv}` | `information_gain/information_gain.{pdf,csv}` |
| | | data: per-cell contraction / NRMSE, posterior draws | `…/ddm/design_{contraction,nrmse,draws}.npz` | `information_gain/{contraction_per_cell,nrmse_per_cell,posterior_draws}.npz` |
| `e1_finalize` | `finalize_experiment_1.py` | tables | — | `MANIFEST.csv`, `README.md` |

## Experiment 1a — amortization gap (DDM/RDM/CDM) → `outputs/finals/experiment-1a/`

| Stage | Script | Artifact | Working location | Finals |
|---|---|---|---|---|
| `e1a_bf` (train) | `experiment_1a/model_family_bf_train.py` | checkpoint: 15 BayesFlow baselines | `outputs/checkpoints/model_family/bf/<fam>/<fam>_families_bf_<case>/model.keras` | — |
| | | table: wall-clock | `outputs/timing/mf_bf_<fam>_<case>.json` | — |
| `e1a_bf` (validate, on `BENCH`) | `experiment_1a/model_family_bf_validate.py` | data: shared datasets + BF draws (`… pred_set`) — also each case's resume marker | `outputs/data/model_family/<fam>_<case>_data.npz` | — |
| | | figures + table: BF `<plots>` | `outputs/figures/model_family/bf/<fam>/<case>/<fam>_family_<case>_bf_<plot>.pdf`, `…_bf_metrics.csv` | `<fam>/<case>/bf_<plot>[_pi].pdf`, `bf_metrics.csv` |
| | | table: NRMSE / calibration / contraction | `outputs/evaluations/model_family/bf/<fam>_families_bf_<case>_evaluations.csv` | `<fam>/<case>/bf_evaluations.csv` |
| (CogFormer arm) | = `e1_validate` | figures: same datasets, CogFormer `<plots>` | `FIG/TAG_validate/<fam>/<case>/` | `<fam>/<case>/cf_<plot>[_pi].pdf` |
| `e1a_c2st` | `experiment_1a/model_family_c2st.py` (`BF_vs_Original`) | tables: per-cell + joint C2ST per case; family summary | `outputs/figures/c2st/<fam>/<case>/<fam>_<case>_orig_vs_bf_c2st.csv`, `outputs/figures/c2st/<fam>/<fam>_c2st_summary.csv` | `<fam>/<case>/orig_vs_bf.csv`, `<fam>/summary.csv` |
| `e1a_finalize` | `finalize_experiment_1a.py` | tables | — | `MANIFEST.csv`, `README.md` (`amortization_gap_tables.tex`: `experiment_1a/gap_tables.py`, re-run after finalize) |

## Experiment 2 — real data, DDM/RDM/CDM → `outputs/finals/experiment-2/`

| Stage | Script | Artifact | Working location | Finals |
|---|---|---|---|---|
| `e2_apply` | `experiment_2/{ddm,rdm,cdm}_mc_cf_apply.py` | data: per-subject posteriors (500 draws) | `outputs/data/real_data/<fam>/mc/<fam>_mc_posteriors.npz` | `posteriors/<fam>.npz` |
| `e2_refit_svfix` | `experiment_2/ddm_mc_cf_apply.py --fix_s_v` | data: DDM refit, s_v fixed at 0 | `outputs/data/real_data/ddm/mc_sv_fixed/ddm_mc_posteriors.npz` | `posteriors/ddm_sv_fixed.npz` |
| `e2_ppc` | `experiment_2/ppc.py` | figures: PPC per condition — `ppc` (RT), `choice_ppc`, `rt_quantiles`, `rt_qq`, `choice_quantiles` (CDM), `combined_ppc` | `outputs/figures/real_data/ppc/<fam>/<fam>_mc_<type>_<cond>.pdf` | `<fam>/<type>_<cond>.pdf` (`ppc`→`rt_ppc`) |
| | | same, constrained variants | `…/ppc/ddm_sv0/`, `…/ppc/cdm_sv0/`, `…/ppc/ddm_sv_fixed/` | `ddm/sv0/`, `cdm/sv0/`, `ddm/sv_fixed/` |
| | `experiment_2/combined_all.py` | figure: all three families on one page (from the PPC cache) | `outputs/figures/real_data/ppc/combined_ppc.pdf` | `combined_ppc.pdf` |
| | | data: simulation cache (keyed on posteriors' sha256) | `outputs/data/real_data/ppc_cache/*.pkl` | — |
| `e2_validate` | `experiment_2/realdata_validate.py` (×4: ddm, ddm `--fix_s_v`, rdm, cdm) | figures + tables: `<plots>` on the real subjects' designs | `FIG/TAG_realdata/<fam>/<realdata\|realdata_sv_fixed>/<fam>_<cfg>_fm_<plot>.pdf/.csv` | `<fam>/validation/`, `ddm/sv_fixed/validation/` |
| | | data: draws (`pred_set true_set params_mask subject_ids`) | `DATA/TAG_realdata/<fam>_<cfg>_cf_pred.npz` | `…/validation/predictions.npz` |
| `e2_finalize` | `finalize_experiment_2.py` | tables | — | `MANIFEST.csv`, `README.md` |

## Experiment 3 — held-out LBA → `outputs/finals/experiment-3/`

| Stage | Script | Artifact | Working location | Finals |
|---|---|---|---|---|
| `e3_bf` | `experiment_1a/model_family_bf_{train,validate}.py --model lba` | checkpoint / data / figures / table: 5 BF-LBA baselines (as `e1a_bf`, `<fam>=lba`) | `outputs/{checkpoints,data,figures,evaluations}/model_family/…lba…` | `gap/<case>/bf_<plot>[_pi].pdf`, `gap/<case>/bf_evaluations.csv` |
| `e3_zeroshot` | `experiment_3/model_class_lba_cf_validate.py` (MC_CKPT, LBA row ← RDM) | figures + tables: LBA `<plots>` (6 scen.) | `LBA/zeroshot/A0.5/<scen>/mid_rdm/lba_<scen>_mid1_<plot>.pdf/.csv` | `zeroshot/lba/<scen>/<plot>[_pi].pdf` |
| | | data: draws | `LBA/zeroshot/A0.5/predictions/lba_<scen>_mid1_cf_pred.npz` | `predictions/zeroshot/` |
| `e3_finetune` | `experiment_3/model_class_cf_finetune_lba.py` (scheme 2, 1000 ep) | checkpoint: 4-family finetune | `LBA_FT_CKPT` (+ `.done`) | — |
| | | figures: in-training validation | `outputs/figures/model_class/cf/lba_finetune/recovery/` | — |
| `e3_ft_validate` | `experiment_3/model_class_lba_finetune_validate.py` | figures + tables: 4 families × 6 scen. `<plots>` | `LBA/finetuned/<fam>/<scen>/<fam>_<scen>_ft_<plot>.pdf/.csv` | `finetuned/<fam>/<scen>/<plot>[_pi].pdf` |
| | | data: draws | `LBA/finetuned/predictions/<fam>_<scen>_ft_pred.npz` | `predictions/finetuned/` |
| `e3_original` | (symlinks, no compute) | = e1_validate figures + preds | `LBA/original/{ddm,rdm,cdm}` → `FIG/TAG_validate/<fam>`; `LBA/original/predictions` → `DATA/TAG_validate` | `original/<fam>/<scen>/`, `predictions/original/` |
| `e3_paired_c2st` | `experiment_3/model_class_paired_c2st.py` (bench5, `--strict`) | tables: forgetting C2ST vs empirical null — summary, per cell, per dataset | `LBA/c2st/forgetting_c2st_summary.csv`, `LBA/c2st/<fam>/<fam>_<scen>_c2st.csv`, `…_per_dataset_c2st.csv` | `c2st/…` (family prefix stripped) |
| `e3_compare` | `experiment_3/model_class_lba_compare.py` | figures: Δr heatmaps (LBA: ft − zero-shot; others: ft − original) | `LBA/compare/<fam>/<case>_{delta_r,forgetting_delta_r}.pdf` | `compare/<fam>/…` |
| | | table + figures: per-parameter metric deltas | `LBA/compare/delta_metrics.csv`, `delta_metrics_<case>.pdf` | `compare/…` |
| | `experiment_3/calibration_plane.py` (panel A, paired) | figure + table: contraction vs calibration, zero-shot → finetuned | `LBA/compare/lba_calibration_plane.{pdf,csv}` | `compare/lba_calibration_plane.{pdf,csv}` |
| `e3_gap_c2st` | `experiment_1a/model_family_c2st.py --model lba` (`BF_vs_Finetuned`) | tables: LBA amortization-gap C2ST | `outputs/figures/c2st/lba/lba_c2st_summary.csv`, `…/lba/<case>/lba_<case>_ft_vs_bf_c2st.csv` | `gap/summary.csv`, `gap/<case>/ft_vs_bf_c2st.csv` |
| `e3_ppc_apply` | `experiment_2/lba_mc_cf_apply.py --design condition_only` | data: 17-subject posteriors (stimulus-coded) | `outputs/data/real_data/lba_stim_cond/mc/lba_mc_posteriors.npz` | `ppc/posteriors.npz` |
| `e3_ppc` | `experiment_2/ppc.py` ×2 | figures: PPC free / conventional (s_v=1, s_τ=0) | `outputs/figures/real_data/ppc/lba_stim_cond[_conventional]/lba_mc_<type>_<cond>.pdf` | `ppc/free/`, `ppc/conventional/` |
| `e3_realdata_validate` | `experiment_2/realdata_validate.py --model lba` (LBA_FT_CKPT) | figures + tables + data: `<plots>` on the real designs | `FIG/TAG_realdata/lba/realdata/lba_realdata_fm_*`, `DATA/TAG_realdata/lba_realdata_cf_pred.npz` | `ppc/validation/` (+ `predictions.npz`) |
| `e3_finalize` | `finalize_experiment_3.py` | tables | — | `MANIFEST.csv`, `README.md` |
| `e3_gap_finalize` | `finalize_experiment_3.py` (second pass, in the `gap` group) | tables | — | adds `gap/` once `e3_bf` + `e3_gap_c2st` exist; `e3_finalize` in `main` runs before them and omits it |

## Experiment 4 — encoder representation (aDDM vsvfixed) → `outputs/finals/experiment-4/`

Finals here are an explicit 26-file allowlist (`KEEP_DEST` in `finalize_experiment_4.py`);
everything else produced is excluded on purpose. Finals paths carry no variant or case
name (no `vsvfixed`, no `intercept_only`): it is the only aDDM result. Working locations keep them.

| Stage | Script (`experiment_4/…`) | Artifact | Working location | Finals |
|---|---|---|---|---|
| `e4_probe` (frozen) | `addm_bf_apply.py` train/validate | checkpoint: frozen encoder + FlowMatching head | `outputs/checkpoints/encoder_rep/bf/addm_vsvfixed/intercept_only/addm_encoder_rep_fm/model.keras` | — |
| | | data: validation summaries + draws | `outputs/data/encoder_rep/addm_vsvfixed/intercept_only/addm_encoder_rep_data.npz` | `probe/frozen/data.npz` |
| | `addm_diagnostics.py --arm frozen` | figures + table: recovery, coverage, calibration_ecdf, z_score_contraction; bf_metrics | `outputs/data/encoder_rep/addm_vsvfixed/intercept_only/addm_intercept_only_<diag>.pdf`, `…_bf_metrics.csv` | `probe/frozen/<diag>.pdf`, `bf_metrics.csv` |
| `e4_probe` (baseline) | `addm_bf_baseline.py` (checkpoint REUSED unless `FORCE`) | checkpoint: learned SetTransformer + FlowMatching | `outputs/checkpoints/encoder_rep/bf_baseline/addm_vsvfixed/intercept_only/item0_vsvfixed_e500/addm_baseline_fm/model.keras` | — |
| | `addm_bf_baseline.py` validate + `addm_diagnostics.py --arm baseline` | data + figures + table | `outputs/figures/encoder_rep/addm_baseline_vsvfixed/intercept_only/item0_vsvfixed_e500/addm_baseline_{data.npz, intercept_only_<diag>.pdf, intercept_only_bf_metrics.csv}` | `probe/baseline/{data.npz, <diag>.pdf, bf_metrics.csv}` |
| `e4_compare` | `addm_arm_compare.py` | data + tables: shared validation set, C2ST arms/null, summary | `outputs/data/encoder_rep/addm_compare/intercept_only_vsvfixed/{shared_validation.npz, c2st_joint.npz, c2st_identified.npz, c2st_summary.csv, contraction_shared.csv, logprob_paired.csv}` | `probe/compare/{shared_validation.npz, c2st_joint.npz, c2st_identified.npz, c2st_summary.csv}` |
| | | figure: arm comparison | `outputs/figures/encoder_rep/addm_compare/intercept_only_vsvfixed/addm_arm_compare_intercept_only_vsvfixed.pdf` | `probe/compare/comparison.pdf` |
| | `combined_recovery.py` | figure: frozen vs baseline recovery | `…/addm_compare/intercept_only_vsvfixed/addm_arm_compare_intercept_only_vsvfixed_recovery.pdf` | `probe/compare/recovery.pdf` |
| `e4_representation` | `crossfamily_representation.py` (MC_CKPT + LBA_FT_CKPT) | figure + data: 5-family pooled MDS | `outputs/figures/encoder_rep/representation/crossfamily/app3_pooled_geometry.pdf`, `outputs/data/encoder_rep/representation/crossfamily/app3_summaries.npz` | `representation/crossfamily/…` |
| | | data: RAW encoder inputs (reload with `--raw_data`) | `…/representation/crossfamily/app3_raw_data.npz` | — (too large) |
| | `addm_representation.py` | figures + data: RSA/CKA frozen vs learned | `outputs/figures/encoder_rep/representation/intercept_only_vsvfixed/{rdm_rsa,gram_cka,rsa_vs_null,cka_vs_null}.pdf`, `outputs/data/encoder_rep/representation/intercept_only_vsvfixed/summaries.npz` | `representation/frozen_vs_learned/…` |
| | | data: RAW encoder inputs (reload with `--raw_data`) | `…/intercept_only_vsvfixed/raw_data.npz` | — (too large) |
| | `combined_panel.py` | figure: combined page (geometry + representation) | `…/addm_compare/intercept_only_vsvfixed/addm_arm_compare_intercept_only_vsvfixed_panel.pdf` | `probe/compare/panel.pdf` |
| `e4_finalize` | `finalize_experiment_4.py` | tables | — | `MANIFEST.csv`, `README.md` |

## Never copied to finals (but needed or useful)

| Artifact | Location | Why it matters |
|---|---|---|
| checkpoints | `outputs/checkpoints/…` | inputs to every downstream stage; copy (with `.done`) between hosts |
| shared benchmark data | `BENCH/` | THE bench5 test set; regenerating it unpairs every arm |
| raw representation inputs | `…/representation/*/raw_data.npz`, `app3_raw_data.npz` | the only way to rebuild Exp 4 figures on identical datasets |
| PPC cache | `outputs/data/real_data/ppc_cache/` | re-plots PPCs without re-simulating |
| in-training validation figures | `FIG/TAG_train/`, `…/lba_finetune/recovery/` | training health checks |
| run timings | `outputs/timing/*.json` | planning estimates |
| stage logs | `logs/finals/<RUN_ID>/<stage>.log` | debugging |
