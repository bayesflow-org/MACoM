# Experiment 2 constrained analyses (archived)

These 41 result files were moved out of the standard final-results snapshot on
2026-11-11. They are not inputs to the overall combined_ppc.pdf.

- ddm/sv0/: standard DDM fit with s_v forced to zero only during simulation.
- cdm/sv0/: standard CDM fit with s_v forced to zero only during simulation.
- ddm/sv_fixed/: a separate DDM fit with s_v fixed at fitting, including validation.
- posteriors/ddm_sv_fixed.npz: posterior draws for that separate fit.

MANIFEST.csv preserves each result's source record. README.before_cleanup.md
preserves the previous analysis notes. Source predictions, figures and cached
simulations remain in outputs/data/ and outputs/figures/.

Restore these files by assembling with:

    python -m experiments.finalize_experiment_2 --include_constrained
