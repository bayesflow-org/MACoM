"""Page styling shared by benchmark contamination recovery rows."""

GLOBAL_RECOVERY_RC = {"xtick.labelsize": 7.9, "ytick.labelsize": 7.9}


def global_recovery_options(cases):
    """Match the random ensemble's square panels and typography in one row."""
    return dict(
        share_limits=True, max_per_row=max(len(cases), 1), wspace=0.06,
        # A single row needs its own allowance for titles and x labels;
        # height-limited square axes otherwise leave large empty gutters.
        figsize=(11.0 * len(cases) / 6, 2.5),
        title_fontsize=12.6, label_fontsize=11.5, annot_fontsize=9,
        labels=[case.replace("_", " ") for case in cases],
    )
