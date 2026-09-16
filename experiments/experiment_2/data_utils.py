"""
Utilities for packaging real behavioral data into the CogFormer batch format.

Real data enters as raw arrays (rts, choices, condition columns). This module
converts them to the padded design-matrix / param-mask format expected by
Adapter.adapt, processing one subject at a time (batch_size=1).

Design matrix column layout (keep_intercept=True, max_num_categories=C,
max_num_regressors=R, block_width = C-1, inter_block_width = (C-1)²):

  col 0              : intercept (1.0)
  cols 1..bw         : u_1 block  (first col = regressor value, rest zero for continuous)
  cols bw+1..2bw     : u_2 block
  ...
  cols R*bw+1..R*bw+ibw : u_1:u_2 block  (ibw = inter_block_width)
  ... (interactions follow all main effects)

Continuous regressors occupy the FIRST column of their block.
Categorical regressors (dummy-coded) occupy columns 0..(n_dummies-1) of their block.
Interaction blocks hold the outer product of each pair of parent blocks.
"""
import numpy as np
from cogformer.simulator.design_utils import build_parameter_mask, build_position_codes


def _regressor_keys(design_config: dict) -> list[str]:
    """Return non-intercept keys in design_config order."""
    return [k for k in design_config if k != "1"]


def _compute_max_num_cols(
    max_num_regressors: int,
    max_num_categories: int,
    keep_intercept: bool,
) -> int:
    block_width = max_num_categories - 1
    n_main = max_num_regressors
    n_inter = max_num_regressors * (max_num_regressors - 1) // 2
    return n_main * block_width + n_inter * block_width ** 2 + (1 if keep_intercept else 0)


def build_real_design_matrix(
    regressors: dict[str, np.ndarray],
    design_config: dict,
    max_num_regressors: int,
    max_num_categories: int,
    keep_intercept: bool = True,
) -> np.ndarray:
    """
    Build a zero-padded design matrix from per-trial regressor arrays.

    Parameters
    ----------
    regressors : dict mapping design_config key → array of shape (num_obs,) or (num_obs, n_cols)
        Continuous regressors: 1-D array → placed in first column of block.
        Categorical / interaction regressors: 2-D array → placed in first n_cols columns of block.
        e.g. {"u_1": speed_cond, "u_2": difficulty_dummies, "u_1:u_2": interaction_dummies}
    design_config : dict
        Same format as CogFormer design configs.
    max_num_regressors, max_num_categories : ints
        Must match the trained CogFormer.
    """
    block_width = max_num_categories - 1
    inter_block_width = block_width ** 2
    max_num_cols = _compute_max_num_cols(max_num_regressors, max_num_categories, keep_intercept)

    num_obs = next(iter(regressors.values())).shape[0]
    dm = np.zeros((num_obs, max_num_cols), dtype=np.float32)

    col_offset = 0
    if keep_intercept:
        dm[:, 0] = 1.0
        col_offset = 1

    keys = _regressor_keys(design_config)
    cur = col_offset
    start_pos = {}
    for key in keys:
        start_pos[key] = cur
        cur += inter_block_width if ":" in key else block_width

    for key in keys:
        if key not in regressors:
            continue
        arr = np.asarray(regressors[key], dtype=np.float32)
        start = start_pos[key]
        width = inter_block_width if ":" in key else block_width
        if arr.ndim == 1:
            dm[:, start] = arr
        else:
            n_cols = min(arr.shape[1], width)
            dm[:, start:start + n_cols] = arr[:, :n_cols]

    return dm


def build_real_param_mask(
    design_config: dict,
    intrinsic_params: list[str],
    max_num_regressors: int,
    max_num_categories: int,
    keep_intercept: bool = True,
) -> np.ndarray:
    """
    Build a padded param_mask (max_num_cols, n_params) for real data.

    Returns shape (max_num_cols * n_params,) — flat, matching Adapter expectations.
    """
    max_num_cols = _compute_max_num_cols(max_num_regressors, max_num_categories, keep_intercept)
    n_params = len(intrinsic_params)

    local_mask = build_parameter_mask(
        design_config=design_config,
        intrinsic_params=intrinsic_params,
        fixed_params=None,
        max_num_categories=max_num_categories,
        keep_intercept=keep_intercept,
    )  # shape: (n_local_rows, n_params)

    padded = np.zeros((max_num_cols, n_params), dtype=np.float32)
    padded[:local_mask.shape[0], :] = local_mask
    return padded.flatten()


def build_subject_batch(
    rts: np.ndarray,
    choices: np.ndarray,
    design_matrix: np.ndarray,
    param_mask: np.ndarray,
    n_params: int,
    max_num_regressors: int,
    max_num_categories: int,
    design_config: dict,
    num_alternatives: int = 2,
    model_id: int | None = None,
    favored_idx: np.ndarray | None = None,
) -> dict:
    """
    Pack a single subject's data into the batch dict expected by Adapter.adapt.

    Parameters
    ----------
    rts, choices : (num_obs,) arrays — real observations
    design_matrix : (num_obs, max_num_cols)
    param_mask : flat (max_num_cols * n_params,) array
    design_config : dict — used to build factorized position codes; must be the same
        config the design matrix and param mask were built from
    favored_idx : (num_obs,) or None — which accumulator carries v + v_diff on each
        trial. REQUIRED whenever num_alternatives > 2 and the favored accumulator
        varies by trial.

        Adapter.adapt reads this from ``per_trial_context`` and SILENTLY DEFAULTS TO
        ZEROS when it is absent, i.e. "accumulator 0 is favored on every trial" with
        no error raised. The simulators vary favored_idx per trial during training, so
        omitting it here is a train/apply mismatch that produces plausible output.
        Left as None for K = 2, where the simulators hardcode accumulator 1 as favored
        and no per-trial index exists.
    """
    rts = np.asarray(rts, dtype=np.float32).reshape(-1, 1)
    choices = np.asarray(choices, dtype=np.float32).reshape(-1, 1)
    num_obs = rts.shape[0]

    max_num_cols = len(param_mask) // n_params
    codes = build_position_codes(design_config, max_num_categories=max_num_categories)
    position_codes = np.zeros((max_num_cols, 4), dtype=np.int64)
    position_codes[:codes.shape[0]] = codes

    batch = {
        "position_codes":   position_codes[np.newaxis],           # (1, max_num_cols, 4)
        "design_matrices":  design_matrix[np.newaxis],            # (1, num_obs, max_num_cols)
        "sim_data": {
            "rts":     rts[np.newaxis],                           # (1, num_obs, 1)
            "choices": choices[np.newaxis],                       # (1, num_obs, 1)
        },
        "param_matrices":   np.zeros((1, len(param_mask)), dtype=np.float32),
        "param_masks":      param_mask[np.newaxis],               # (1, max_num_cols * n_params)
        "n_params":         n_params,
        "max_num_regressors": max_num_regressors,
        "max_num_categories": max_num_categories,
        "num_alternatives": np.array([num_alternatives], dtype=np.int32),
    }
    if favored_idx is not None:
        batch["per_trial_context"] = {
            "favored_idx": np.asarray(favored_idx, dtype=np.float32).reshape(1, num_obs)
        }
    elif num_alternatives > 2:
        raise ValueError(
            f"num_alternatives={num_alternatives} but favored_idx was not supplied. "
            "The adapter would default it to zeros, telling the encoder accumulator 0 "
            "is favored on every trial. Pass favored_idx explicitly."
        )
    if model_id is not None:
        batch["model_ids"] = np.array([model_id], dtype=np.int64)
    return batch


def load_ddm_data(csv_path: str) -> tuple[dict, list[int]]:
    """
    Load DDM CSV data (Stump et al., 2026).

    Regressors:
      u_1  : stim_type,  effect coded ±0.5  (continuous)
      u_2  : truth,      effect coded ±0.5  (continuous)
      u_1:u_2 : stim_type × truth           (continuous × continuous → 1 column)
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    counts = df.groupby("id")["id"].transform("size")
    df = df[counts >= 100].copy()

    data = {}
    for sid in sorted(df["id"].unique()):
        sdf = df[df["id"] == sid].reset_index(drop=True)
        stim_type   = (sdf["stim_type"].values * 1.0 - 0.5).astype(np.float32)
        truth       = (sdf["factual_truth"].values * 1.0 - 0.5).astype(np.float32)
        interaction = (stim_type * truth).astype(np.float32)
        data[int(sid)] = {
            "rts":         sdf["rt"].values.astype(np.float32),
            "choices":     sdf["resp"].values.astype(np.float32),
            "u_1":         stim_type,
            "u_2":         truth,
            "u_1:u_2":     interaction,
        }
    return data, sorted(data.keys())


def load_rdm_data(csv_path: str) -> tuple[dict, list[int]]:
    """
    Load RDM CSV data (3-alternative Stroop, 149 subjects).

    Regressors:
      u_1     : congruent, effect coded ±0.5  (congruent=+0.5, incongruent=-0.5)
      u_2     : session,   effect coded ±0.5  (session 0 → -0.5, session 1 → +0.5)
      u_1:u_2 : congruent × session           (continuous × continuous → 1 column)

    favored_idx is the INK colour (``color``), i.e. the correct response, and it is the
    accumulator that carries v + v_diff. It was previously not returned at all, so
    build_subject_batch never passed it and Adapter.adapt silently defaulted it to
    zeros -- telling the encoder "accumulator 0 is favored on every trial" for a
    3-alternative task where that is true on only 33% of trials, while the simulator
    varied it per trial throughout training.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    counts = df.groupby("id")["id"].transform("size")
    df = df[counts >= 100].copy()

    data = {}
    for sid in sorted(df["id"].unique()):
        sdf = df[df["id"] == sid].reset_index(drop=True)
        congruent   = (sdf["congruent"].values * 1.0 - 0.5).astype(np.float32)
        session     = (sdf["session"].values * 1.0 - 0.5).astype(np.float32)
        interaction = (congruent * session).astype(np.float32)
        data[int(sid)] = {
            "rts":         sdf["rt"].values.astype(np.float32),
            "choices":     sdf["resp"].values.astype(np.float32),
            "favored_idx": sdf["color"].values.astype(np.float32),
            "u_1":         congruent,
            "u_2":         session,
            "u_1:u_2":     interaction,
        }
    return data, sorted(data.keys())


def load_cdm_data(csv_path: str) -> tuple[dict, list[int]]:
    """
    Load CDM CSV data (Kvam, 2019).

    Regressors:
      u_1     : speed_condition, effect coded ±0.5      (continuous, 1 column)
      u_2     : difficulty (3 levels: 1,2,3), dummy coded to (d2, d3)
                reference category = 1                  (categorical, 2 columns)
      u_1:u_2 : speed_condition × (d2, d3)             (cat×cont → 2 columns)

    RESPONSE SCALE: raw ``resp_deviation`` spans [-pi/2, pi/2], but CDM emits
    ``arctan2(x1, x0)`` on [-pi, pi] — the column is a HALF-ANGLE. It is doubled
    here so data and simulator share a support. Evidence (2026-08-19, mc r4c4,
    500 draws, choice-dimension PPC): KS 0.067->0.026 (Speeded) and 0.099->0.034
    (Unspeeded); PPC/observed angular sd mismatch +29% -> +3%; quantiles inside
    the 95% band 1/5 -> 5/5 for Unspeeded. Confirm against Kvam (2019)'s response
    apparatus before treating as settled.
    """
    RESP_DEVIATION_SCALE = 2.0
    import pandas as pd
    df = pd.read_csv(csv_path)

    data = {}
    for sid in sorted(df["id"].unique()):
        sdf = df[df["id"] == sid].reset_index(drop=True)
        speed = (sdf["speed_condition"].values * 1.0 - 0.5).astype(np.float32)
        difficulty = sdf["difficulty"].values

        # Dummy coding: reference category = 1
        d2 = (difficulty == 2).astype(np.float32)
        d3 = (difficulty == 3).astype(np.float32)

        # cat × cont interaction: each dummy × speed_condition
        inter_d2 = d2 * speed
        inter_d3 = d3 * speed

        data[int(sid)] = {
            "rts":     sdf["rt"].values.astype(np.float32),
            "choices": (sdf["resp_deviation"].values * RESP_DEVIATION_SCALE).astype(np.float32),
            "u_1":     speed,
            "u_2":     np.column_stack([d2, d3]),          # (num_obs, 2)
            "u_1:u_2": np.column_stack([inter_d2, inter_d3]),  # (num_obs, 2)
        }
    return data, sorted(data.keys())


def load_lba_data(csv_path: str) -> tuple[dict, list[int]]:
    """
    Load LBA CSV data — lexical decision, Wagenmakers, Ratcliff, Gomez & McKoon (2008),
    distributed as ``speed_acc`` in the R package ``rtdists`` (GPL-3).

    Regressors (a balanced 2 x 3 x 2 factorial):
      u_1     : condition, speed vs accuracy, effect coded +/-0.5  (1 column)
      u_2     : freq_band (high / low / very_low), dummy coded to
                (low, very_low); reference = high                  (2 columns)
      u_3     : lexicality, word vs nonword, effect coded +/-0.5   (1 column)
      u_1:u_2 : condition x (low, very_low)                        (2 columns)

    WHY freq_band AND lexicality RATHER THAN ONE 6-LEVEL FACTOR. The source column
    ``frequency`` has six levels -- high, low, very_low and nw_high, nw_low,
    nw_very_low -- which exceeds max_num_categories=4. They are not six unrelated
    conditions though: nonwords are frequency-matched to their base words, so the
    column is a 3x2 factorial written as one string. Splitting it keeps both
    manipulations; collapsing the three nonword levels into a single bucket, the
    obvious alternative, silently discards the nonword frequency manipulation.

    CHOICES ARE STIMULUS-CODED (changed 2026-09-16). Accumulator 0 is a "nonword"
    response and accumulator 1 a "word" response, and ``favored_idx`` names the
    accumulator the STIMULUS calls for on each trial -- 1 on word trials, 0 on
    nonword trials. ``v_diff`` is therefore the drift advantage of the
    stimulus-appropriate accumulator, and the two accumulators are the two RESPONSES,
    which is both the standard LBA parameterisation for lexical decision and the same
    convention load_ddm_data uses. The two applications are now comparable on v_diff.

    This used to be accuracy coding (``choices = correct``), forced by the simulator:
    LBA.prepare_params hardcoded accumulator 1 as favored for K = 2 and had no
    favored_idx path. That path exists as of 2026-09-15, so the coding is now a
    property of the data rather than of the model.

    BOTH quantities come from columns already present. The response is recoverable
    because a correct trial means the response matched the stimulus:

        favored_idx = 1 if lexicality == "word" else 0
        response    = favored_idx if correct else 1 - favored_idx

    Verified on the shipped CSV: response equals favored_idx on 89.1% of trials,
    which is exactly the accuracy, as it must be.

    ``favored_idx`` MUST be forwarded to build_subject_batch and to the PPC's
    simulation context. Adapter.adapt defaults the channel to zeros when it is absent
    and raises nothing, so an unforwarded favored_idx tells the encoder "accumulator 0
    is favoured" on every trial while the choices say otherwise.

    NOTE u_3 (lexicality) is kept as a regressor even though it now also determines
    favored_idx. The two are not redundant: favored_idx selects WHICH accumulator is
    boosted, u_3 modulates HOW LARGE that boost is for words against nonwords.

    Cleaning is the dataset's own ``censor`` flag, applied when the CSV is built:
    31,351 of 31,522 trials retained, RT 0.181-2.986 s, and every ``error`` response
    removed.

    SEQUENCE LENGTH IS OUT OF DISTRIBUTION. Median 1,917 trials per subject against a
    trained num_obs range of 200-1000 (both the base model-class run and the LBA
    finetune used max_num_obs=1000). Each subject is fitted whole, so datasets are
    ~1.9x the longest sequence the encoder saw in training. If posteriors look wrong,
    suspect this before the design.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)

    data = {}
    for sid in sorted(df["id"].unique()):
        sdf = df[df["id"] == sid].reset_index(drop=True)

        # speed = +0.5, accuracy = -0.5, matching load_ddm_data's binary coding.
        cond = np.where(sdf["condition"].values == "speed", 0.5, -0.5).astype(np.float32)

        # Dummy coding, reference = "high" (the easiest, highest-drift level).
        band = sdf["freq_band"].values
        d_low = (band == "low").astype(np.float32)
        d_vlo = (band == "very_low").astype(np.float32)

        is_word = (sdf["lexicality"].values == "word")
        lex = np.where(is_word, 0.5, -0.5).astype(np.float32)

        # Stimulus coding: accumulator 1 = "word" response, 0 = "nonword" response.
        # favored_idx is the stimulus-appropriate accumulator; the response followed
        # it on correct trials and departed from it on errors.
        favored = is_word.astype(np.int32)
        correct = sdf["correct"].values.astype(bool)
        response = np.where(correct, favored, 1 - favored).astype(np.float32)

        data[int(sid)] = {
            "rts":         sdf["rt"].values.astype(np.float32),
            "choices":     response,                                   # 1 = "word"
            "favored_idx": favored,                                    # 1 on word trials
            "u_1":         cond,
            "u_2":         np.column_stack([d_low, d_vlo]),            # (num_obs, 2)
            "u_3":         lex,
            "u_1:u_2":     np.column_stack([d_low * cond, d_vlo * cond]),
        }
    return data, sorted(data.keys())


def load_stroop_data(csv_path: str, min_trials: int = 60) -> tuple[dict, list[int]]:
    """
    Load Stroop colour-word data for a 3-accumulator race model.

    Regressors:
      u_1 : congruency, effect coded +/-0.5 (incongruent -0.5, congruent +0.5)

    WHAT THIS DESIGN CAN AND CANNOT MEASURE. The simulator gives every accumulator
    drift ``v`` and the single ``favored_idx`` accumulator ``v + v_diff``, so a trial
    has exactly TWO distinct drifts. Stroop's natural parameterisation
    (v ~ ink_match + word_match) needs THREE: the ink colour, the word-named colour,
    and the irrelevant third. The encoder cannot express it either -- its per-trial
    input carries ``favored_idx`` as one scalar, with no per-accumulator channel.

    So this fits the REDUCED design: favored = the ink colour (the correct response),
    and v_diff regressed on congruency, which measures interference as a reduction in
    the target's drift advantage. It captures the RT and accuracy Stroop effect
    (measured here: 0.673 -> 0.794 s, 0.983 -> 0.941 accuracy) but NOT the error
    asymmetry -- 71.4% of incongruent errors go to the word-named colour, and with both
    distractors sharing one drift that direction is unmodellable. Do not describe
    output from this as a full three-drift Stroop model.

    Choices are accumulator INDICES (0=blue, 1=green, 2=red), not accuracy codes: with
    K > 2 the favored accumulator is named explicitly by favored_idx, so the response
    can stay in its natural coding.

    SHORT SEQUENCES. Median 96 trials per participant (range 15-96) against a trained
    num_obs range of 200-1000 -- under half the minimum, so every dataset is out of
    distribution on length. ``min_trials`` drops the shortest participants; it does not
    fix the problem, it only bounds how bad it gets.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)

    # Fixed ordering so accumulator index means the same thing for every participant.
    colours = ["blue", "green", "red"]
    idx = {c: i for i, c in enumerate(colours)}

    counts = df.groupby("participant")["participant"].transform("size")
    df = df[counts >= min_trials]

    data = {}
    for k, (pid, sdf) in enumerate(df.groupby("participant", sort=True)):
        sdf = sdf.reset_index(drop=True)
        congruent = (sdf["ink"].values == sdf["word"].values)
        data[k] = {
            "rts":          sdf["rt"].values.astype(np.float32),
            "choices":      sdf["response"].map(idx).values.astype(np.float32),
            "favored_idx":  sdf["ink"].map(idx).values.astype(np.float32),
            "u_1":          np.where(congruent, 0.5, -0.5).astype(np.float32),
            "participant":  pid,
        }
    return data, sorted(data.keys())
