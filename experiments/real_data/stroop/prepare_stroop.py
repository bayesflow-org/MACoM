from pathlib import Path

import pandas as pd


data_path = Path(__file__).with_name("stroop.csv")

data = pd.read_csv(data_path)
data["trial"] = data.groupby("participant").cumcount() + 1

accumulators = pd.DataFrame({"accumulator": ["red", "green", "blue"]})
expanded = data.merge(accumulators, how="cross")
expanded["ink_match"] = expanded["accumulator"].eq(expanded["ink"]).map(
    {True: 0.5, False: -0.5}
)
expanded["word_match"] = expanded["accumulator"].eq(expanded["word"]).map(
    {True: 0.5, False: -0.5}
)


# Regressor analysis:
#   t0 ~ 1            one non-decision time per participant
#   v  ~ ink_match + word_match
#   B  ~ accumulator  accumulator-specific boundaries, with red as reference
#   s  = 1            fixed drift-rate standard deviation
#   A  = 0            fixed starting-point range
first_participant = expanded["participant"].iloc[0]
first = expanded.loc[expanded["participant"] == first_participant].copy()

t0_design = pd.DataFrame(
    {"Intercept": 1.0},
    index=first.index,
)

v_design = pd.DataFrame(
    {
        "v_mean": 1.0,
        "ink_match": first["ink_match"],
        "word_match": first["word_match"],
    },
    index=first.index,
)

accumulator_dummies = pd.get_dummies(first["accumulator"], dtype=float)
B_design = pd.DataFrame(
    {
        "Intercept": 1.0,
        "accumulator[green]": accumulator_dummies["green"],
        "accumulator[blue]": accumulator_dummies["blue"],
    },
    index=first.index,
)

design_matrices = {
    "t0": t0_design,
    "v": v_design,
    "B": B_design,
}
fixed_parameters = {"s": 1.0, "A": 0.0}

first_trials = data.loc[data["participant"] == first_participant].copy()
first_trials["condition"] = first_trials["ink"].eq(first_trials["word"]).map(
    {True: "congruent", False: "incongruent"}
)
first_trials["correct"] = first_trials["response"].eq(first_trials["ink"])

condition_summary = first_trials.groupby("condition").agg(
    median_rt=("rt", "median"),
    accuracy=("correct", "mean"),
)
condition_summary = condition_summary.reindex(["congruent", "incongruent"])


print(f"Original data: {len(data):,} rows")
print(f"Expanded data: {len(expanded):,} rows")
print(f"\nFirst participant: {first_participant}")
print(first.head(20).to_string(index=False))

for parameter, matrix in design_matrices.items():
    print(f"\n{parameter} design matrix: {matrix.shape}")
    print(matrix.head(20).to_string(index=False))

print(f"\nFixed parameters: {fixed_parameters}")
print("\nRT and accuracy by condition")
print(condition_summary.to_string())
