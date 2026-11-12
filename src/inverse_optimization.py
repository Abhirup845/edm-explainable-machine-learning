
import itertools
import math
import numpy as np
import pandas as pd
from pathlib import Path
from interpret.glassbox import ExplainableBoostingRegressor

# --------------------------------------------------
# 1. Load the EDM dataset
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

CSV_FILE = BASE_DIR / "data" / "EDM_dataset_262_samples.csv"

df = pd.read_csv(CSV_FILE)

FEATURES = [
    "Ip_A",
    "Vg_V",
    "Ton_us",
    "Toff_us",
    "Flushing_Pressure_kgf_cm2"
]
TARGET = "SR_um"

df = pd.read_csv(CSV_FILE)
df.columns = df.columns.str.strip()

required = FEATURES + [TARGET]
missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}\n"
        f"Columns found: {df.columns.tolist()}"
    )

df = df[required].apply(pd.to_numeric, errors="coerce")
df = df.replace([np.inf, -np.inf], np.nan).dropna()

if len(df) < 20:
    raise ValueError("Too few valid observations after cleaning.")

X = df[FEATURES]
y = df[TARGET]

print(f"Valid observations: {len(df)}")
print(f"Observed SR range: {y.min():.4f} to {y.max():.4f} um")

# --------------------------------------------------
# 2. Train the EBM surrogate model
# --------------------------------------------------
model = ExplainableBoostingRegressor(
    interactions=5,
    random_state=42
)
model.fit(X, y)

print("\nEBM model trained successfully.")

# --------------------------------------------------
# 3. Enter the desired surface roughness
# --------------------------------------------------
target_sr = float(
    input("\nEnter your target surface roughness (um): ")
)

if not math.isfinite(target_sr):
    raise ValueError("Target SR must be a finite number.")

if not (y.min() <= target_sr <= y.max()):
    print(
        "\nWARNING: Your target is outside the observed "
        "SR range. Recommendations would require extrapolation."
    )

# --------------------------------------------------
# 4. Generate feasible factor-level combinations
# --------------------------------------------------
# Use values actually observed for each input parameter.
levels = [sorted(X[col].unique()) for col in FEATURES]

total_combinations = math.prod(len(v) for v in levels)
print(f"\nPossible factor-level combinations: {total_combinations}")

if total_combinations <= 300_000:
    candidates = list(itertools.product(*levels))
    candidates = pd.DataFrame(candidates, columns=FEATURES)
else:
    # Keep computation manageable if there are many levels.
    rng = np.random.default_rng(42)
    n_samples = 300_000
    candidates = pd.DataFrame({
        col: rng.choice(X[col].unique(), size=n_samples)
        for col in FEATURES
    }).drop_duplicates()

# --------------------------------------------------
# 5. Predict SR and calculate target error
# --------------------------------------------------
candidates["Predicted_SR_um"] = model.predict(
    candidates[FEATURES]
)

candidates["Absolute_Error_um"] = (
    candidates["Predicted_SR_um"] - target_sr
).abs()

results = candidates.sort_values(
    "Absolute_Error_um"
).reset_index(drop=True)

# Display the ten closest parameter combinations.
top_results = results.head(10).copy()
top_results.insert(
    0, "Rank", range(1, len(top_results) + 1)
)

print("\nTOP 10 INVERSE-OPTIMIZATION RECOMMENDATIONS")
print(f"Target SR = {target_sr:.4f} um\n")

print(top_results.to_string(
    index=False,
    float_format=lambda value: f"{value:.5f}"
))

# --------------------------------------------------
# 6. Save the results
# --------------------------------------------------
output_file = "Inverse_Optimization_Results_2.csv"
top_results.to_csv(output_file, index=False)

print(f"\nResults saved to: {output_file}")
print(
    "\nIMPORTANT: These are model-based recommendations, "
    "not experimentally verified settings."
)
