import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from interpret.glassbox import ExplainableBoostingRegressor


# -----------------------------------------
# 1. Load dataset
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "EDM_dataset_262_samples.csv"
df = pd.read_csv(CSV_PATH)


# -----------------------------------------
# 2. Inputs and target
# -----------------------------------------

X = df[
    [
        "Ip_A",
        "Vg_V",
        "Ton_us",
        "Toff_us",
        "Flushing_Pressure_kgf_cm2"
    ]
]

y = df["SR_um"]


# -----------------------------------------
# 3. Create 5-fold cross-validation
# -----------------------------------------

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# -----------------------------------------
# 4. Store results
# -----------------------------------------

results = {
    "Linear Regression": [],
    "Random Forest": [],
    "XGBoost": [],
    "EBM (No Interactions)": [],
    "EBM (With Interactions)": []
}


# -----------------------------------------
# 5. Cross-validation
# -----------------------------------------

for fold, (train_index, test_index) in enumerate(
    kf.split(X), start=1
):

    print(f"\n========== Fold {fold} ==========")

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]


    # -------------------------------------
    # Linear Regression
    # -------------------------------------

    lr = LinearRegression()

    lr.fit(X_train, y_train)

    pred = lr.predict(X_test)

    r2 = r2_score(y_test, pred)

    results["Linear Regression"].append(r2)

    print("Linear Regression R²:", round(r2, 4))


    # -------------------------------------
    # Random Forest
    # -------------------------------------

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    rf.fit(X_train, y_train)

    pred = rf.predict(X_test)

    r2 = r2_score(y_test, pred)

    results["Random Forest"].append(r2)

    print("Random Forest R²:", round(r2, 4))


    # -------------------------------------
    # XGBoost
    # -------------------------------------

    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )

    xgb.fit(X_train, y_train)

    pred = xgb.predict(X_test)

    r2 = r2_score(y_test, pred)

    results["XGBoost"].append(r2)

    print("XGBoost R²:", round(r2, 4))


 # -------------------------------------
    # -------------------------------------
    # EBM WITHOUT interactions
    # -------------------------------------

    ebm_no_interactions = ExplainableBoostingRegressor(
        interactions=0,
        random_state=42
    )

    ebm_no_interactions.fit(X_train, y_train)

    pred = ebm_no_interactions.predict(X_test)

    r2 = r2_score(y_test, pred)

    results["EBM (No Interactions)"].append(r2)

    print("EBM (No Interactions) R²:", round(r2, 4))


    # -------------------------------------
    # EBM WITH interactions
    # -------------------------------------

    ebm_interactions = ExplainableBoostingRegressor(
        interactions=5,
        random_state=42
    )

    ebm_interactions.fit(X_train, y_train)

    pred = ebm_interactions.predict(X_test)

    r2 = r2_score(y_test, pred)

    results["EBM (With Interactions)"].append(r2)

    print("EBM (With Interactions) R²:", round(r2, 4))

# -----------------------------------------
# 6. Final results
# -----------------------------------------

print("\n\n===================================")
print("5-FOLD CROSS-VALIDATION RESULTS")
print("===================================")

for model, scores in results.items():

    mean_r2 = np.mean(scores)
    std_r2 = np.std(scores)

    print(f"\n{model}")
    print("Fold R²:", [round(x, 4) for x in scores])
    print("Mean R²:", round(mean_r2, 4))
    print("Std R² :", round(std_r2, 4))


    # -----------------------------------------
# 7. Save cross-validation results
# -----------------------------------------

RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Save individual fold scores
fold_results = pd.DataFrame({
    "Fold": [1, 2, 3, 4, 5],
    "Linear Regression": results["Linear Regression"],
    "Random Forest": results["Random Forest"],
    "XGBoost": results["XGBoost"],
    "EBM (No Interactions)": results["EBM (No Interactions)"],
    "EBM (With Interactions)": results["EBM (With Interactions)"]
})

fold_results.to_csv(
    RESULTS_DIR / "Cross_Validation_Fold_Results.csv",
    index=False
)

# Save summary statistics
summary_results = pd.DataFrame([
    {
        "Model": model,
        "Mean R2": np.mean(scores),
        "Std R2": np.std(scores)
    }
    for model, scores in results.items()
])

summary_results.to_csv(
    RESULTS_DIR / "Cross_Validation_Summary.csv",
    index=False
)

print("\nCross-validation CSV files saved successfully!")
    