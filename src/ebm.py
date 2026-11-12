import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from interpret.glassbox import ExplainableBoostingRegressor


# -----------------------------------------
# 1. Load dataset
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "EDM_dataset_262_samples.csv"

df = pd.read_csv(CSV_PATH)


# -----------------------------------------
# 2. Select inputs and target
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
# 3. Train/test split
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# -----------------------------------------
# 4. Create EBM
# -----------------------------------------

ebm = ExplainableBoostingRegressor(
    interactions=5,
    random_state=42
)


# -----------------------------------------
# 5. Train EBM
# -----------------------------------------

ebm.fit(X_train, y_train)


# -----------------------------------------
# 6. Predict
# -----------------------------------------

y_pred = ebm.predict(X_test)


# -----------------------------------------
# 7. Evaluate
# -----------------------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(y_test, y_pred)


print("\nEBM Results")
print("----------------")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)


from interpret import show

# -----------------------------------------
import matplotlib.pyplot as plt

# -----------------------------------------
# 8. Print EBM feature importance
# -----------------------------------------

print("\nEBM Feature Importance")
print("----------------------")

importance = ebm.term_importances()

for name, value in zip(ebm.term_names_, importance):
    print(f"{name}: {value:.6f}")


# -----------------------------------------
# 9. Plot EBM global importance
# -----------------------------------------

names = ebm.term_names_
values = ebm.term_importances()

plt.figure(figsize=(10, 6))

plt.barh(names, values)

plt.xlabel("Mean Absolute Contribution")
plt.ylabel("Feature / Interaction")
plt.title("EBM Feature and Interaction Importance")

plt.tight_layout()
plt.show()
