import pandas as pd
from pathlib import Path
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


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
# 3. Train/test split
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# -----------------------------------------
# 4. Train XGBoost
# -----------------------------------------

model = XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)


# -----------------------------------------
# 5. Create SHAP explainer
# -----------------------------------------

explainer = shap.TreeExplainer(model)

shap_values = explainer(X_test)


# -----------------------------------------
# 6. SHAP summary plot
# -----------------------------------------

shap.plots.beeswarm(shap_values, show=False)

plt.title("SHAP Summary Plot — EDM Surface Roughness")
plt.tight_layout()
plt.show()