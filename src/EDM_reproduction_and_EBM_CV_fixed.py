
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from interpret.glassbox import ExplainableBoostingRegressor

CSV_PATH = "EDM_dataset_262_samples.csv"
FEATURES = ["Ip_A", "Vg_V", "Ton_us", "Toff_us", "Flushing_Pressure_kgf_cm2"]
TARGET = "SR_um"

df = pd.read_csv(CSV_PATH)
X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        random_state=42, objective="reg:squarederror", n_jobs=1
    ),
    "EBM_interactions_0": ExplainableBoostingRegressor(interactions=0, random_state=42),
    "EBM_interactions_5": ExplainableBoostingRegressor(interactions=5, random_state=42),
}

print("\\n=== HOLD-OUT TEST METRICS ===")
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{name}: MAE={mean_absolute_error(y_test,pred):.10f}, "
          f"RMSE={mean_squared_error(y_test,pred)**0.5:.10f}, R2={r2_score(y_test,pred):.10f}")

print("\\n=== EBM 5-FOLD CROSS-VALIDATION (FIT INSIDE EACH FOLD) ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []
for model_name, interactions in [("EBM interactions=0", 0), ("EBM interactions=5", 5)]:
    fold_r2, fold_mae, fold_rmse = [], [], []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        # IMPORTANT: create a NEW EBM model and fit it inside every fold.
        model = ExplainableBoostingRegressor(interactions=interactions, random_state=42)
        Xtr, Xval = X.iloc[train_idx], X.iloc[test_idx]
        ytr, yval = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(Xtr, ytr)
        pred = model.predict(Xval)
        r2 = r2_score(yval, pred)
        mae = mean_absolute_error(yval, pred)
        rmse = mean_squared_error(yval, pred) ** 0.5
        fold_r2.append(r2); fold_mae.append(mae); fold_rmse.append(rmse)
        print(f"{model_name}, fold {fold}: R2={r2:.6f}, MAE={mae:.6f}, RMSE={rmse:.6f}")
    print(f"{model_name} SUMMARY: mean R2={np.mean(fold_r2):.6f}, "
          f"SD R2={np.std(fold_r2, ddof=0):.6f}, mean MAE={np.mean(fold_mae):.6f}, "
          f"mean RMSE={np.mean(fold_rmse):.6f}")
    cv_rows.append({"Model":model_name, "Fold_R2":fold_r2,
                    "Mean_R2":np.mean(fold_r2), "SD_R2_population":np.std(fold_r2,ddof=0),
                    "Mean_MAE":np.mean(fold_mae), "Mean_RMSE":np.mean(fold_rmse)})
pd.DataFrame(cv_rows).to_csv("EBM_5fold_CV_results.csv", index=False)
print("\\nSaved EBM_5fold_CV_results.csv")
