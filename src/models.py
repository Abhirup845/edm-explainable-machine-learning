import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------------------
# 1. Load the dataset
# -----------------------------------------

df = pd.read_csv("EDM_dataset_262_samples.csv")


# -----------------------------------------
# 2. Select input features and target
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
# 3. Split data into training and testing
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------------------
# 4. Create the Linear Regression model
# -----------------------------------------

model = LinearRegression()


# -----------------------------------------
# 5. Train the model
# -----------------------------------------

model.fit(X_train, y_train)


# -----------------------------------------
# 6. Make predictions
# -----------------------------------------

y_pred = model.predict(X_test)


# -----------------------------------------
# 7. Evaluate the model
# -----------------------------------------

mae = mean_absolute_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(y_test, y_pred)


print("\nLinear Regression Results")
print("-------------------------")
print("MAE :", mae)
print("RMSE:", rmse)
print("R²  :", r2)

print("\nLinear Regression Coefficients")
print("------------------------------")

for feature, coefficient in zip(X.columns, model.coef_):
    print(f"{feature}: {coefficient:.6f}")

print(f"\nIntercept: {model.intercept_:.6f}")

# -----------------------------------------
# 8. Random Forest
# -----------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

# Train
rf_model.fit(X_train, y_train)

# Predict
rf_pred = rf_model.predict(X_test)

# Evaluate
rf_mae = mean_absolute_error(y_test, rf_pred)

rf_rmse = mean_squared_error(
    y_test,
    rf_pred
) ** 0.5

rf_r2 = r2_score(y_test, rf_pred)


print("\nRandom Forest Results")
print("---------------------")
print("MAE :", rf_mae)
print("RMSE:", rf_rmse)
print("R²  :", rf_r2)

# -----------------------------------------
# 9. XGBoost
# -----------------------------------------

xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    random_state=42
)

# Train
xgb_model.fit(X_train, y_train)

# Predict
xgb_pred = xgb_model.predict(X_test)

# Evaluate
xgb_mae = mean_absolute_error(y_test, xgb_pred)

xgb_rmse = mean_squared_error(
    y_test,
    xgb_pred
) ** 0.5

xgb_r2 = r2_score(y_test, xgb_pred)

print("\nXGBoost Results")
print("----------------")
print("MAE :", xgb_mae)
print("RMSE:", xgb_rmse)
print("R²  :", xgb_r2)

