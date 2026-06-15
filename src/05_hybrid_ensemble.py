"""
Hibridni model: Prophet + XGBoost
Strategija: 
  1. Prophet predviđa trend + sezonu (rezidualni baseline)
  2. XGBoost predviđa rezidual (sto Prophet promaši)
  3. Final = Prophet_forecast + XGBoost_residual_forecast

Ovo je standardni pristup u industriji - Prophet je dobar
za makro patterns, XGBoost je dobar za kompleksne interakcije.

Confidence intervali dolaze iz Prophet-ovog yhat_lower/upper,
prosireno sa XGBoost residual std.
"""
import pandas as pd
import numpy as np
from prophet import Prophet
import xgboost as xgb
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "/home/ognjen/Desktop/DataMIning/projekat/data/processed"
MODELS_DIR = "/home/ognjen/Desktop/DataMIning/projekat/models"

print("Ucitavam podatke...")
df = pd.read_parquet(f"{DATA_DIR}/processed_full.parquet")

# Odaberi 10 prodavnica za hibridni demo (Prophet je spor pa ne za sve 1115)
np.random.seed(42)
DEMO_STORES = np.random.choice(df["Store"].unique(), 10, replace=False).tolist()
print(f"Demo na prodavnicama: {DEMO_STORES}")

# Učitaj globalni XGBoost
print("\nUcitavam XGBoost...")
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(f"{MODELS_DIR}/xgboost_global.json")
with open(f"{MODELS_DIR}/xgboost_features.pkl", "rb") as f:
    XGB_FEATURES = pickle.load(f)

DROP_COLS = ["Date", "Sales", "Customers", "StateHoliday", "PromoInterval", "Open"]
TARGET = "Sales"

# ============================================================
# HIBRIDNI POSTUPAK
# ============================================================
results = []

for store_id in DEMO_STORES:
    print(f"\n{'='*60}")
    print(f"HIBRID ZA STORE {store_id}")
    print(f"{'='*60}")
    
    sdf = df[df["Store"] == store_id].copy().sort_values("Date").reset_index(drop=True)
    
    if len(sdf) < 200:
        print(f"  Preskacem - premalo podataka")
        continue
    
    split_date = sdf["Date"].max() - pd.Timedelta(days=42)
    train = sdf[sdf["Date"] <= split_date].copy()
    test = sdf[sdf["Date"] > split_date].copy()
    
    # ---------------------------
    # 1. PROPHET na trainu
    # ---------------------------
    prophet_train = train[["Date", "Sales", "Promo", "IsHoliday", "SchoolHoliday", "IsCGHoliday"]].rename(
        columns={"Date": "ds", "Sales": "y"}
    )
    
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                seasonality_mode="multiplicative", interval_width=0.95,
                changepoint_prior_scale=0.05)
    for reg in ["Promo", "IsHoliday", "SchoolHoliday", "IsCGHoliday"]:
        m.add_regressor(reg)
    m.fit(prophet_train)
    
    # Prophet predikcija na trainu (za izracunavanje rezidual-a)
    train_fcst = m.predict(prophet_train[["ds", "Promo", "IsHoliday", "SchoolHoliday", "IsCGHoliday"]])
    train_prophet_pred = train_fcst["yhat"].values
    train_residual = train["Sales"].values - train_prophet_pred
    
    # Prophet predikcija na testu
    test_input = test[["Date", "Promo", "IsHoliday", "SchoolHoliday", "IsCGHoliday"]].rename(columns={"Date": "ds"})
    test_fcst = m.predict(test_input)
    test_prophet_pred = test_fcst["yhat"].values
    test_yhat_lower = test_fcst["yhat_lower"].values
    test_yhat_upper = test_fcst["yhat_upper"].values
    
    # ---------------------------
    # 2. XGBOOST na rezidualu
    # ---------------------------
    X_train = train[XGB_FEATURES].copy()
    X_test = test[XGB_FEATURES].copy()
    for col in X_train.select_dtypes(include="bool").columns:
        X_train[col] = X_train[col].astype(int)
        X_test[col] = X_test[col].astype(int)
    
    # Treniraj mali XGBoost da predvidja rezidual (ne pune Sales)
    resid_model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:squarederror", tree_method="hist",
        random_state=42, n_jobs=-1,
    )
    resid_model.fit(X_train, train_residual, verbose=False)
    
    test_residual_pred = resid_model.predict(X_test)
    
    # ---------------------------
    # 3. FINAL HIBRID
    # ---------------------------
    y_true = test["Sales"].values
    
    # Prophet sam
    y_prophet = test_prophet_pred
    # XGBoost sam (globalni)
    y_xgb_global = np.maximum(xgb_model.predict(X_test), 0)
    # Hibrid: Prophet baseline + XGBoost residual
    y_hybrid = np.maximum(test_prophet_pred + test_residual_pred, 0)
    
    # Confidence interval za hibrid: Prophet CI prosiren rezidualnim std
    resid_std = np.std(train_residual)
    y_hybrid_lower = np.maximum(test_yhat_lower + test_residual_pred - 1.96 * resid_std * 0.5, 0)
    y_hybrid_upper = test_yhat_upper + test_residual_pred + 1.96 * resid_std * 0.5
    
    def metrics(yt, yp):
        mae = np.mean(np.abs(yt - yp))
        rmse = np.sqrt(np.mean((yt - yp) ** 2))
        smape = np.mean(2 * np.abs(yt - yp) / (np.abs(yt) + np.abs(yp) + 1e-9)) * 100
        return mae, rmse, smape
    
    mae_p, rmse_p, smape_p = metrics(y_true, y_prophet)
    mae_x, rmse_x, smape_x = metrics(y_true, y_xgb_global)
    mae_h, rmse_h, smape_h = metrics(y_true, y_hybrid)
    
    ci_cov = np.mean((y_true >= y_hybrid_lower) & (y_true <= y_hybrid_upper)) * 100
    
    print(f"  Prophet sam:   MAE={mae_p:7.2f} | SMAPE={smape_p:5.2f}%")
    print(f"  XGBoost sam:   MAE={mae_x:7.2f} | SMAPE={smape_x:5.2f}%")
    print(f"  HIBRID:        MAE={mae_h:7.2f} | SMAPE={smape_h:5.2f}% | CI cov={ci_cov:.1f}%")
    
    results.append({
        "Store": store_id,
        "Prophet_MAE": mae_p, "Prophet_SMAPE": smape_p,
        "XGB_MAE": mae_x, "XGB_SMAPE": smape_x,
        "Hybrid_MAE": mae_h, "Hybrid_SMAPE": smape_h,
        "CI_coverage": ci_cov,
    })

# ============================================================
# SUMMARY
# ============================================================
results_df = pd.DataFrame(results)
print(f"\n{'='*60}")
print("FINALNO POREDJENJE (10 prodavnica)")
print(f"{'='*60}")
print(results_df.to_string(index=False))

print(f"\n{'='*60}")
print("PROSJEK")
print(f"{'='*60}")
print(f"Prophet:  MAE={results_df['Prophet_MAE'].mean():7.2f} | SMAPE={results_df['Prophet_SMAPE'].mean():.2f}%")
print(f"XGBoost:  MAE={results_df['XGB_MAE'].mean():7.2f} | SMAPE={results_df['XGB_SMAPE'].mean():.2f}%")
print(f"HIBRID:   MAE={results_df['Hybrid_MAE'].mean():7.2f} | SMAPE={results_df['Hybrid_SMAPE'].mean():.2f}%")
print(f"CI coverage prosjek: {results_df['CI_coverage'].mean():.1f}%")

# Sacuvaj
results_df.to_csv(f"{MODELS_DIR}/hybrid_comparison.csv", index=False)
print(f"\nSacuvano: {MODELS_DIR}/hybrid_comparison.csv")
print("\nHibrid gotov!")
