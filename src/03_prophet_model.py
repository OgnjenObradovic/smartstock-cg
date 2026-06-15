"""
Prophet baseline model
- Trenira se po prodavnici (Prophet je univariate)
- Demonstriramo na jednoj prodavnici, pa skaliramo
- Sa eksternim regresorima: Promo, IsHoliday, SchoolHoliday
- Confidence intervali ukljuceni
"""
import pandas as pd
import numpy as np
from prophet import Prophet
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "/home/ognjen/Desktop/DataMIning/projekat/data/processed"
MODELS_DIR = "/home/ognjen/Desktop/DataMIning/projekat/models"
os.makedirs(MODELS_DIR, exist_ok=True)

print("Ucitavam podatke...")
df = pd.read_parquet(f"{DATA_DIR}/processed_full.parquet")
print(f"Shape: {df.shape}")

# ============================================================
# 1. DEMO: PROPHET NA JEDNOJ PRODAVNICI (Store=1)
# ============================================================
STORE_ID = 1
print(f"\n{'='*60}")
print(f"PROPHET MODEL ZA PRODAVNICU {STORE_ID}")
print(f"{'='*60}")

store_df = df[df["Store"] == STORE_ID].copy().sort_values("Date")
print(f"Broj dana: {len(store_df)}")
print(f"Datumski opseg: {store_df['Date'].min()} -> {store_df['Date'].max()}")

# Prophet formati: 'ds' (date), 'y' (target)
prophet_df = store_df[["Date", "Sales", "Promo", "IsHoliday", "SchoolHoliday", "IsCGHoliday"]].rename(
    columns={"Date": "ds", "Sales": "y"}
)

# Train/test split: posljednjih 42 dana = test (6 sedmica)
split_date = prophet_df["ds"].max() - pd.Timedelta(days=42)
train_df = prophet_df[prophet_df["ds"] <= split_date].copy()
test_df = prophet_df[prophet_df["ds"] > split_date].copy()
print(f"Train: {len(train_df)} dana | Test: {len(test_df)} dana")

# 2. INICIJALIZUJ PROPHET
print("\nTreniram Prophet...")
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode="multiplicative",  # bolje za retail (skala raste sa trendom)
    interval_width=0.95,  # 95% confidence interval
    changepoint_prior_scale=0.05,
)

# Dodaj eksterne regresore
model.add_regressor("Promo")
model.add_regressor("IsHoliday")
model.add_regressor("SchoolHoliday")
model.add_regressor("IsCGHoliday")

# Treniraj
model.fit(train_df)
print("Prophet istreniran!")

# 3. PREDIKCIJA NA TEST SETU
print("\nPredikcija...")
future = test_df[["ds", "Promo", "IsHoliday", "SchoolHoliday", "IsCGHoliday"]].copy()
forecast = model.predict(future)

# 4. EVALUACIJA
y_true = test_df["y"].values
y_pred = forecast["yhat"].values

mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

print(f"\n{'='*60}")
print(f"REZULTATI ZA STORE {STORE_ID}")
print(f"{'='*60}")
print(f"MAE:   {mae:.2f}")
print(f"RMSE:  {rmse:.2f}")
print(f"MAPE:  {mape:.2f}%")
print(f"SMAPE: {smape:.2f}%")
print(f"Mean Sales: {y_true.mean():.2f}")
print(f"Pokrivenost 95% CI: {((y_true >= forecast['yhat_lower']) & (y_true <= forecast['yhat_upper'])).mean() * 100:.1f}%")

# 5. SACUVAJ MODEL
with open(f"{MODELS_DIR}/prophet_store_{STORE_ID}.pkl", "wb") as f:
    pickle.dump(model, f)
print(f"\nModel sacuvan: {MODELS_DIR}/prophet_store_{STORE_ID}.pkl")

# 6. SACUVAJ FORECAST CSV (za vizualizaciju kasnije)
out_df = test_df[["ds", "y"]].copy()
out_df["yhat"] = forecast["yhat"].values
out_df["yhat_lower"] = forecast["yhat_lower"].values
out_df["yhat_upper"] = forecast["yhat_upper"].values
out_df.to_csv(f"{MODELS_DIR}/prophet_forecast_store_{STORE_ID}.csv", index=False)
print(f"Forecast sacuvan: {MODELS_DIR}/prophet_forecast_store_{STORE_ID}.csv")

# 7. PROVJERI NEKOLIKO PRODAVNICA (sample)
print(f"\n{'='*60}")
print("KRATKA EVALUACIJA NA 5 NASUMICNIH PRODAVNICA")
print(f"{'='*60}")
np.random.seed(42)
sample_stores = np.random.choice(df["Store"].unique(), 5, replace=False)

results = []
for sid in sample_stores:
    sdf = df[df["Store"] == sid].copy().sort_values("Date")
    pdf = sdf[["Date", "Sales", "Promo", "IsHoliday", "SchoolHoliday", "IsCGHoliday"]].rename(
        columns={"Date": "ds", "Sales": "y"}
    )
    split = pdf["ds"].max() - pd.Timedelta(days=42)
    tr = pdf[pdf["ds"] <= split]
    te = pdf[pdf["ds"] > split]
    
    if len(tr) < 100 or len(te) < 10:
        print(f"  Store {sid}: nedovoljno podataka, preskacem")
        continue
    
    m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                seasonality_mode="multiplicative", interval_width=0.95)
    for reg in ["Promo", "IsHoliday", "SchoolHoliday", "IsCGHoliday"]:
        m.add_regressor(reg)
    m.fit(tr)
    
    fut = te[["ds", "Promo", "IsHoliday", "SchoolHoliday", "IsCGHoliday"]]
    f = m.predict(fut)
    
    yt = te["y"].values
    yp = f["yhat"].values
    sm = np.mean(2 * np.abs(yt - yp) / (np.abs(yt) + np.abs(yp))) * 100
    ma = np.mean(np.abs(yt - yp))
    results.append({"Store": sid, "MAE": ma, "SMAPE": sm})
    print(f"  Store {sid:4d}: MAE={ma:7.2f} | SMAPE={sm:.2f}%")

results_df = pd.DataFrame(results)
print(f"\nProsjecna SMAPE: {results_df['SMAPE'].mean():.2f}%")
print(f"Prosjecni MAE: {results_df['MAE'].mean():.2f}")

print("\nProphet baseline gotov!")
