"""
XGBoost globalni model
- Jedan model za SVE prodavnice (Store kao kategorijski feature)
- Koristi sve feature-e (lag, rolling, promo, kompeticija, CG kontekst)
- Vremenski split (NE random!) - zadnja 42 dana = test
- Evaluacija: MAE, RMSE, MAPE, SMAPE
"""
import pandas as pd
import numpy as np
import xgboost as xgb
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
# 1. ODABIR FEATURE-A
# ============================================================
# Drop kolone koje nisu feature-i (target, datum, originalne pre-encoding kolone)
DROP_COLS = ["Date", "Sales", "Customers", "StateHoliday", "PromoInterval", "Open"]
FEATURES = [c for c in df.columns if c not in DROP_COLS]
TARGET = "Sales"

print(f"\nBroj feature-a: {len(FEATURES)}")
print(f"Feature-i: {FEATURES}")

# ============================================================
# 2. TIME-BASED SPLIT (KRITICNO za time series!)
# ============================================================
split_date = df["Date"].max() - pd.Timedelta(days=42)
train = df[df["Date"] <= split_date].copy()
test = df[df["Date"] > split_date].copy()

print(f"\nTrain shape: {train.shape} | Datumi: {train['Date'].min()} -> {train['Date'].max()}")
print(f"Test shape:  {test.shape} | Datumi: {test['Date'].min()} -> {test['Date'].max()}")

X_train, y_train = train[FEATURES], train[TARGET]
X_test, y_test = test[FEATURES], test[TARGET]

# Konvertuj bool u int (XGBoost zahtjeva)
for col in X_train.select_dtypes(include="bool").columns:
    X_train[col] = X_train[col].astype(int)
    X_test[col] = X_test[col].astype(int)

# ============================================================
# 3. XGBOOST MODEL
# ============================================================
print("\nTreniram XGBoost (ovo traje 2-5 min)...")

model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    objective="reg:squarederror",
    tree_method="hist",  # brze treniranje
    early_stopping_rounds=50,
    eval_metric="rmse",
    random_state=42,
    n_jobs=-1,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100,
)

print("\nXGBoost istreniran!")
print(f"Best iteration: {model.best_iteration}")

# ============================================================
# 4. PREDIKCIJA + EVALUACIJA
# ============================================================
print("\nPredikcija na test setu...")
y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0)  # negativna prodaja nemoguca

mae = np.mean(np.abs(y_test - y_pred))
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
smape = np.mean(2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred))) * 100

print(f"\n{'='*60}")
print(f"XGBOOST REZULTATI - GLOBALNI MODEL")
print(f"{'='*60}")
print(f"MAE:   {mae:.2f}")
print(f"RMSE:  {rmse:.2f}")
print(f"MAPE:  {mape:.2f}%")
print(f"SMAPE: {smape:.2f}%")
print(f"Mean Sales (test): {y_test.mean():.2f}")

# ============================================================
# 5. EVALUACIJA PO PRODAVNICAMA (za poredjenje sa Prophetom)
# ============================================================
test_eval = test[["Store", "Date", TARGET]].copy()
test_eval["yhat"] = y_pred

per_store = test_eval.groupby("Store").apply(
    lambda g: pd.Series({
        "MAE": np.mean(np.abs(g[TARGET] - g["yhat"])),
        "SMAPE": np.mean(2 * np.abs(g[TARGET] - g["yhat"]) /
                         (np.abs(g[TARGET]) + np.abs(g["yhat"]))) * 100,
    })
).reset_index()

print(f"\nPo prodavnicama:")
print(f"  Prosjecna MAE:   {per_store['MAE'].mean():.2f}")
print(f"  Prosjecna SMAPE: {per_store['SMAPE'].mean():.2f}%")
print(f"  Median SMAPE:    {per_store['SMAPE'].median():.2f}%")

# Poredjenje sa Prophet-om na istih 5 nasumicnih
np.random.seed(42)
sample_stores = np.random.choice(df["Store"].unique(), 5, replace=False)
print(f"\nNa istih 5 store-ova kao Prophet:")
sub = per_store[per_store["Store"].isin(sample_stores)]
print(sub.to_string(index=False))
print(f"  Prosjek: SMAPE={sub['SMAPE'].mean():.2f}%, MAE={sub['MAE'].mean():.2f}")

# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================
print(f"\n{'='*60}")
print("TOP 20 NAJVAZNIJIH FEATURE-A")
print(f"{'='*60}")
importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)
print(importance.head(20).to_string(index=False))

# ============================================================
# 7. SACUVAJ MODEL
# ============================================================
model.save_model(f"{MODELS_DIR}/xgboost_global.json")
with open(f"{MODELS_DIR}/xgboost_features.pkl", "wb") as f:
    pickle.dump(FEATURES, f)

# Sacuvaj predikcije i importance za kasnije
test_eval.to_parquet(f"{MODELS_DIR}/xgboost_predictions.parquet", index=False)
importance.to_csv(f"{MODELS_DIR}/xgboost_feature_importance.csv", index=False)

print(f"\nModel sacuvan: {MODELS_DIR}/xgboost_global.json")
print("XGBoost gotov!")
