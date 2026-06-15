"""SHAP - koristimo Booster direktno da izbjegnemo XGBoost wrapper bug."""
import pandas as pd, numpy as np, xgboost as xgb, shap, pickle, os, json, warnings
warnings.filterwarnings("ignore")

DATA_DIR = "/home/ognjen/Desktop/DataMIning/projekat/data/processed"
MODELS_DIR = "/home/ognjen/Desktop/DataMIning/projekat/models"
OUT_DIR = "/home/ognjen/Desktop/DataMIning/projekat/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_parquet(f"{DATA_DIR}/processed_full.parquet")
with open(f"{MODELS_DIR}/xgboost_features.pkl", "rb") as f:
    FEATURES = pickle.load(f)

# Ucitaj kao Booster direktno
booster = xgb.Booster()
booster.load_model(f"{MODELS_DIR}/xgboost_global.json")

# Fix base_score
cfg = json.loads(booster.save_config())
bs = cfg["learner"]["learner_model_param"]["base_score"]
if bs.startswith("["):
    cfg["learner"]["learner_model_param"]["base_score"] = bs.strip("[]")
    booster.load_config(json.dumps(cfg))

print("Sampliram 3000 redova...")
sample = df.sample(n=3000, random_state=42).copy()
X_sample = sample[FEATURES].copy()
for col in X_sample.select_dtypes(include="bool").columns:
    X_sample[col] = X_sample[col].astype(int)

print("Racunam SHAP...")
explainer = shap.TreeExplainer(booster)
shap_values = explainer.shap_values(X_sample)
print(f"SHAP shape: {shap_values.shape}")

# Globalna importance
shap_imp = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
shap_imp = shap_imp.sort_values("mean_abs_shap", ascending=False)
print("\nTOP 15 SHAP feature-a:")
print(shap_imp.head(15).to_string(index=False))
shap_imp.to_csv(f"{OUT_DIR}/shap_global_importance.csv", index=False)

# Sacuvaj
dmatrix = xgb.DMatrix(X_sample.values, feature_names=FEATURES)
yhat = booster.predict(dmatrix)
sample["yhat"] = yhat
with open(f"{OUT_DIR}/shap_results.pkl", "wb") as f:
    pickle.dump({
        "shap_values": shap_values, "feature_names": FEATURES,
        "X_sample": X_sample.values,
        "sample_metadata": sample[["Store", "Date", "Sales", "yhat"]].reset_index(drop=True),
        "baseline": float(explainer.expected_value),
    }, f)
print(f"\nSacuvano u {OUT_DIR}")
