"""
Anomaly Detection - neocekivani skokovi/padovi u potraznji
==========================================================
Dvije komplementarne metode:

1. STATISTICKA: residual z-score (Sales - rolling_mean) / rolling_std
   - Brza, interpretabilna
   - Pretpostavka normalnosti rezidual-a
   - Threshold: |z| > 3 = anomalija

2. ISOLATION FOREST: ML-based unsupervised anomaly detection
   - Hvata kompleksne kombinacije feature-a
   - Ne zahtjeva labele
   - Contamination: ocekivani % anomalija
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "/home/ognjen/Desktop/DataMIning/projekat/data/processed"
OUT_DIR = "/home/ognjen/Desktop/DataMIning/projekat/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

print("Ucitavam podatke...")
df = pd.read_parquet(f"{DATA_DIR}/processed_full.parquet")

# ============================================================
# METODA 1: Z-SCORE NA ROLLING REZIDUALU
# ============================================================
print(f"\n{'='*70}")
print("METODA 1: STATISTICKA ANOMALY DETECTION (Z-SCORE)")
print(f"{'='*70}")

def detect_zscore_anomalies(group, window=14, threshold=3.0):
    """
    Za svaki red izracunamo:
      rolling_mean = prosjek poslednjih `window` dana
      rolling_std = std poslednjih `window` dana
      z_score = (Sales - rolling_mean) / rolling_std
    Anomalija = |z_score| > threshold
    """
    group = group.sort_values("Date").copy()
    rolling_mean = group["Sales"].shift(1).rolling(window).mean()
    rolling_std = group["Sales"].shift(1).rolling(window).std()
    group["z_score"] = (group["Sales"] - rolling_mean) / (rolling_std + 1e-9)
    group["is_anomaly_zscore"] = (group["z_score"].abs() > threshold).astype(int)
    return group

print("Racunam z-score po prodavnici (traje ~30s)...")
df_anom = df.groupby("Store", group_keys=False).apply(detect_zscore_anomalies)

n_anomalies_z = df_anom["is_anomaly_zscore"].sum()
total = len(df_anom)
print(f"\nUkupno anomalija (z-score): {n_anomalies_z} / {total} ({n_anomalies_z/total*100:.2f}%)")

# Top 10 najekstremnijih anomalija
print(f"\nTOP 10 NAJEKSTREMNIJIH POZITIVNIH ANOMALIJA (skok prodaje):")
top_pos = df_anom.nlargest(10, "z_score")[["Store", "Date", "Sales", "z_score", "Promo", "IsHoliday", "SchoolHoliday"]]
print(top_pos.to_string(index=False))

print(f"\nTOP 10 NAJEKSTREMNIJIH NEGATIVNIH ANOMALIJA (pad prodaje):")
top_neg = df_anom.nsmallest(10, "z_score")[["Store", "Date", "Sales", "z_score", "Promo", "IsHoliday", "SchoolHoliday"]]
print(top_neg.to_string(index=False))

# ============================================================
# METODA 2: ISOLATION FOREST (ML-based)
# ============================================================
print(f"\n{'='*70}")
print("METODA 2: ISOLATION FOREST")
print(f"{'='*70}")

# Demonstracija na jednoj prodavnici (Store 1)
DEMO_STORE = 1
sdf = df[df["Store"] == DEMO_STORE].copy().sort_values("Date").reset_index(drop=True)

# Feature-i za detekciju
iso_features = [
    "Sales", "Customers", "Promo", "DayOfWeek", "IsHoliday",
    "SchoolHoliday", "Sales_lag_1", "Sales_lag_7", "Sales_rolling_mean_14",
]

X = sdf[iso_features].values

print(f"Treniram Isolation Forest na Store {DEMO_STORE} ({len(X)} redova)...")
iso = IsolationForest(
    contamination=0.02,  # ocekujemo ~2% anomalija
    random_state=42,
    n_estimators=200,
    n_jobs=-1,
)
sdf["anomaly_iso"] = iso.fit_predict(X)
sdf["anomaly_score"] = iso.score_samples(X)
# -1 = anomalija, 1 = normalno -> konvertujemo u 0/1
sdf["is_anomaly_iso"] = (sdf["anomaly_iso"] == -1).astype(int)

n_iso = sdf["is_anomaly_iso"].sum()
print(f"Anomalija detektovano (Isolation Forest): {n_iso} ({n_iso/len(sdf)*100:.2f}%)")

print(f"\nTOP 10 ANOMALIJA NA STORE {DEMO_STORE} (Isolation Forest):")
iso_anom = sdf[sdf["is_anomaly_iso"] == 1].nsmallest(10, "anomaly_score")[
    ["Date", "Sales", "Customers", "Promo", "IsHoliday", "SchoolHoliday", "anomaly_score"]
]
print(iso_anom.to_string(index=False))

# ============================================================
# 3. AUTOMATSKO OBJASNJENJE ANOMALIJA
# ============================================================
print(f"\n{'='*70}")
print("AUTOMATSKA OBJASNJENJA (na osnovu konteksta)")
print(f"{'='*70}")

def explain_anomaly(row):
    """Generise tekstualno objasnjenje zasto je dan anomalija."""
    reasons = []
    
    if row["Promo"] == 1:
        reasons.append("aktivna PROMOCIJA")
    if row["IsHoliday"] == 1:
        reasons.append("drzavni praznik")
    if row["SchoolHoliday"] == 1:
        reasons.append("skolski raspust")
    if row.get("IsCGHoliday", 0) == 1:
        reasons.append("crnogorski praznik")
    if row["DayOfWeek"] == 1:
        reasons.append("ponedjeljak (uobicajeno visa prodaja)")
    if row.get("IsWeekend", 0) == 1:
        reasons.append("vikend")
    if row.get("TouristSeason", 0) >= 1:
        reasons.append(f"turisticka sezona (intenzitet={row['TouristSeason']})")
    
    z = row.get("z_score", 0)
    if z > 0:
        magnitude = f"SKOK prodaje ({z:+.2f} std)"
    else:
        magnitude = f"PAD prodaje ({z:+.2f} std)"
    
    if not reasons:
        explanation = f"{magnitude}: nema ocitog konteksta - moguce eksterni faktor."
    else:
        explanation = f"{magnitude}: {' + '.join(reasons)}"
    
    return explanation

# Pokazimo na top 5 z-score anomalija
print(f"\nObjasnjenja za top 5 anomalija (z-score):")
top5 = df_anom.nlargest(5, "z_score")
for _, row in top5.iterrows():
    expl = explain_anomaly(row)
    print(f"  Store {int(row['Store']):4d} | {row['Date'].date()} | Sales={row['Sales']:>6.0f} -> {expl}")

# ============================================================
# 4. SACUVAJ
# ============================================================
# Sacuvamo samo anomalije za laksu vizualizaciju
anomalies_only = df_anom[df_anom["is_anomaly_zscore"] == 1][
    ["Store", "Date", "Sales", "z_score", "Promo", "IsHoliday", "SchoolHoliday",
     "IsCGHoliday", "DayOfWeek", "IsWeekend", "TouristSeason"]
].copy()
anomalies_only["explanation"] = anomalies_only.apply(explain_anomaly, axis=1)
anomalies_only.to_csv(f"{OUT_DIR}/anomalies_zscore.csv", index=False)

sdf[["Date", "Sales", "is_anomaly_iso", "anomaly_score"]].to_csv(
    f"{OUT_DIR}/anomalies_isolation_store{DEMO_STORE}.csv", index=False
)

print(f"\nSacuvano:")
print(f"  {OUT_DIR}/anomalies_zscore.csv ({len(anomalies_only)} anomalija)")
print(f"  {OUT_DIR}/anomalies_isolation_store{DEMO_STORE}.csv")
print("\nAnomaly detection gotov!")
