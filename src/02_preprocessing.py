"""
Preprocessing + Feature Engineering
- Spajanje train/store
- Vremenski feature-i
- Lag i rolling feature-i
- Crnogorski kontekst (praznici, turistička sezona, CPI proxy)
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = "/home/ognjen/Desktop/DataMIning/projekat/data"
OUT_DIR = "/home/ognjen/Desktop/DataMIning/projekat/data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

print("Ucitavam podatke...")
train = pd.read_csv(f"{DATA_DIR}/train.csv", parse_dates=["Date"], low_memory=False)
store = pd.read_csv(f"{DATA_DIR}/store.csv")

# 1. SPOJI sa store podacima
df = train.merge(store, on="Store", how="left")
print(f"Posle merge: {df.shape}")

# 2. UKLONI zatvorene dane i Sales=0 dane (nemaju informacije za forecast)
df = df[df["Open"] == 1].copy()
df = df[df["Sales"] > 0].copy()
print(f"Posle filtriranja zatvorenih: {df.shape}")

# 3. SORTIRAJ po Store i Date (vazno za lag feature-e!)
df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

# 4. VREMENSKI FEATURE-I
print("Pravim vremenske feature-e...")
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
df["DayOfYear"] = df["Date"].dt.dayofyear
df["Quarter"] = df["Date"].dt.quarter
df["IsWeekend"] = (df["DayOfWeek"] >= 6).astype(int)
df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)

# Ciklicni feature-i (vrijednost je periodicna - bolje za ML)
df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

# 5. STATEHOLIDAY ENCODING
df["StateHoliday"] = df["StateHoliday"].astype(str)
df["IsHoliday"] = (df["StateHoliday"] != "0").astype(int)

# 6. STORETYPE I ASSORTMENT - one-hot
df = pd.get_dummies(df, columns=["StoreType", "Assortment"], prefix=["ST", "AS"])

# 7. COMPETITION feature
print("Competition features...")
df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())
df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(df["Year"])
df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(df["Month"])

df["CompetitionOpenMonths"] = (
    12 * (df["Year"] - df["CompetitionOpenSinceYear"]) +
    (df["Month"] - df["CompetitionOpenSinceMonth"])
)
df["CompetitionOpenMonths"] = df["CompetitionOpenMonths"].clip(lower=0)

# 8. PROMO2 feature
df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(df["Year"])
df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(df["WeekOfYear"])
df["Promo2OpenWeeks"] = (
    52 * (df["Year"] - df["Promo2SinceYear"]) +
    (df["WeekOfYear"] - df["Promo2SinceWeek"])
)
df["Promo2OpenWeeks"] = df["Promo2OpenWeeks"].clip(lower=0)
df["Promo2OpenWeeks"] = df["Promo2OpenWeeks"] * df["Promo2"]

# 9. PROMOINTERVAL - da li je tekuci mjesec u Promo2 ciklusu
month_to_str = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                7:"Jul",8:"Aug",9:"Sept",10:"Oct",11:"Nov",12:"Dec"}
df["MonthStr"] = df["Month"].map(month_to_str)
df["PromoInterval"] = df["PromoInterval"].fillna("")
df["IsPromo2Month"] = df.apply(
    lambda r: int(r["MonthStr"] in r["PromoInterval"].split(",")) if r["Promo2"] == 1 else 0,
    axis=1
)
df = df.drop(columns=["MonthStr"])

# 10. CRNOGORSKI KONTEKST
print("Dodajem crnogorski kontekst...")

# 10a. Crnogorski drzavni praznici (zakonski neradni dani)
cg_praznici = {
    "Nova godina": [(1, 1), (1, 2)],
    "Pravoslavni Bozic": [(1, 7)],
    "Dan rada": [(5, 1), (5, 2)],
    "Dan nezavisnosti": [(5, 21), (5, 22)],
    "Dan drzavnosti": [(7, 13), (7, 14)],
}
praznici_set = set()
for naziv, datumi in cg_praznici.items():
    for m, d in datumi:
        praznici_set.add((m, d))

df["IsCGHoliday"] = df.apply(
    lambda r: int((r["Month"], r["Day"]) in praznici_set), axis=1
)

# 10b. Turisticka sezona u Crnoj Gori (jun-septembar = vrhunac)
df["TouristSeason"] = df["Month"].apply(
    lambda m: 2 if m in [7, 8] else (1 if m in [6, 9] else 0)
)

# 10c. Sezona (prolece, ljeto, jesen, zima)
def get_sezona(m):
    if m in [12, 1, 2]: return "zima"
    elif m in [3, 4, 5]: return "prolece"
    elif m in [6, 7, 8]: return "ljeto"
    else: return "jesen"
df["Sezona"] = df["Month"].apply(get_sezona)
df = pd.get_dummies(df, columns=["Sezona"], prefix="Sez")

# 11. LAG I ROLLING FEATURE-I (kljucno za time series!)
print("Lag i rolling features (ovo traje 30-60 sekundi)...")
df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

for lag in [1, 7, 14, 28]:
    df[f"Sales_lag_{lag}"] = df.groupby("Store")["Sales"].shift(lag)

for window in [7, 14, 30]:
    df[f"Sales_rolling_mean_{window}"] = (
        df.groupby("Store")["Sales"].shift(1).rolling(window).mean().reset_index(0, drop=True)
    )
    df[f"Sales_rolling_std_{window}"] = (
        df.groupby("Store")["Sales"].shift(1).rolling(window).std().reset_index(0, drop=True)
    )

# Customers lag (prosli broj kupaca je proxy za potraznju)
df["Customers_lag_7"] = df.groupby("Store")["Customers"].shift(7)

# 12. UKLONI redove sa NaN (prvih 28 dana po prodavnici)
print(f"Prije dropna: {df.shape}")
df = df.dropna().reset_index(drop=True)
print(f"Posle dropna: {df.shape}")

# 13. SACUVAJ
out_path = f"{OUT_DIR}/processed_full.parquet"
df.to_parquet(out_path, index=False)
print(f"\nSacuvano: {out_path}")
print(f"Final shape: {df.shape}")
print(f"\nKolone ({len(df.columns)}):")
print(list(df.columns))

print(f"\nMemorija: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
