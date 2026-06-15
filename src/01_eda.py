"""
EDA - Exploratory Data Analysis
Rossmann Store Sales Dataset
"""
import pandas as pd
import numpy as np

# Putanja do podataka
DATA_DIR = "/home/ognjen/Desktop/DataMIning/projekat/data"

# Ucitavanje
print("=" * 60)
print("UCITAVANJE PODATAKA")
print("=" * 60)
train = pd.read_csv(f"{DATA_DIR}/train.csv", parse_dates=["Date"], low_memory=False)
store = pd.read_csv(f"{DATA_DIR}/store.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv", parse_dates=["Date"], low_memory=False)

print(f"Train shape: {train.shape}")
print(f"Store shape: {store.shape}")
print(f"Test shape:  {test.shape}")

print("\n" + "=" * 60)
print("TRAIN - KOLONE I TIPOVI")
print("=" * 60)
print(train.dtypes)
print(f"\nDatumski opseg: {train.Date.min()} -> {train.Date.max()}")
print(f"Broj jedinstvenih prodavnica: {train.Store.nunique()}")

print("\n" + "=" * 60)
print("OSNOVNA STATISTIKA - SALES")
print("=" * 60)
print(train["Sales"].describe())

print("\n" + "=" * 60)
print("MISSING VRIJEDNOSTI")
print("=" * 60)
print("Train missing:")
print(train.isnull().sum())
print("\nStore missing:")
print(store.isnull().sum())

print("\n" + "=" * 60)
print("ZATVORENE PRODAVNICE")
print("=" * 60)
print(f"Open=0 redova: {(train['Open'] == 0).sum()}")
print(f"Sales=0 kad je Open=1: {((train['Open'] == 1) & (train['Sales'] == 0)).sum()}")

print("\n" + "=" * 60)
print("PROMO I PRAZNICI")
print("=" * 60)
print(f"Promo=1: {(train['Promo'] == 1).sum()} redova ({(train['Promo']==1).mean()*100:.1f}%)")
print(f"StateHoliday vrijednosti: {train['StateHoliday'].unique()}")
print(f"SchoolHoliday=1: {(train['SchoolHoliday'] == 1).sum()} redova")

print("\n" + "=" * 60)
print("STORE TYPES")
print("=" * 60)
print(store["StoreType"].value_counts())
print("\nAssortment:")
print(store["Assortment"].value_counts())

print("\n" + "=" * 60)
print("PROSJECNA DNEVNA PRODAJA PO DANU U SEDMICI")
print("=" * 60)
dow = train[train["Open"] == 1].groupby("DayOfWeek")["Sales"].agg(["mean", "median", "count"])
print(dow)

print("\nEDA gotov!")
