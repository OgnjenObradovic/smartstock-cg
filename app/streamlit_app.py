import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle, json, os
from scipy.stats import norm

st.set_page_config(page_title="SmartStock CG")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = f"{BASE}/data/processed/processed_full.parquet"
MODELS_DIR = f"{BASE}/models"


@st.cache_data
def ucitaj_podatke():
    return pd.read_parquet(DATA_PATH)


@st.cache_resource
def ucitaj_model():
    with open(f"{MODELS_DIR}/xgboost_features.pkl", "rb") as f:
        feats = pickle.load(f)
    bst = xgb.Booster()
    bst.load_model(f"{MODELS_DIR}/xgboost_global.json")
    cfg = json.loads(bst.save_config())
    bs = cfg["learner"]["learner_model_param"]["base_score"]
    if bs.startswith("["):
        cfg["learner"]["learner_model_param"]["base_score"] = bs.strip("[]")
        bst.load_config(json.dumps(cfg))
    return bst, feats


df = ucitaj_podatke()
model, FEATURES = ucitaj_model()


def predvidi(data):
    X = data[FEATURES].copy()
    for c in X.select_dtypes(include="bool").columns:
        X[c] = X[c].astype(int)
    return np.maximum(model.predict(xgb.DMatrix(X.values, feature_names=FEATURES)), 0)


st.sidebar.title("SmartStock CG")
stranica = st.sidebar.radio("Stranice", [
    "Pregled",
    "Forecast",
    "EOQ",
    "Anomalije",
    "Narudzbenice",
    "O projektu",
])


if stranica == "Pregled":
    st.title("Pregled podataka")

    st.write(f"Broj prodavnica: {df['Store'].nunique()}")
    st.write(f"Broj transakcija: {len(df)}")
    st.write(f"Period: {df['Date'].min().date()} - {df['Date'].max().date()}")
    st.write(f"Prosjecna prodaja: {df['Sales'].mean():.0f} EUR")

    st.subheader("Trend ukupne prodaje")
    daily = df.groupby("Date")["Sales"].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily.index, daily.values)
    ax.set_xlabel("Datum")
    ax.set_ylabel("Prodaja (EUR)")
    st.pyplot(fig)

    st.subheader("Prosjecna prodaja po danu u sedmici")
    dow = df.groupby("DayOfWeek")["Sales"].mean()
    dani = ["Pon","Uto","Sri","Cet","Pet","Sub","Ned"]
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.bar(dani, dow.values)
    ax2.set_ylabel("Prosjecna prodaja")
    st.pyplot(fig2)

    st.subheader("Efekat promocije")
    promo = df.groupby("Promo")["Sales"].mean()
    st.write(f"Bez promocije: {promo[0]:.0f} EUR")
    st.write(f"Sa promocijom: {promo[1]:.0f} EUR")
    st.write(f"Razlika: +{(promo[1]/promo[0]-1)*100:.1f}%")


elif stranica == "Forecast":
    st.title("Forecast prodaje")

    sid = st.selectbox("Prodavnica", sorted(df["Store"].unique()))
    dana = st.slider("Broj dana za prikaz", 14, 90, 30)

    sdf = df[df["Store"] == sid].sort_values("Date").copy()
    sdf["yhat"] = predvidi(sdf)
    posljednji = sdf.tail(dana)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(posljednji["Date"], posljednji["Sales"], label="Stvarno", marker="o", markersize=3)
    ax.plot(posljednji["Date"], posljednji["yhat"], label="Predikcija", linestyle="--")
    ax.set_xlabel("Datum")
    ax.set_ylabel("Prodaja (EUR)")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    mae = np.mean(np.abs(posljednji["Sales"] - posljednji["yhat"]))
    smape = np.mean(2*np.abs(posljednji["Sales"]-posljednji["yhat"]) /
                    (np.abs(posljednji["Sales"]) + np.abs(posljednji["yhat"]) + 1e-9))*100
    st.write(f"MAE: {mae:.0f} EUR")
    st.write(f"SMAPE: {smape:.2f}%")


elif stranica == "EOQ":
    st.title("Optimalna kolicina narudzbe")

    sid = st.selectbox("Prodavnica", sorted(df["Store"].unique()))
    trosak_narudzbe = st.number_input("Trosak narudzbe (EUR)", value=50.0)
    cijena = st.number_input("Cijena jedinice (EUR)", value=10.0)
    lt = st.number_input("Lead time (dana)", value=7)
    sl = st.slider("Service level (%)", 80, 99, 95) / 100

    sdf = df[df["Store"]==sid].tail(90)
    dnevna = sdf["Sales"].values / cijena
    avg_d = dnevna.mean()
    std_d = dnevna.std()
    godisnja = avg_d * 365
    holding = cijena * 0.25

    eoq = np.sqrt(2 * godisnja * trosak_narudzbe / holding)
    sigma_lt = std_d * np.sqrt(lt)
    safety = norm.ppf(sl) * sigma_lt
    rop = avg_d * lt + safety
    ukupni = (godisnja/eoq)*trosak_narudzbe + (eoq/2)*holding

    st.write(f"Prosjecna dnevna potraznja: {avg_d:.0f} jedinica")
    st.write(f"Godisnja potraznja: {godisnja:.0f} jedinica")
    st.write("")
    st.write(f"**EOQ (optimalna kolicina): {eoq:.0f} jedinica**")
    st.write(f"**Safety stock: {safety:.0f} jedinica**")
    st.write(f"**Reorder point: {rop:.0f} jedinica**")
    st.write(f"**Ukupni godisnji trosak: {ukupni:.2f} EUR**")
    st.write("")
    st.write("Formule:")
    st.code("EOQ = sqrt(2 * D * S / H)\nSS = Z * std * sqrt(LT)\nROP = avg_d * LT + SS")


elif stranica == "Anomalije":
    st.title("Detekcija anomalija")

    sid = st.selectbox("Prodavnica", sorted(df["Store"].unique()))
    prag = st.slider("Prag z-score", 2.0, 5.0, 3.0, 0.1)

    sdf = df[df["Store"]==sid].sort_values("Date").copy()
    rm = sdf["Sales"].shift(1).rolling(14).mean()
    rs = sdf["Sales"].shift(1).rolling(14).std()
    sdf["z"] = (sdf["Sales"] - rm) / (rs + 1e-9)
    sdf["anomalija"] = (sdf["z"].abs() > prag).astype(int)

    nadjeno = sdf[sdf["anomalija"]==1]
    st.write(f"Detektovano anomalija: {len(nadjeno)} od {len(sdf)} dana")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sdf["Date"], sdf["Sales"], label="Prodaja")
    ax.scatter(nadjeno["Date"], nadjeno["Sales"], color="red", label="Anomalije", zorder=5)
    ax.set_xlabel("Datum")
    ax.set_ylabel("Prodaja (EUR)")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

    if len(nadjeno) > 0:
        st.subheader("Detalji")
        def razlog(r):
            r_list = []
            if r["Promo"]==1: r_list.append("promocija")
            if r["IsHoliday"]==1: r_list.append("praznik")
            if r["SchoolHoliday"]==1: r_list.append("raspust")
            return ", ".join(r_list) if r_list else "nepoznato"
        nadjeno = nadjeno.copy()
        nadjeno["razlog"] = nadjeno.apply(razlog, axis=1)
        st.dataframe(nadjeno[["Date","Sales","z","razlog"]].head(15), hide_index=True)


elif stranica == "Narudzbenice":
    st.title("Generisanje narudzbenica")

    trosak_n = st.number_input("Trosak narudzbe (EUR)", value=50.0)
    cijena_j = st.number_input("Cijena jedinice (EUR)", value=10.0)
    lt = st.number_input("Lead time (dana)", value=7)
    broj = st.slider("Broj prodavnica", 5, 30, 10)

    np.random.seed(42)
    prodavnice = np.random.choice(df["Store"].unique(), broj, replace=False)
    holding = cijena_j * 0.25

    redovi = []
    for sid in prodavnice:
        s = df[df["Store"]==sid].tail(90)
        dnevna = s["Sales"].values / cijena_j
        avg_d, std_d = dnevna.mean(), dnevna.std()
        godisnja = avg_d * 365
        eoq = np.sqrt(2*godisnja*trosak_n/holding)
        ss = norm.ppf(0.95) * std_d * np.sqrt(lt)
        rop = avg_d*lt + ss
        trenutno = np.random.randint(0, int(rop*1.5))
        treba = trenutno <= rop
        redovi.append({
            "Prodavnica": sid,
            "Stanje": trenutno,
            "ROP": round(rop),
            "Treba narudzbu": "DA" if treba else "NE",
            "Kolicina": round(eoq) if treba else 0,
            "Trosak": round(eoq*cijena_j + trosak_n, 2) if treba else 0,
        })

    tabela = pd.DataFrame(redovi)
    st.dataframe(tabela, hide_index=True)
    st.write(f"Ukupno: {tabela['Trosak'].sum():.2f} EUR")


elif stranica == "O projektu":
    st.title("O projektu")
    st.write("SmartStock CG - sistem za predikciju potraznje i optimizaciju zaliha.")
    st.write("")
    st.write("Predmet: Data Mining, FIST UDG")
    st.write("")
    st.write("Koristene tehnologije:")
    st.write("- Python, pandas, scikit-learn")
    st.write("- XGBoost (forecasting)")
    st.write("- Prophet (baseline)")
    st.write("- Streamlit (web app)")
    st.write("- matplotlib (vizualizacija)")
    st.write("")
    st.write("Dataset: Rossmann Store Sales (Kaggle)")
    st.write("Broj redova: 1.017.000+")
    st.write("Broj prodavnica: 1.115")
    st.write("")
    st.write("Implementirano:")
    st.write("1. Predikcija prodaje (XGBoost)")
    st.write("2. EOQ + Safety Stock formule")
    st.write("3. Detekcija anomalija (z-score)")
    st.write("4. Generisanje narudzbenica")
    st.write("")
    st.write("Rezultati na test setu (zadnja 42 dana):")
    st.write("- SMAPE: 8.24%")
    st.write("- MAE: 549 EUR")
