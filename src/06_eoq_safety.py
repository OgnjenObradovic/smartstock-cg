"""
EOQ (Economic Order Quantity) + Safety Stock + Reorder Point
================================================================
Klasične formule iz supply chain managementa:

1. EOQ = sqrt(2 * D * S / H)
   D = godišnja potraznja
   S = trosak narudzbe (fiksni, po naruzbi)
   H = trosak drzanja (po jedinici, godisnje)

2. Safety Stock = Z * sigma_LT
   Z = z-score za service level (npr. 95% -> 1.645)
   sigma_LT = std potraznje tokom lead time-a

3. Reorder Point = (D/365 * lead_time) + Safety_Stock

4. Service Level: vjerovatnoca da nece doci do stockout-a
"""
import pandas as pd
import numpy as np
from scipy.stats import norm
import xgboost as xgb
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = "/home/ognjen/Desktop/DataMIning/projekat/data/processed"
MODELS_DIR = "/home/ognjen/Desktop/DataMIning/projekat/models"
OUT_DIR = "/home/ognjen/Desktop/DataMIning/projekat/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

print("Ucitavam podatke i model...")
df = pd.read_parquet(f"{DATA_DIR}/processed_full.parquet")
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(f"{MODELS_DIR}/xgboost_global.json")
with open(f"{MODELS_DIR}/xgboost_features.pkl", "rb") as f:
    XGB_FEATURES = pickle.load(f)

# ============================================================
# 1. EOQ FORMULE - osnovne funkcije
# ============================================================
def eoq(annual_demand, order_cost, holding_cost_per_unit):
    """Klasicna Wilsonova EOQ formula."""
    if holding_cost_per_unit <= 0 or annual_demand <= 0:
        return 0
    return np.sqrt(2 * annual_demand * order_cost / holding_cost_per_unit)

def safety_stock(demand_std_during_lead_time, service_level=0.95):
    """Safety stock = Z * std_potraznje_tokom_lead_time."""
    z = norm.ppf(service_level)
    return z * demand_std_during_lead_time

def reorder_point(avg_daily_demand, lead_time_days, safety_stk):
    """ROP = prosjecna potraznja tokom lead time-a + safety stock."""
    return avg_daily_demand * lead_time_days + safety_stk

def total_inventory_cost(eoq_qty, annual_demand, order_cost, holding_cost):
    """Ukupan godisnji trosak inventara."""
    if eoq_qty <= 0:
        return float("inf")
    return (annual_demand / eoq_qty) * order_cost + (eoq_qty / 2) * holding_cost

# ============================================================
# 2. DEMO: izracunaj EOQ za 5 prodavnica
# ============================================================
print(f"\n{'='*70}")
print("EOQ + SAFETY STOCK ANALIZA")
print(f"{'='*70}")

# Parametri (mogu se kasnije mijenjati u Streamlit aplikaciji)
ORDER_COST = 50.0           # 50€ po narudzbi (fiksni trosak)
UNIT_PRICE = 10.0           # cijena jedinice (proxy: Sales -> jedinice)
HOLDING_RATE = 0.25         # 25% godisnje od cijene jedinice
HOLDING_COST = UNIT_PRICE * HOLDING_RATE  # 2.5€ po jedinici godisnje
LEAD_TIME_DAYS = 7          # 7 dana lead time
SERVICE_LEVEL = 0.95        # 95% service level

print(f"Parametri:")
print(f"  Order cost:    {ORDER_COST}€ po narudzbi")
print(f"  Unit price:    {UNIT_PRICE}€")
print(f"  Holding cost:  {HOLDING_COST}€ po jedinici godisnje ({HOLDING_RATE*100:.0f}%)")
print(f"  Lead time:     {LEAD_TIME_DAYS} dana")
print(f"  Service level: {SERVICE_LEVEL*100:.0f}%")
print()

# Uzmi 5 nasumicnih prodavnica
np.random.seed(42)
demo_stores = np.random.choice(df["Store"].unique(), 5, replace=False)

eoq_results = []
for sid in demo_stores:
    sdf = df[df["Store"] == sid].copy().sort_values("Date")
    
    # Posljednjih 90 dana kao representativni period
    last_90 = sdf.tail(90)
    
    # Sales konvertujemo u "jedinice prodaje" (delimo sa UNIT_PRICE)
    daily_units = last_90["Sales"].values / UNIT_PRICE
    
    avg_daily_demand = daily_units.mean()
    std_daily_demand = daily_units.std()
    annual_demand = avg_daily_demand * 365
    
    # Std tokom lead time-a (sqrt(LT) skaliranje za nezavisne dane)
    sigma_lt = std_daily_demand * np.sqrt(LEAD_TIME_DAYS)
    
    # Izracunaj
    q_eoq = eoq(annual_demand, ORDER_COST, HOLDING_COST)
    ss = safety_stock(sigma_lt, SERVICE_LEVEL)
    rop = reorder_point(avg_daily_demand, LEAD_TIME_DAYS, ss)
    total_cost = total_inventory_cost(q_eoq, annual_demand, ORDER_COST, HOLDING_COST)
    num_orders_per_year = annual_demand / q_eoq if q_eoq > 0 else 0
    
    eoq_results.append({
        "Store": sid,
        "AvgDailyDemand": round(avg_daily_demand, 1),
        "StdDailyDemand": round(std_daily_demand, 1),
        "AnnualDemand": round(annual_demand, 0),
        "EOQ": round(q_eoq, 0),
        "SafetyStock": round(ss, 0),
        "ReorderPoint": round(rop, 0),
        "OrdersPerYear": round(num_orders_per_year, 1),
        "TotalAnnualCost_EUR": round(total_cost, 2),
    })

eoq_df = pd.DataFrame(eoq_results)
print(eoq_df.to_string(index=False))

# ============================================================
# 3. SCENARIO ANALIZA - sta ako podignemo service level?
# ============================================================
print(f"\n{'='*70}")
print("SCENARIO: KAKO SERVICE LEVEL UTICE NA SAFETY STOCK?")
print(f"{'='*70}")
print("(Demonstrirano na Store 266)")

sdf = df[df["Store"] == 266].copy().sort_values("Date").tail(90)
daily_units = sdf["Sales"].values / UNIT_PRICE
std_daily = daily_units.std()
sigma_lt = std_daily * np.sqrt(LEAD_TIME_DAYS)
avg_d = daily_units.mean()

print(f"\n{'Service Level':<15}{'Z-score':<10}{'Safety Stock':<15}{'Reorder Point':<15}")
print("-" * 55)
for sl in [0.80, 0.90, 0.95, 0.98, 0.99, 0.995]:
    ss = safety_stock(sigma_lt, sl)
    rop = reorder_point(avg_d, LEAD_TIME_DAYS, ss)
    z = norm.ppf(sl)
    print(f"{sl*100:>6.1f}%{'':<8}{z:<10.3f}{ss:<15.1f}{rop:<15.1f}")

# ============================================================
# 4. AUTOMATSKA NARUDZBENICA (za sledeci ciklus)
# ============================================================
print(f"\n{'='*70}")
print("AUTOMATSKE NARUDZBENICE (na osnovu trenutnog stanja)")
print(f"{'='*70}")

# Simulacija: trenutno stanje zaliha = random (u realnosti dolazi iz ERP-a)
np.random.seed(7)
orders = []
for r in eoq_results:
    current_stock = np.random.randint(0, int(r["ReorderPoint"] * 1.5))
    needs_order = current_stock <= r["ReorderPoint"]
    order_qty = r["EOQ"] if needs_order else 0
    orders.append({
        "Store": r["Store"],
        "CurrentStock": current_stock,
        "ReorderPoint": r["ReorderPoint"],
        "NeedsOrder": "DA" if needs_order else "NE",
        "OrderQty": order_qty,
        "EstimatedCost_EUR": round(order_qty * UNIT_PRICE + (ORDER_COST if needs_order else 0), 2),
    })

orders_df = pd.DataFrame(orders)
print(orders_df.to_string(index=False))

# ============================================================
# 5. SACUVAJ REZULTATE
# ============================================================
eoq_df.to_csv(f"{OUT_DIR}/eoq_analysis.csv", index=False)
orders_df.to_csv(f"{OUT_DIR}/automatic_orders.csv", index=False)
print(f"\nSacuvano:")
print(f"  {OUT_DIR}/eoq_analysis.csv")
print(f"  {OUT_DIR}/automatic_orders.csv")
print("\nEOQ + Safety Stock gotov!")
