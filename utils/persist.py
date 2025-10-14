import os, pandas as pd

BASE = "./data"; os.makedirs(BASE, exist_ok=True)
PATHS = {
    "orders_df": f"{BASE}/orders_df.csv",
    "routes_df": f"{BASE}/routes_df.csv",
    "dispatch_log": f"{BASE}/dispatch_log.csv",
}

def save_df(name, df):
    p = PATHS[name]; df.to_csv(p, index=False)

def load_df(name):
    p = PATHS[name]
    return pd.read_csv(p) if os.path.exists(p) else None
