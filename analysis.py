"""Reproducible analysis for UCI Online Retail customer spending case study."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Online Retail.xlsx"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
CUTOFF = pd.Timestamp("2011-09-01")
FEATURES = ["Historical net revenue", "Historical order count", "Historical units purchased",
            "Historical product variety", "Average order value", "Days since last purchase"]
TARGET = "Future net revenue"

def load_transactions(path):
    raw = pd.read_excel(path) if str(path).lower().endswith((".xlsx", ".xls")) else pd.read_csv(path, encoding_errors="replace")
    required = ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = raw.copy()
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    for c in ["Quantity", "UnitPrice", "CustomerID"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"])
    df = df[(df["Quantity"] != 0) & (df["UnitPrice"] >= 0)].copy()
    df["CustomerID"] = df["CustomerID"].astype("int64").astype(str)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["IsCancellation"] = df["InvoiceNo"].str.upper().str.startswith("C")
    df["IsSaleLine"] = (~df["IsCancellation"]) & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)
    return raw, df

def build_customer_table(df, cutoff=CUTOFF):
    hist, future = df[df["InvoiceDate"] < cutoff].copy(), df[df["InvoiceDate"] >= cutoff].copy()
    if hist.empty or future.empty:
        raise ValueError("Cutoff must leave records in both historical and future periods.")
    ids = hist["CustomerID"].drop_duplicates()
    sales = hist[hist["IsSaleLine"]]
    features = pd.DataFrame(index=ids)
    features.index.name = "CustomerID"
    features["Historical net revenue"] = hist.groupby("CustomerID")["Revenue"].sum().reindex(ids).fillna(0)
    features["Historical order count"] = sales.groupby("CustomerID")["InvoiceNo"].nunique().reindex(ids).fillna(0)
    features["Historical units purchased"] = sales.groupby("CustomerID")["Quantity"].sum().reindex(ids).fillna(0)
    features["Historical product variety"] = sales.groupby("CustomerID")["StockCode"].nunique().reindex(ids).fillna(0)
    order_revenue = sales.groupby(["CustomerID", "InvoiceNo"])["Revenue"].sum()
    features["Average order value"] = order_revenue.groupby("CustomerID").mean().reindex(ids).fillna(0)
    last = sales.groupby("CustomerID")["InvoiceDate"].max().reindex(ids)
    features["Days since last purchase"] = (cutoff - last).dt.total_seconds().div(86400).fillna((cutoff - hist["InvoiceDate"].min()).days)
    features[TARGET] = future.groupby("CustomerID")["Revenue"].sum().reindex(ids).fillna(0)
    return features.replace([np.inf, -np.inf], np.nan).fillna(0).reset_index(), hist, future

def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"Place the UCI Excel file at: {DATA_PATH}")
    raw, tx = load_transactions(DATA_PATH)
    data, hist, future = build_customer_table(tx)
    print("Raw shape:", raw.shape)
    print("Transaction rows used:", len(tx))
    print("Rows missing CustomerID:", int(raw["CustomerID"].isna().sum()))
    print("Cancellation invoices retained:", int(tx["IsCancellation"].sum()))
    print("Negative-quantity lines retained:", int((tx["Quantity"] < 0).sum()))
    print("Historical period:", hist["InvoiceDate"].min(), "to", CUTOFF - pd.Timedelta(days=1))
    print("Future period:", CUTOFF, "to", future["InvoiceDate"].max())
    print("Customer-level table shape:", data.shape)
    print("\\nDESCRIPTIVE STATISTICS\\n", data[FEATURES + [TARGET]].describe().T.round(3))
    corr = data[FEATURES + [TARGET]].corr(method="pearson")
    print("\\nCORRELATION WITH FUTURE NET REVENUE\\n", corr[TARGET].drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False).round(4))
    data.to_csv(OUT / "customer_spending_features.csv", index=False)
    data[FEATURES + [TARGET]].describe().T.to_csv(OUT / "descriptive_statistics.csv")
    corr.to_csv(OUT / "correlation_matrix.csv")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Pearson Correlation Matrix")
    fig.tight_layout(); fig.savefig(OUT / "correlation_heatmap.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data[TARGET], bins=40, edgecolor="black", alpha=0.75)
    ax.set_title("Distribution of Future Net Revenue")
    fig.tight_layout(); fig.savefig(OUT / "future_spending_distribution.png", dpi=160); plt.close(fig)
    X, y = data[FEATURES].astype(float), data[TARGET].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {"Mean baseline": DummyRegressor(strategy="mean"),
              "Linear Regression": LinearRegression(),
              "Random Forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=-1)}
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train); pred = model.predict(X_test)
        rows.append({"Model": name, "MAE": mean_absolute_error(y_test, pred),
                     "Median AE": median_absolute_error(y_test, pred),
                     "RMSE": np.sqrt(mean_squared_error(y_test, pred)), "R2": r2_score(y_test, pred)})
        if name == "Linear Regression":
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(y_test, pred, alpha=0.6)
            lo, hi = min(float(y_test.min()), float(pred.min())), max(float(y_test.max()), float(pred.max()))
            ax.plot([lo, hi], [lo, hi], linestyle="--")
            ax.set_xscale("symlog", linthresh=100); ax.set_yscale("symlog", linthresh=100)
            ax.set_xlabel("Actual future net revenue (symlog scale)")
            ax.set_ylabel("Predicted future net revenue (symlog scale)")
            ax.set_title("Linear Regression: Actual vs Predicted")
            fig.tight_layout(); fig.savefig(OUT / "actual_vs_predicted.png", dpi=160); plt.close(fig)
    scores = pd.DataFrame(rows).sort_values("RMSE")
    scores.to_csv(OUT / "model_comparison.csv", index=False)
    print("\\nMODEL COMPARISON\\n", scores.round(3).to_string(index=False))
    print("\\nOutputs saved to:", OUT)

if __name__ == "__main__":
    main()
