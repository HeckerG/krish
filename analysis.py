"""Standalone exploratory analysis for the E-commerce Customer Prediction project."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Ecommerce Customers.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

FEATURES = ["Avg. Session Length", "Time on App", "Time on Website", "Length of Membership"]
TARGET = "Yearly Amount Spent"
ALIASES = {
    "Avg. Session Length": ["Avg. Session Length", "Avg..Session.Length"],
    "Time on App": ["Time on App", "Time.on.App"],
    "Time on Website": ["Time on Website", "Time.on.Website"],
    "Length of Membership": ["Length of Membership", "Length.of.Membership"],
    "Yearly Amount Spent": ["Yearly Amount Spent", "Yearly.Amount.Spent"],
}

if not DATA_PATH.exists():
    raise SystemExit(f"Dataset not found: {DATA_PATH}\nDownload it and save it to data/Ecommerce Customers.csv.")

df = pd.read_csv(DATA_PATH)
rename = {}
for standard, variants in ALIASES.items():
    for col in df.columns:
        if str(col).strip().lower() in {v.lower() for v in variants}:
            rename[col] = standard
            break
df = df.rename(columns=rename)
required = FEATURES + [TARGET]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing required columns: {missing}")

df = df[required].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates()
if len(df) < 10:
    raise SystemExit("Need at least 10 complete rows.")

print("\nDATASET SHAPE:", df.shape)
print("\nDESCRIPTIVE STATISTICS\n", df.describe().T.round(3))
print("\nMISSING VALUES\n", df.isna().sum())
print("\nPEARSON CORRELATION WITH TARGET\n", df.corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(ascending=False))

df.describe().T.to_csv(OUT / "descriptive_statistics.csv")
df.corr(numeric_only=True).to_csv(OUT / "correlation_matrix.csv")

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Pearson Correlation Matrix")
fig.tight_layout()
fig.savefig(OUT / "correlation_heatmap.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(df[TARGET], kde=True, ax=ax)
ax.set_title("Distribution of Yearly Amount Spent")
ax.set_xlabel("Yearly Amount Spent (USD)")
fig.tight_layout()
fig.savefig(OUT / "spending_distribution.png", dpi=160)
plt.close(fig)

X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES], df[TARGET], test_size=0.20, random_state=42
)
models = {
    "Mean baseline": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
}
rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
    })
    if name == "Linear Regression":
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_test, pred, alpha=0.65)
        low = min(float(y_test.min()), float(np.min(pred)))
        high = max(float(y_test.max()), float(np.max(pred)))
        ax.plot([low, high], [low, high], linestyle="--")
        ax.set_xlabel("Actual spending (USD)")
        ax.set_ylabel("Predicted spending (USD)")
        ax.set_title("Linear Regression: Actual vs Predicted")
        fig.tight_layout()
        fig.savefig(OUT / "actual_vs_predicted.png", dpi=160)
        plt.close(fig)

scores = pd.DataFrame(rows).sort_values("RMSE")
scores.to_csv(OUT / "model_comparison.csv", index=False)
print("\nMODEL COMPARISON\n", scores.round(3).to_string(index=False))
print(f"\nSaved outputs to: {OUT}")
