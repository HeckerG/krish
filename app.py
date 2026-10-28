from pathlib import Path
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Online Retail Customer Spending", page_icon="🛍️", layout="wide")
st.title("E-commerce Customer Spending Prediction")
st.caption("UCI Online Retail transaction analysis | Historical behavior → later-period net spending")

FEATURES = [
    "Historical net revenue",
    "Historical order count",
    "Historical units purchased",
    "Historical product variety",
    "Average order value",
    "Days since last purchase",
]
TARGET = "Future net revenue"

@st.cache_data(show_spinner="Reading transaction file...")
def read_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    bio = BytesIO(file_bytes)
    if filename.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(bio)
    return pd.read_csv(bio, encoding_errors="replace")

def clean_transactions(raw: pd.DataFrame):
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    required = ["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required UCI Online Retail columns: " + ", ".join(missing))
    initial_rows = len(df)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    for c in ["Quantity", "UnitPrice", "CustomerID"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["CustomerID", "InvoiceDate", "Quantity", "UnitPrice"])
    df = df[df["Quantity"] != 0]
    df = df[df["UnitPrice"] >= 0]
    df["CustomerID"] = df["CustomerID"].astype("int64").astype(str)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    # UCI cancellations typically use invoice numbers beginning with C. Negative quantities also capture returns.
    df["IsCancellation"] = df["InvoiceNo"].str.upper().str.startswith("C")
    df["IsSaleLine"] = (~df["IsCancellation"]) & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    info = {
        "input_rows": initial_rows,
        "rows_after_required-field_cleaning": len(df),
        "rows_without_customer_id": int(raw["CustomerID"].isna().sum()),
        "duplicate_rows_in_input": int(raw.duplicated().sum()),
        "cancellation_lines_retained_for_net_revenue": int(df["IsCancellation"].sum()),
        "negative_quantity_lines_retained_for_net_revenue": int((df["Quantity"] < 0).sum()),
        "date_min": df["InvoiceDate"].min(),
        "date_max": df["InvoiceDate"].max(),
    }
    return df, info

def make_customer_table(df: pd.DataFrame, cutoff: pd.Timestamp):
    hist = df[df["InvoiceDate"] < cutoff].copy()
    future = df[df["InvoiceDate"] >= cutoff].copy()
    if hist.empty or future.empty:
        raise ValueError("The cutoff leaves no transactions in either the historical or future period.")
    # All customers with identified activity before cutoff are eligible, including those with zero later spend.
    customer_ids = hist["CustomerID"].drop_duplicates()
    hist["SaleRevenue"] = np.where(hist["IsSaleLine"], hist["Revenue"], 0.0)
    sales = hist[hist["IsSaleLine"]].copy()
    order_counts = sales.groupby("CustomerID")["InvoiceNo"].nunique()
    units = sales.groupby("CustomerID")["Quantity"].sum()
    variety = sales.groupby("CustomerID")["StockCode"].nunique()
    net_revenue = hist.groupby("CustomerID")["Revenue"].sum()
    last_date = sales.groupby("CustomerID")["InvoiceDate"].max()
    order_revenue = sales.groupby(["CustomerID", "InvoiceNo"])["Revenue"].sum()
    avg_order = order_revenue.groupby("CustomerID").mean()
    features = pd.DataFrame(index=customer_ids)
    features.index.name = "CustomerID"
    features["Historical net revenue"] = net_revenue.reindex(features.index).fillna(0)
    features["Historical order count"] = order_counts.reindex(features.index).fillna(0)
    features["Historical units purchased"] = units.reindex(features.index).fillna(0)
    features["Historical product variety"] = variety.reindex(features.index).fillna(0)
    features["Average order value"] = avg_order.reindex(features.index).fillna(0)
    features["Days since last purchase"] = (cutoff - last_date.reindex(features.index)).dt.total_seconds().div(86400).fillna((cutoff - hist["InvoiceDate"].min()).days)
    future_net = future.groupby("CustomerID")["Revenue"].sum()
    features[TARGET] = future_net.reindex(features.index).fillna(0)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0).reset_index()
    return features, hist, future

def evaluate_models(data: pd.DataFrame):
    X = data[FEATURES].astype(float)
    y = data[TARGET].astype(float)
    if len(data) < 20:
        raise ValueError("At least 20 customer records are required to evaluate models.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=-1),
    }
    fitted, rows, predictions = {}, [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        fitted[name] = model
        predictions[name] = pred
        rows.append({"Model": name, "MAE": mean_absolute_error(y_test, pred),
                     "Median AE": median_absolute_error(y_test, pred),
                     "RMSE": np.sqrt(mean_squared_error(y_test, pred)), "R²": r2_score(y_test, pred)})
    return pd.DataFrame(rows).sort_values("RMSE"), fitted, X_test, y_test, predictions

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload UCI Online Retail file", type=["xlsx", "xls", "csv"])
    default_path = Path("data/Online Retail.xlsx")
    st.caption("Expected fields include InvoiceNo, Quantity, InvoiceDate, UnitPrice and CustomerID.")
    st.divider()
    st.header("Method")
    cutoff_date = st.date_input("Historical/future cutoff", value=pd.Timestamp("2011-09-01").date(),
                                min_value=pd.Timestamp("2011-01-01").date(),
                                max_value=pd.Timestamp("2011-12-01").date())
    st.caption("Transactions before cutoff build features. Transactions on/after cutoff form the target.")

try:
    if uploaded is not None:
        raw = read_file(uploaded.getvalue(), uploaded.name)
    elif default_path.exists():
        raw = read_file(default_path.read_bytes(), default_path.name)
    else:
        st.info("Upload the official UCI Online Retail Excel file in the sidebar to begin.")
        st.markdown("Dataset source: [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)")
        st.stop()
    transactions, quality = clean_transactions(raw)
    data, hist, future = make_customer_table(transactions, pd.Timestamp(cutoff_date))
except Exception as exc:
    st.error(f"Could not prepare the dataset: {exc}")
    st.stop()

st.markdown("### Study design")
st.write(f"Historical period: **{hist['InvoiceDate'].min().date()} to {(pd.Timestamp(cutoff_date) - pd.Timedelta(days=1)).date()}**. "
         f"Prediction period: **{pd.Timestamp(cutoff_date).date()} to {future['InvoiceDate'].max().date()}**. "
         "Revenue is Quantity × UnitPrice. Negative quantities and cancellation invoices remain in net revenue, while order-behavior features count positive non-cancelled sales only.")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Raw transaction rows", f"{quality['input_rows']:,}")
m2.metric("Rows with identified customers", f"{len(transactions):,}")
m3.metric("Customers modelled", f"{len(data):,}")
m4.metric("Mean future net spend", f"{data[TARGET].mean():,.2f}")

with st.expander("Data quality and cleaning details"):
    st.json({k: (str(v) if isinstance(v, (pd.Timestamp,)) else v) for k, v in quality.items()})
    st.write("Rows without a customer ID are excluded from customer-level modelling. Exact duplicate rows are reported but not automatically removed because identical invoice lines can be legitimate records.")
    st.dataframe(data.head(10), use_container_width=True)

tabs = st.tabs(["Overview", "Statistics & correlation", "Model evaluation", "Predict a customer"])

with tabs[0]:
    st.subheader("Future net spending distribution")
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(data[TARGET], bins=40, edgecolor="black", alpha=0.75)
        ax.set_xlabel("Future net revenue (dataset currency)")
        ax.set_title("Customer-level target distribution")
        st.pyplot(fig); plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(x=data[TARGET], ax=ax)
        ax.set_xlabel("Future net revenue (dataset currency)")
        ax.set_title("Spread and outliers")
        st.pyplot(fig); plt.close(fig)
    st.caption("The dataset's currency is not explicitly represented by a currency column. Revenue is reported in the source's monetary units; do not label it USD.")

with tabs[1]:
    st.subheader("Descriptive statistics")
    st.dataframe(data[FEATURES + [TARGET]].describe().T, use_container_width=True)
    corr = data[FEATURES + [TARGET]].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Pearson correlation matrix")
    st.pyplot(fig); plt.close(fig)
    target_corr = corr[TARGET].drop(TARGET).sort_values(key=lambda s: s.abs(), ascending=False)
    st.markdown("**Correlation with future net spending**")
    st.dataframe(target_corr.rename("Pearson r").to_frame(), use_container_width=True)
    st.write("Correlation describes linear association, not causation. Skewed spending and extreme values may affect Pearson correlations.")

with tabs[2]:
    st.subheader("Hold-out customer evaluation")
    scores, fitted, X_test, y_test, preds = evaluate_models(data)
    st.write("Features are built only from the historical period. The hold-out test split is across customers; each customer's target comes from the later period.")
    st.dataframe(scores.style.format({"MAE": "{:,.2f}", "Median AE": "{:,.2f}", "RMSE": "{:,.2f}", "R²": "{:.3f}"}), use_container_width=True)
    selected = st.selectbox("Inspect model", list(fitted.keys()))
    pred = preds[selected]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, pred, alpha=0.6)
    lo = min(float(y_test.min()), float(np.min(pred))); hi = max(float(y_test.max()), float(np.max(pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xscale("symlog", linthresh=100)
    ax.set_yscale("symlog", linthresh=100)
    ax.set_xlabel("Actual future net revenue (symlog scale)")
    ax.set_ylabel("Predicted future net revenue (symlog scale)")
    ax.set_title(f"Actual vs predicted: {selected}")
    st.pyplot(fig); plt.close(fig)

    residuals = np.asarray(y_test) - np.asarray(pred)
    st.markdown("#### Prediction error diagnostics")
    d1, d2, d3 = st.columns(3)
    d1.metric("Median absolute error", f"{median_absolute_error(y_test, pred):,.2f}")
    d2.metric("Largest absolute error", f"{np.max(np.abs(residuals)):,.2f}")
    p99 = float(pd.Series(y_test).quantile(0.99))
    high_spend_count = int((pd.Series(y_test) >= p99).sum())
    d3.metric("Test customers at/above 99th percentile", f"{high_spend_count:,}")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(y_test, residuals, alpha=0.55)
    ax.axhline(0, linestyle="--")
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel("Actual future net revenue (symlog scale)")
    ax.set_ylabel("Residual: actual − predicted")
    ax.set_title("Residuals by actual future spending")
    st.pyplot(fig); plt.close(fig)
    st.caption("Symlog axes keep ordinary customers visible while still showing extreme values. MAE describes typical absolute error; RMSE penalizes very large errors more strongly. The 99th-percentile count is a diagnostic, not a reason to discard high-value customers.")

    st.download_button("Download model comparison CSV", scores.to_csv(index=False).encode("utf-8"),
                       file_name="model_comparison.csv", mime="text/csv")
    st.download_button("Download customer-level modelling table CSV", data.to_csv(index=False).encode("utf-8"),
                       file_name="customer_spending_features.csv", mime="text/csv")

with tabs[3]:
    st.subheader("Estimate later-period spending")
    st.caption("Adjust a hypothetical customer's historical behavior. Predictions may be negative because the target is net revenue after returns/cancellations.")
    controls = {}
    cols = st.columns(2)
    for i, feature in enumerate(FEATURES):
        low = float(data[feature].quantile(0.01)); high = float(data[feature].quantile(0.99))
        if not np.isfinite(low): low = 0.0
        if not np.isfinite(high) or high <= low: high = low + 1.0
        val = float(data[feature].median())
        with cols[i % 2]:
            controls[feature] = st.number_input(feature, min_value=low, max_value=high, value=min(max(val, low), high))
    chosen = st.selectbox("Prediction model", ["Linear Regression", "Random Forest"], key="predict_model")
    _, fitted, _, _, _ = evaluate_models(data)
    estimate = float(fitted[chosen].predict(pd.DataFrame([controls], columns=FEATURES))[0])
    st.metric("Estimated future net revenue", f"{estimate:,.2f}")
    st.caption("This is a statistical estimate for a hypothetical profile, not a guarantee of customer behavior.")

st.divider()
st.caption("Data source: UCI Machine Learning Repository, Online Retail dataset (CC BY 4.0). Educational case study; historical data from 2010–2011.")
