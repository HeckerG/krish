from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="E-commerce Customer Analytics", page_icon="🛍️", layout="wide")

st.title("🛍️ E-commerce Customer Behavior Prediction")
st.caption("Explore customer engagement, study correlations, and estimate yearly spending.")

FEATURES = [
    "Avg. Session Length",
    "Time on App",
    "Time on Website",
    "Length of Membership",
]
TARGET = "Yearly Amount Spent"
EXPECTED = FEATURES + [TARGET]

ALIASES = {
    "Avg. Session Length": ["Avg. Session Length", "Avg..Session.Length", "avg_session_length"],
    "Time on App": ["Time on App", "Time.on.App", "time_on_app"],
    "Time on Website": ["Time on Website", "Time.on.Website", "time_on_website"],
    "Length of Membership": ["Length of Membership", "Length.of.Membership", "length_of_membership"],
    "Yearly Amount Spent": ["Yearly Amount Spent", "Yearly.Amount.Spent", "yearly_amount_spent"],
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common CSV header variants to consistent column names."""
    rename = {}
    for standard, variants in ALIASES.items():
        for col in df.columns:
            if str(col).strip().lower() in {v.lower() for v in variants}:
                rename[col] = standard
                break
    return df.rename(columns=rename)

@st.cache_data
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO
    return normalize_columns(pd.read_csv(BytesIO(file_bytes)))

def prepare_data(raw: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in EXPECTED if c not in raw.columns]
    if missing:
        raise ValueError(
            "The CSV is missing required columns: " + ", ".join(missing) +
            ". Use the Ecommerce Customers dataset linked in README.md."
        )
    df = raw[EXPECTED].copy()
    for col in EXPECTED:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.drop_duplicates()
    if len(df) < 10:
        raise ValueError("At least 10 complete rows are needed for a meaningful demonstration.")
    return df

def model_results(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }
    fitted = {}
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R²": r2_score(y_test, pred),
        })
        fitted[name] = model
    return pd.DataFrame(rows).sort_values("RMSE"), fitted, X_test, y_test

with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload Ecommerce Customers CSV", type=["csv"])
    st.markdown("Expected file: `Ecommerce Customers.csv`")
    st.markdown("If you need the dataset, open the link in the project README.")
    st.divider()
    st.header("About")
    st.write("Target: yearly customer spending")
    st.write("Methods: Pearson correlation and regression")
    st.caption("Predictions are estimates, not guarantees.")

if uploaded is not None:
    raw = load_csv(uploaded.getvalue())
else:
    default_path = Path("data/Ecommerce Customers.csv")
    if default_path.exists():
        raw = normalize_columns(pd.read_csv(default_path))
    else:
        st.info(
            "Upload the dataset using the sidebar, or download it and save it as "
            "`data/Ecommerce Customers.csv`. See README.md for the dataset link."
        )
        st.stop()

try:
    df = prepare_data(raw)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

with st.expander("Data quality checks", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Complete rows used", f"{len(df):,}")
    c2.metric("Duplicate rows removed", f"{len(raw) - len(df):,}")
    c3.metric("Missing values after cleaning", f"{int(df.isna().sum().sum()):,}")
    st.write("The model uses only the four numeric engagement features listed below. Identifying fields such as email and address are excluded.")
    st.dataframe(df.head(10), use_container_width=True)

st.subheader("Dataset overview")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Customers", f"{len(df):,}")
m2.metric("Average yearly spend", f"${df[TARGET].mean():,.2f}")
m3.metric("Median yearly spend", f"${df[TARGET].median():,.2f}")
m4.metric("Spend standard deviation", f"${df[TARGET].std():,.2f}")

tab1, tab2, tab3, tab4 = st.tabs([
    "Exploratory analysis", "Correlation", "Model evaluation", "Predict spending"
])

with tab1:
    st.markdown("### Descriptive statistics")
    st.dataframe(df[EXPECTED].describe().T.rename_axis("Variable"), use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df[TARGET], kde=True, ax=ax)
        ax.set_title("Distribution of yearly spending")
        ax.set_xlabel("Yearly amount spent (dataset currency: USD)")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(x=df[TARGET], ax=ax)
        ax.set_title("Spread of yearly spending")
        ax.set_xlabel("Yearly amount spent (USD)")
        st.pyplot(fig)
        plt.close(fig)

    selected_feature = st.selectbox("Explore spending against a feature", FEATURES)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.regplot(data=df, x=selected_feature, y=TARGET, scatter_kws={"alpha": 0.55}, line_kws={"color": "red"}, ax=ax)
    ax.set_title(f"{selected_feature} vs yearly spending")
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.markdown("### Pearson correlation matrix")
    corr = df[EXPECTED].corr(numeric_only=True, method="pearson")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation between numeric variables")
    st.pyplot(fig)
    plt.close(fig)

    target_corr = corr[TARGET].drop(TARGET).sort_values(key=lambda x: x.abs(), ascending=False)
    st.markdown("### Correlation with yearly spending")
    st.dataframe(target_corr.rename("Pearson correlation (r)").to_frame(), use_container_width=True)
    strongest = target_corr.abs().idxmax()
    st.write(
        f"The strongest linear association in this dataset is **{strongest}** "
        f"(r = {target_corr[strongest]:.3f}). This is an association, not proof that the feature causes spending to change."
    )

with tab3:
    st.markdown("### Hold-out test-set evaluation")
    scores, fitted, X_test, y_test = model_results(df)
    st.write("A fixed random 80/20 split is used. Lower MAE/RMSE is better; R² closer to 1 is better, but R² can be negative.")
    st.dataframe(scores.style.format({"MAE": "${:,.2f}", "RMSE": "${:,.2f}", "R²": "{:.3f}"}), use_container_width=True)

    model_choice = st.selectbox("Model to inspect", ["Linear Regression", "Random Forest", "Mean baseline"])
    selected_model = fitted[model_choice]
    pred = selected_model.predict(X_test)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_test, pred, alpha=0.65)
    lo = min(float(y_test.min()), float(np.min(pred)))
    hi = max(float(y_test.max()), float(np.max(pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("Actual yearly spending (USD)")
    ax.set_ylabel("Predicted yearly spending (USD)")
    ax.set_title(f"Actual vs predicted: {model_choice}")
    st.pyplot(fig)
    plt.close(fig)

    residuals = y_test.to_numpy() - pred
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title("Prediction error distribution (actual − predicted)")
    ax.set_xlabel("Residual (USD)")
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Download model comparison CSV",
        scores.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

with tab4:
    st.markdown("### Estimate yearly spending for a customer profile")
    st.caption("Move the sliders to create a hypothetical profile. Values are restricted to the observed dataset ranges.")
    controls = {}
    cols = st.columns(2)
    for i, feature in enumerate(FEATURES):
        min_v = float(df[feature].min())
        max_v = float(df[feature].max())
        mean_v = float(df[feature].mean())
        step = max((max_v - min_v) / 100, 0.01)
        with cols[i % 2]:
            controls[feature] = st.slider(
                feature,
                min_value=min_v,
                max_value=max_v,
                value=min(max(mean_v, min_v), max_v),
                step=step,
            )
    chosen = st.selectbox("Prediction model", ["Linear Regression", "Random Forest"], key="prediction_model")
    _, fitted, _, _ = model_results(df)
    input_df = pd.DataFrame([controls], columns=FEATURES)
    estimate = float(fitted[chosen].predict(input_df)[0])
    st.metric("Estimated yearly spending", f"${estimate:,.2f}")
    st.write("This is a model estimate for an example profile, not a guarantee of future customer behavior.")

st.divider()
st.caption("Educational project. Interpret model performance in context; correlation does not imply causation.")
