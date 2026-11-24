from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "fraud_detection_xgboost_model.pkl"
DATA_PATH = ROOT / "Fraud_Analysis_Dataset.csv"
RAW_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
BALANCE_COLUMNS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_model() -> dict[str, Any]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH.name}")
    artifact = joblib.load(MODEL_PATH)
    if not isinstance(artifact, dict) or not {
        "model",
        "feature_columns",
        "required_raw_columns",
    }.issubset(artifact):
        raise ValueError("The model file does not contain the expected feature schema.")
    return artifact


@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset file not found: {DATA_PATH.name}")
    data = pd.read_csv(DATA_PATH)
    required = set(RAW_COLUMNS + ["isFraud"])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return data


def prepare_features(data: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    required = artifact["required_raw_columns"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    features = data.loc[:, required].copy()
    features["type"] = features["type"].astype("string").str.strip().str.upper()
    invalid_types = sorted(set(features["type"].dropna()) - set(TRANSACTION_TYPES))
    if invalid_types or features["type"].isna().any():
        raise ValueError(f"Unsupported transaction type(s): {invalid_types}")

    numeric_columns = ["step", *BALANCE_COLUMNS]
    for column in numeric_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    values = features[numeric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Step, amount, and balances must contain finite numbers.")
    if (features[numeric_columns] < 0).any().any():
        raise ValueError("Step, amount, and balances cannot be negative.")
    if (features["step"] < 1).any() or (features["step"] % 1 != 0).any():
        raise ValueError("Step must be a positive whole number.")

    features = pd.get_dummies(features, columns=["type"], drop_first=True, dtype=int)
    features[BALANCE_COLUMNS] = np.log1p(features[BALANCE_COLUMNS])
    return features.reindex(columns=artifact["feature_columns"], fill_value=0)


def score(data: pd.DataFrame, artifact: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    features = prepare_features(data, artifact)
    predictions = artifact["model"].predict(features).astype(int)
    probabilities = artifact["model"].predict_proba(features)[:, 1]
    return predictions, probabilities


def dashboard(data: pd.DataFrame) -> None:
    st.title("Fraud Detection Dashboard")
    st.caption("Local transaction-risk analysis using the trained XGBoost model.")
    count = len(data)
    fraud_count = int(data["isFraud"].sum())
    metrics = st.columns(4)
    metrics[0].metric("Transactions", f"{count:,}")
    metrics[1].metric("Fraud cases", f"{fraud_count:,}")
    metrics[2].metric("Fraud rate", f"{fraud_count / count:.2%}" if count else "0%")
    metrics[3].metric("Total amount", f"{data['amount'].sum():,.0f}")

    st.subheader("Transactions by type")
    summary = (
        data.assign(label=data["isFraud"].map({0: "Legitimate", 1: "Fraud"}))
        .groupby(["type", "label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Legitimate", "Fraud"], fill_value=0)
    )
    st.bar_chart(summary, color=["#2e8b57", "#d64545"])
    st.subheader("Dataset sample")
    st.dataframe(data.head(20), width="stretch", hide_index=True)


def single_transaction(artifact: dict[str, Any]) -> None:
    st.title("Score a transaction")
    st.caption("Enter the raw transaction features used by the trained model.")
    with st.form("transaction"):
        left, right = st.columns(2)
        with left:
            step = st.number_input("Step (hour)", min_value=1, value=1, step=1)
            kind = st.selectbox("Transaction type", TRANSACTION_TYPES)
            amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0)
            old_org = st.number_input(
                "Origin balance before", min_value=0.0, value=1000.0, step=100.0
            )
        with right:
            new_org = st.number_input(
                "Origin balance after", min_value=0.0, value=0.0, step=100.0
            )
            old_dest = st.number_input(
                "Destination balance before", min_value=0.0, value=0.0, step=100.0
            )
            new_dest = st.number_input(
                "Destination balance after", min_value=0.0, value=1000.0, step=100.0
            )
        submitted = st.form_submit_button("Analyze", type="primary")

    if submitted:
        transaction = pd.DataFrame(
            [
                {
                    "step": step,
                    "type": kind,
                    "amount": amount,
                    "oldbalanceOrg": old_org,
                    "newbalanceOrig": new_org,
                    "oldbalanceDest": old_dest,
                    "newbalanceDest": new_dest,
                }
            ]
        )
        try:
            predictions, probabilities = score(transaction, artifact)
        except (KeyError, TypeError, ValueError) as exc:
            st.error(f"Could not analyze the transaction: {exc}")
            return
        probability = float(probabilities[0])
        if predictions[0] == 1:
            st.error("Model result: **Potentially fraudulent**")
        else:
            st.success("Model result: **Likely legitimate**")
        st.metric("Model fraud probability", f"{probability:.2%}")
        st.progress(float(np.clip(probability, 0.0, 1.0)))
        st.caption("For educational review only; do not use as the sole basis for financial decisions.")


def batch_screening(artifact: dict[str, Any]) -> None:
    st.title("Batch CSV screening")
    st.write("Upload a CSV with the required raw transaction fields:")
    st.code(", ".join(artifact["required_raw_columns"]), language="text")
    uploaded = st.file_uploader("Select transaction CSV", type="csv")
    if uploaded is None:
        return
    try:
        data = pd.read_csv(uploaded)
        if data.empty:
            raise ValueError("The CSV has no transaction rows.")
        if data.columns.duplicated().any():
            raise ValueError("The CSV contains duplicate column names.")
        predictions, probabilities = score(data, artifact)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, KeyError, TypeError, ValueError) as exc:
        st.error(f"Could not process the uploaded CSV: {exc}")
        return

    results = data.copy()
    results["fraud_prediction"] = predictions
    results["fraud_status"] = np.where(predictions == 1, "Potential fraud", "Likely legitimate")
    results["fraud_probability"] = probabilities
    m1, m2 = st.columns(2)
    m1.metric("Rows screened", f"{len(results):,}")
    m2.metric("Flagged", f"{int(predictions.sum()):,}")
    st.dataframe(results, width="stretch", hide_index=True)
    st.download_button(
        "Download scored CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="fraud_screening_results.csv",
        mime="text/csv",
    )


def main() -> None:
    st.sidebar.title("Fraud Detection")
    page = st.sidebar.radio("Page", ["Dashboard", "Score transaction", "Batch screening"])
    try:
        artifact = load_model()
        data = load_dataset()
    except (FileNotFoundError, ImportError, OSError, TypeError, ValueError) as exc:
        st.error(f"Unable to load the project model or dataset: {exc}")
        st.info("Keep app.py, the model, and dataset together, install requirements.txt, then restart.")
        st.stop()

    if page == "Dashboard":
        dashboard(data)
    elif page == "Score transaction":
        single_transaction(artifact)
    else:
        batch_screening(artifact)


if __name__ == "__main__":
    main()
