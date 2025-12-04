import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ------------------------------------------------------
# Page setup
# ------------------------------------------------------
st.set_page_config(page_title="Credit Card Fraud Dashboard", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")
st.write("Dataset automatically loaded from Kaggle. No upload required.")

DATASET = "mlg-ulb/creditcardfraud"
CSV_PATH = Path("creditcard.csv")
USER_AGENT = "streamlit-kaggle-client-v2"


# ------------------------------------------------------
# Helpers
# ------------------------------------------------------
def get_secret(key: str):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)


def has_secret(key: str) -> bool:
    value = get_secret(key)
    return value is not None and str(value).strip() != ""


def prepare_kaggle_credentials():
    username = get_secret("KAGGLE_USERNAME")
    # Prefer API token; fall back to legacy key
    api_token = get_secret("KAGGLE_API_TOKEN")
    legacy_key = get_secret("KAGGLE_KEY")
    key = api_token or legacy_key

    if not username or not key:
        raise RuntimeError("KAGGLE_USERNAME and KAGGLE_API_TOKEN (or KAGGLE_KEY) are required.")

    kaggle_dir = Path(os.path.expanduser("~/.kaggle"))
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json_path = kaggle_dir / "kaggle.json"

    kaggle_json_path.write_text(json.dumps({"username": str(username).strip(), "key": str(key).strip()}))
    try:
        kaggle_json_path.chmod(0o600)
    except Exception:
        pass

    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_dir)
    os.environ["KAGGLE_USERNAME"] = str(username).strip()
    os.environ["KAGGLE_KEY"] = str(key).strip()
    os.environ["KAGGLE_USER_AGENT"] = USER_AGENT

    return kaggle_json_path


# ------------------------------------------------------
# Data loading
# ------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)

    prepare_kaggle_credentials()

    # Import after env is set so the client picks up our user agent and config.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi(user_agent=USER_AGENT)
    api.authenticate()

    # Double-ensure the session headers include a non-None User-Agent
    try:
        api._session.headers.update({"User-Agent": USER_AGENT})
    except Exception:
        pass

    api.dataset_download_files(DATASET, path=".", unzip=True, quiet=True, force=True)

    if not CSV_PATH.exists():
        raise FileNotFoundError("Download finished but creditcard.csv not found.")

    return pd.read_csv(CSV_PATH)


# ------------------------------------------------------
# Debug credential presence
# ------------------------------------------------------
try:
    secrets_loaded = bool(st.secrets)
except Exception:
    secrets_loaded = False

st.write("Secrets loaded:", secrets_loaded)
st.write("Has KAGGLE_USERNAME:", has_secret("KAGGLE_USERNAME"))
st.write("Has KAGGLE_API_TOKEN:", has_secret("KAGGLE_API_TOKEN"))
st.write("Has KAGGLE_KEY:", has_secret("KAGGLE_KEY"))


# ------------------------------------------------------
# Load data
# ------------------------------------------------------
try:
    with st.spinner("Downloading dataset from Kaggle… (first run only)"):
        df = load_data()
except Exception as e:
    st.error(
        "❌ Failed to download or load dataset. "
        "Check secrets (KAGGLE_USERNAME + KAGGLE_API_TOKEN), dataset access, and Kaggle terms."
    )
    st.write("Error details:", str(e))
    st.stop()

st.success("Dataset ready.")
st.write("Rows:", f"{len(df):,}")
st.write("Columns:", list(df.columns))
st.write("Preview:", df.head())

# Basic stats (no heavy plots to avoid extra failure points)
st.subheader("Fraud Summary")
st.write({
    "fraud_cases": int(df["Class"].sum()),
    "fraud_rate": float(df["Class"].mean()),
    "avg_amount": float(df["Amount"].mean()),
})
