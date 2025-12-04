import os
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------
# Streamlit Page Setup
# ------------------------------------------------------
st.set_page_config(page_title="Credit Card Fraud Dashboard", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")
st.write("Dataset automatically loaded from Kaggle. No upload required.")

DATASET = "mlg-ulb/creditcardfraud"
CSV_PATH = Path("creditcard.csv")
KAGGLE_USER_AGENT = "streamlit-kaggle-client-v1"


# ------------------------------------------------------
# Helpers for secrets/env handling
# ------------------------------------------------------
def get_secret(key: str):
    """Read key from Streamlit secrets if available, else env."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key)


def has_secret(key: str) -> bool:
    value = get_secret(key)
    return value is not None and str(value).strip() != ""


def resolve_kaggle_credentials():
    """
    Prefer the new API token (KAGGLE_API_TOKEN). Fall back to legacy username/key.
    Raises if nothing usable is present.
    """
    api_token = get_secret("KAGGLE_API_TOKEN")
    username = get_secret("KAGGLE_USERNAME")
    key = get_secret("KAGGLE_KEY")

    if api_token:
        return {"token": str(api_token).strip()}, "token"
    if username and key:
        return {"username": str(username).strip(), "key": str(key).strip()}, "legacy"

    raise RuntimeError(
        "No Kaggle credentials found. Set KAGGLE_API_TOKEN (preferred) or "
        "KAGGLE_USERNAME and KAGGLE_KEY in Streamlit secrets."
    )


def write_kaggle_config(creds: dict):
    """
    Write ~/.kaggle/kaggle.json and set env vars appropriately.
    Supports both token-only and username/key.
    """
    kaggle_dir = Path(os.path.expanduser("~/.kaggle"))
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    kaggle_json_path = kaggle_dir / "kaggle.json"
    kaggle_json_path.write_text(json.dumps(creds))
    try:
        kaggle_json_path.chmod(0o600)
    except Exception:
        # Some platforms may not support chmod semantics; ignore.
        pass

    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_dir)
    os.environ["KAGGLE_USER_AGENT"] = KAGGLE_USER_AGENT

    if "token" in creds:
        os.environ["KAGGLE_API_TOKEN"] = creds["token"]
        os.environ.pop("KAGGLE_USERNAME", None)
        os.environ.pop("KAGGLE_KEY", None)
    else:
        os.environ["KAGGLE_USERNAME"] = creds["username"]
        os.environ["KAGGLE_KEY"] = creds["key"]

    return kaggle_json_path


# Debug Secret State
try:
    secrets_loaded = bool(st.secrets)
except Exception:
    secrets_loaded = False

st.write("Secrets loaded:", secrets_loaded)
st.write("Has KAGGLE_USERNAME:", has_secret("KAGGLE_USERNAME"))
st.write("Has KAGGLE_KEY:", has_secret("KAGGLE_KEY"))
st.write("Has KAGGLE_API_TOKEN:", has_secret("KAGGLE_API_TOKEN"))


# ------------------------------------------------------
# Function: Download + load dataset
# ------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH)

    creds, mode = resolve_kaggle_credentials()
    kaggle_json_path = write_kaggle_config(creds)

    api = KaggleApi()
    # Ensure user_agent is set before authenticate to avoid None header errors.
    api.config_values["user_agent"] = KAGGLE_USER_AGENT
    api.authenticate()
    try:
        api.set_config_value("user_agent", KAGGLE_USER_AGENT)
    except Exception:
        # Older kaggle versions may not expose this; safe to ignore.
        pass

    api.dataset_download_files(DATASET, path=".", unzip=True, quiet=True, force=True)

    if not CSV_PATH.exists():
        raise FileNotFoundError("Download succeeded but creditcard.csv was not found.")

    return pd.read_csv(CSV_PATH)


# ------------------------------------------------------
# Load the dataset
# ------------------------------------------------------
try:
    with st.spinner("Downloading dataset from Kaggle… (first run only, please wait)"):
        df = load_data()
except Exception as e:
    st.error(
        "❌ Failed to download or load dataset. Check Streamlit secrets, Kaggle credentials, "
        "dataset access, and that your Kaggle account accepted the dataset terms."
    )
    st.write("Error details:", str(e))
    st.stop()

st.write("Kaggle JSON Exists:", Path(os.path.expanduser("~/.kaggle/kaggle.json")).exists())

# Sidebar Overview
st.sidebar.header("Dataset Overview")
st.sidebar.metric("Total Transactions", f"{len(df):,}")
st.sidebar.metric("Fraud Cases", f"{df['Class'].sum():,}")
st.sidebar.metric("Fraud Rate", f"{df['Class'].mean():.4%}")
st.sidebar.metric("Average Amount", f"${df['Amount'].mean():.2f}")

# ======================================================
#               SECTION 1: GENERAL DISTRIBUTIONS
# ======================================================
st.header("📦 General Dataset Insights")

colA, colB = st.columns(2)

# Fraud count
with colA:
    st.subheader("📊 Fraud vs Non-Fraud Count")
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="Class", ax=ax1)
    ax1.set_xticklabels(["Non-Fraud", "Fraud"])
    ax1.set_title("Fraud Distribution")
    st.pyplot(fig1)

# Boxplot
with colB:
    st.subheader("💲 Amount Distribution by Class")
    fig_box, ax_box = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="Class", y="Amount", ax=ax_box)
    ax_box.set_xticklabels(["Non-Fraud", "Fraud"])
    st.pyplot(fig_box)

# KDE Amount
st.subheader("💲 Transaction Amount Density")
fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.kdeplot(df[df["Class"] == 0]["Amount"], label="Non-Fraud", fill=True)
sns.kdeplot(df[df["Class"] == 1]["Amount"], label="Fraud", fill=True, color="red")
ax2.legend()
st.pyplot(fig2)

# Time KDE
st.subheader("⏱ Fraud vs Non-Fraud by Time")
fig_time, ax_time = plt.subplots(figsize=(8, 4))
sns.kdeplot(df[df["Class"] == 0]["Time"], label="Non-Fraud", fill=True)
sns.kdeplot(df[df["Class"] == 1]["Time"], label="Fraud", fill=True, color="red")
ax_time.legend()
st.pyplot(fig_time)

# ======================================================
# MODEL TRAINING
# ======================================================
st.header("🤖 Predictive Modeling (Logistic Regression)")

X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

model = LogisticRegression(max_iter=5000, class_weight="balanced")
model.fit(X_train, y_train)

probs = model.predict_proba(X_test)[:, 1]
preds = model.predict(X_test)

# Confusion Matrix
st.subheader("🧮 Confusion Matrix")
cm = confusion_matrix(y_test, preds)
fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
st.pyplot(fig_cm)

# ROC Curve
st.subheader("📈 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, probs)
auc = roc_auc_score(y_test, probs)
fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
ax_roc.legend()
st.pyplot(fig_roc)

# Precision Recall
st.subheader("🎯 Precision-Recall Curve")
precision, recall, _ = precision_recall_curve(y_test, probs)
fig_pr, ax_pr = plt.subplots(figsize=(6, 4))
ax_pr.plot(recall, precision)
st.pyplot(fig_pr)

# ======================================================
# FEATURE IMPORTANCE
# ======================================================
st.header("🔥 Top Fraud Predictors")
coef = pd.Series(abs(model.coef_[0]), index=X.columns).sort_values(ascending=False).head(10)
st.write(coef)
fig3, ax3 = plt.subplots(figsize=(8, 4))
sns.barplot(x=coef.values, y=coef.index, palette="viridis", ax=ax3)
st.pyplot(fig3)

# ======================================================
# RISK SEGMENTATION
# ======================================================
st.header("⚠ Fraud Risk Segmentation")
risk = pd.cut(
    probs,
    bins=[0, 0.2, 0.7, 1],
    labels=["Low Risk", "Medium Risk", "High Risk"],
    right=False,
)

results = pd.DataFrame(
    {
        "Probability": probs,
        "Risk": risk,
        "Amount": X_test[:, X.columns.get_loc("Amount")],
    }
)

counts = results["Risk"].value_counts()
colR1, colR2 = st.columns(2)

with colR1:
    st.subheader("📊 Counts per Risk Bucket")
    st.write(counts)

with colR2:
    st.subheader("🍩 Donut Chart")
    fig_donut, ax_donut = plt.subplots(figsize=(5, 5))
    ax_donut.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    centre = plt.Circle((0, 0), 0.7, fc="white")
    fig_donut.gca().add_artist(centre)
    st.pyplot(fig_donut)

st.success("Dashboard Loaded Successfully 🎉")
