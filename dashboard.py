import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import zipfile
import subprocess
import json

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve
)

# ------------------------------------------------------
# Streamlit Page Setup   
# ------------------------------------------------------
st.set_page_config(page_title="Credit Card Fraud Dashboard", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")
st.write("Dataset automatically loaded from Kaggle. No upload required.")
st.write("Secrets loaded:", bool(st.secrets))
st.write("Has KAGGLE_USERNAME:", "KAGGLE_USERNAME" in st.secrets)
st.write("Has KAGGLE_KEY:", "KAGGLE_KEY" in st.secrets)

# ------------------------------------------------------
# Function: Download dataset from Kaggle (Fully Correct)
# ------------------------------------------------------
@st.cache_data
def load_data():
    kaggle_dataset = "mlg-ulb/creditcardfraud"   # Official dataset ID
    csv_path = "creditcard.csv"

    # Already downloaded?
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    st.info("Downloading dataset from Kaggle… (first time only, please wait)")

    # --- Create ~/.kaggle directory ---
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    # --- Build correct kaggle.json using Streamlit secrets ---
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

    kaggle_credentials = {
        "username": st.secrets["KAGGLE_USERNAME"],
        "key": st.secrets["KAGGLE_KEY"]
    }

    with open(kaggle_json_path, "w") as f:
        json.dump(kaggle_credentials, f)

    # Correct permissions (required by Kaggle)
    os.chmod(kaggle_json_path, 0o600)

    # --- Download ZIP from Kaggle ---
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", kaggle_dataset],
        check=True
    )

    # --- Unzip the dataset ---
    zip_name = kaggle_dataset.split("/")[-1] + ".zip"
    with zipfile.ZipFile(zip_name, "r") as z:
        z.extractall(".")

    return pd.read_csv(csv_path)

# ------------------------------------------------------
# Load the dataset
# ------------------------------------------------------
try:
    df = load_data()
except Exception as e:
    st.error("❌ Failed to download dataset. Check your Streamlit Secrets formatting.")
    st.stop()

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

# ------------------------------------------------------
# Fraud vs Non-Fraud
# ------------------------------------------------------
with colA:
    st.subheader("📊 Fraud vs Non-Fraud Count")
    fig1, ax1 = plt.subplots(figsize=(7,4))
    sns.countplot(data=df, x="Class", ax=ax1)
    ax1.set_xticklabels(["Non-Fraud", "Fraud"])
    ax1.set_title("Fraud Distribution")
    st.pyplot(fig1)

# ------------------------------------------------------
# Amount boxplot
# ------------------------------------------------------
with colB:
    st.subheader("💲 Amount Distribution by Class (Boxplot)")
    fig_box, ax_box = plt.subplots(figsize=(7,4))
    sns.boxplot(data=df, x="Class", y="Amount", ax=ax_box)
    ax_box.set_xticklabels(["Non-Fraud", "Fraud"])
    st.pyplot(fig_box)

# ------------------------------------------------------
# KDE Amount Distribution
# ------------------------------------------------------
st.subheader("💲 Transaction Amount Density (Fraud vs Non-Fraud)")
fig2, ax2 = plt.subplots(figsize=(8,4))
sns.kdeplot(df[df["Class"] == 0]["Amount"], label="Non-Fraud", shade=True)
sns.kdeplot(df[df["Class"] == 1]["Amount"], label="Fraud", shade=True, color="red")
ax2.legend()
st.pyplot(fig2)

# ------------------------------------------------------
# Time of Day Distribution
# ------------------------------------------------------
st.subheader("⏱ Fraud vs Non-Fraud by Time of Day")
fig_time, ax_time = plt.subplots(figsize=(8,4))
sns.kdeplot(df[df["Class"] == 0]["Time"], label="Non-Fraud", shade=True)
sns.kdeplot(df[df["Class"] == 1]["Time"], label="Fraud", shade=True, color="red")
ax_time.legend()
st.pyplot(fig_time)

# ======================================================
#               SECTION 2: MODEL TRAINING
# ======================================================
st.header("🤖 Predictive Modeling with Logistic Regression")

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

# ------------------------------------------------------
# Confusion Matrix
# ------------------------------------------------------
st.subheader("🧮 Confusion Matrix")
cm = confusion_matrix(y_test, preds)

fig_cm, ax_cm = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
ax_cm.set_xlabel("Predicted")
ax_cm.set_ylabel("Actual")
st.pyplot(fig_cm)

# ------------------------------------------------------
# ROC Curve
# ------------------------------------------------------
st.subheader("📈 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, probs)
auc = roc_auc_score(y_test, probs)

fig_roc, ax_roc = plt.subplots(figsize=(6,4))
ax_roc.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
ax_roc.plot([0,1], [0,1], linestyle="--", color="gray")
ax_roc.set_title("ROC Curve")
ax_roc.legend()
st.pyplot(fig_roc)

# ------------------------------------------------------
# Precision-Recall Curve
# ------------------------------------------------------
st.subheader("🎯 Precision-Recall Curve")
precision, recall, _ = precision_recall_curve(y_test, probs)

fig_pr, ax_pr = plt.subplots(figsize=(6,4))
ax_pr.plot(recall, precision, color="purple")
ax_pr.set_title("Precision-Recall Curve")
st.pyplot(fig_pr)

# ======================================================
#                SECTION 3: FEATURE IMPORTANCE
# ======================================================
st.header("🔥 Top Fraud Predictors")

coef = pd.Series(abs(model.coef_[0]), index=X.columns)
top10 = coef.sort_values(ascending=False).head(10)

st.write(top10)

fig3, ax3 = plt.subplots(figsize=(8,4))
sns.barplot(x=top10.values, y=top10.index, palette="viridis", ax=ax3)
st.pyplot(fig3)

# ======================================================
#                  SECTION 4: RISK SEGMENTATION
# ======================================================
st.header("⚠ Fraud Risk Segmentation")

risk = pd.cut(
    probs,
    bins=[0, 0.2, 0.7, 1],
    labels=["Low Risk", "Medium Risk", "High Risk"],
    right=False
)

results = pd.DataFrame({
    "Probability": probs,
    "Risk": risk,
    "Amount": X_test[:, X.columns.get_loc("Amount")]
})

counts = results["Risk"].value_counts()

colR1, colR2 = st.columns(2)

with colR1:
    st.subheader("📊 Risk Segment Distribution")
    st.write(counts)

with colR2:
    st.subheader("🍩 Risk Distribution Donut Chart")
    fig_donut, ax_donut = plt.subplots(figsize=(5,5))
    ax_donut.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    centre = plt.Circle((0,0),0.7,fc='white')
    fig_donut.gca().add_artist(centre)
    st.pyplot(fig_donut)

st.success("Dashboard Loaded Successfully 🎉")