"""
DMT Activity 3 - Team 16
Banking Transaction Monitoring Dashboard

Team Members:
  - Dharshini R (732924ADR021)
  - Kiruthika T (732924ADR057)
  - Aamina Thasin M (732924ADR001)

Run with:
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# ----------------------------- Page Config -----------------------------
st.set_page_config(
    page_title="Banking Transaction Monitoring - Team 16",
    page_icon="🏦",
    layout="wide",
)

# ----------------------------- Data Loading -----------------------------
@st.cache_data
def generate_data(n=1000, seed=42):
    """Generate synthetic banking transactions with a few injected anomalies."""
    rng = np.random.default_rng(seed)
    locations = ["Chennai", "Mumbai", "Delhi", "Bangalore", "Kolkata", "Hyderabad"]
    types = ["Debit", "Credit", "Transfer", "Withdrawal"]

    data = {
        "TransactionID": [f"TXN{i:05d}" for i in range(1, n + 1)],
        "Amount": np.round(rng.normal(5000, 1500, n).clip(100, None), 2),
        "Hour": rng.integers(6, 23, n),
        "Location": rng.choice(locations, n),
        "Type": rng.choice(types, n),
    }
    df = pd.DataFrame(data)

    anomalies = pd.DataFrame({
        "TransactionID": [f"TXN{n + i:05d}" for i in range(1, 31)],
        "Amount": np.round(rng.uniform(50000, 200000, 30), 2),
        "Hour": rng.choice([1, 2, 3, 4], 30),
        "Location": rng.choice(locations, 30),
        "Type": rng.choice(types, 30),
    })
    return pd.concat([df, anomalies], ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)


@st.cache_data
def detect_anomalies(df: pd.DataFrame, contamination: float):
    model = IsolationForest(contamination=contamination, random_state=42)
    df = df.copy()
    df["Anomaly"] = model.fit_predict(df[["Amount", "Hour"]])
    df["Status"] = np.where(df["Anomaly"] == -1, "Suspicious", "Normal")
    return df


# ----------------------------- Sidebar -----------------------------
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("### Team 16")
st.sidebar.markdown(
    "- **Dharshini R** (732924ADR021)\n"
    "- **Kiruthika T** (732924ADR057)\n"
    "- **Aamina Thasin M** (732924ADR001)"
)
st.sidebar.divider()

uploaded = st.sidebar.file_uploader("Upload transactions CSV (optional)", type=["csv"])
contamination = st.sidebar.slider(
    "Contamination (expected % of fraud)", 0.01, 0.10, 0.03, 0.01
)

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = generate_data()

df = detect_anomalies(df_raw, contamination)

# ----------------------------- Header -----------------------------
st.title("🏦 Banking Transaction Monitoring")
st.caption("DMT Activity 3 — Fraud Detection using Isolation Forest")

# ----------------------------- KPIs -----------------------------
total = len(df)
suspicious = int((df["Status"] == "Suspicious").sum())
normal = total - suspicious
flagged_amount = df.loc[df["Status"] == "Suspicious", "Amount"].sum()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions", f"{total:,}")
c2.metric("Suspicious", f"{suspicious:,}", f"{suspicious / total * 100:.2f}%")
c3.metric("Normal", f"{normal:,}")
c4.metric("Flagged Amount", f"₹{flagged_amount:,.0f}")

st.divider()

# ----------------------------- Tabs -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Suspicious", "📈 Analysis", "📁 Data"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Amount vs Hour")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(
            data=df, x="Hour", y="Amount", hue="Status",
            palette={"Normal": "#4CAF50", "Suspicious": "#E53935"},
            alpha=0.7, ax=ax,
        )
        ax.set_title("Transaction Pattern")
        st.pyplot(fig)

    with col2:
        st.subheader("Status Distribution")
        fig, ax = plt.subplots(figsize=(7, 5))
        counts = df["Status"].value_counts()
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%",
               colors=["#4CAF50", "#E53935"], startangle=90)
        ax.set_title("Normal vs Suspicious")
        st.pyplot(fig)

with tab2:
    st.subheader("🚨 Flagged Suspicious Transactions")
    susp = df[df["Status"] == "Suspicious"].sort_values("Amount", ascending=False)
    st.dataframe(susp, use_container_width=True)

    st.download_button(
        "⬇️ Download Suspicious CSV",
        susp.to_csv(index=False).encode(),
        "suspicious_transactions.csv",
        "text/csv",
    )

    st.subheader("Suspicious by Location")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.countplot(data=susp, x="Location",
                  order=susp["Location"].value_counts().index,
                  color="#E53935", ax=ax)
    ax.set_title("Where suspicious activity occurs")
    st.pyplot(fig)

with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Amount Distribution")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(df["Amount"], bins=50, kde=True, color="#1E88E5", ax=ax)
        ax.set_title("Transaction Amount Histogram")
        st.pyplot(fig)

    with col2:
        st.subheader("Transactions by Hour")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.countplot(data=df, x="Hour", hue="Status",
                      palette={"Normal": "#4CAF50", "Suspicious": "#E53935"}, ax=ax)
        ax.set_title("Hourly Activity")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.subheader("Average Amount by Type")
    avg = df.groupby(["Type", "Status"])["Amount"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=avg, x="Type", y="Amount", hue="Status",
                palette={"Normal": "#4CAF50", "Suspicious": "#E53935"}, ax=ax)
    ax.set_title("Average Transaction Amount")
    st.pyplot(fig)

with tab4:
    st.subheader("📁 Full Dataset")
    status_filter = st.multiselect("Filter Status", ["Normal", "Suspicious"],
                                   default=["Normal", "Suspicious"])
    loc_filter = st.multiselect("Filter Location",
                                sorted(df["Location"].unique()),
                                default=sorted(df["Location"].unique()))
    filtered = df[df["Status"].isin(status_filter) & df["Location"].isin(loc_filter)]
    st.dataframe(filtered, use_container_width=True)
    st.download_button(
        "⬇️ Download Full CSV",
        df.to_csv(index=False).encode(),
        "transactions.csv",
        "text/csv",
    )

st.divider()
st.caption("DMT Activity 3 © Team 16 | Built with Streamlit + Scikit-learn")
