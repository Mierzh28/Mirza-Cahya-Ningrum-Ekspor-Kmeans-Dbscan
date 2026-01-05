# =========================================================
# CEISA STREAMLIT APP
# Analisis Pola Transaksi Kepabeanan
# Segmentasi Keaktifan Perusahaan Mitra
# =========================================================

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# CONFIG APLIKASI
# ---------------------------------------------------------
st.set_page_config(
    page_title="Segmentasi Keaktifan Perusahaan CEISA",
    layout="wide"
)

st.title("Analisis Pola Transaksi Kepabeanan untuk Segmentasi Keaktifan Perusahaan Mitra")
st.caption("Clustering menggunakan K-Means dan DBSCAN (data selalu update otomatis)")

# ---------------------------------------------------------
# SIDEBAR – INPUT USER
# ---------------------------------------------------------
st.sidebar.header("Input Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload file Excel (.xlsx)",
    type=["xlsx"]
)

st.sidebar.header("Parameter Model")

K = st.sidebar.slider("K-Means: Jumlah Cluster (K)", 2, 6, 3)
EPS = st.sidebar.slider("DBSCAN: eps", 0.1, 5.0, 0.9, 0.1)
MIN_SAMPLES = st.sidebar.slider("DBSCAN: min_samples", 1, 20, 2)

# ---------------------------------------------------------
# VALIDASI FILE
# ---------------------------------------------------------
if uploaded_file is None:
    st.info("⬅️ Upload file Excel di sidebar untuk memulai.")
    st.stop()

df_raw = pd.read_excel(uploaded_file)

REQUIRED_COLS = ["Nama_Perusahaan", "Tanggal Ekspor"]
missing_cols = [c for c in REQUIRED_COLS if c not in df_raw.columns]

if missing_cols:
    st.error(f"Kolom wajib tidak ditemukan: {missing_cols}")
    st.info(f"Kolom yang tersedia: {df_raw.columns.tolist()}")
    st.stop()

# ---------------------------------------------------------
# FUNGSI PREPROCESSING & AGREGASI
# ---------------------------------------------------------
def preprocess_and_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["Tanggal Ekspor"] = pd.to_datetime(df["Tanggal Ekspor"], errors="coerce")
    df = df.dropna(subset=["Nama_Perusahaan", "Tanggal Ekspor"])

    df["Nama_Perusahaan"] = df["Nama_Perusahaan"].astype(str).str.strip()
    df["bulan"] = df["Tanggal Ekspor"].dt.to_period("M")

    monthly = (
        df.groupby(["Nama_Perusahaan", "bulan"])
          .size()
          .reset_index(name="transaksi_bulanan")
    )

    agg = monthly.groupby("Nama_Perusahaan").agg(
        total_transaksi=("transaksi_bulanan", "sum"),
        bulan_aktif=("bulan", "nunique"),
        rata_rata_per_bulan=("transaksi_bulanan", "mean"),
        std_transaksi_bulanan=("transaksi_bulanan", "std")
    ).reset_index()

    agg["std_transaksi_bulanan"] = agg["std_transaksi_bulanan"].fillna(0)
    return agg

# ---------------------------------------------------------
# PROSES INTI (AUTO UPDATE)
# ---------------------------------------------------------
agg = preprocess_and_aggregate(df_raw)

FEATURES = [
    "total_transaksi",
    "bulan_aktif",
    "rata_rata_per_bulan",
    "std_transaksi_bulanan"
]

X = agg[FEATURES].values
X_scaled = StandardScaler().fit_transform(X)

# --- K-MEANS
kmeans = KMeans(n_clusters=K, random_state=42, n_init="auto")
agg["cluster_kmeans"] = kmeans.fit_predict(X_scaled)

# --- DBSCAN
dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
agg["cluster_dbscan"] = dbscan.fit_predict(X_scaled)

# ---------------------------------------------------------
# TAB LAYOUT
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📄 Data", "🔗 Clustering", "📊 Evaluasi", "📈 Visualisasi", "⬇️ Unduh"]
)

# ---------------------------------------------------------
# TAB 1 – DATA
# ---------------------------------------------------------
with tab1:
    st.subheader("Data Mentah (Preview)")
    st.dataframe(df_raw.head(20), use_container_width=True)

    st.subheader("Data Agregasi (1 baris = 1 perusahaan)")
    st.dataframe(agg, use_container_width=True)

# ---------------------
