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

# ---------------------------------------------------------
# TAB 2 – CLUSTERING
# ---------------------------------------------------------
with tab2:
    st.subheader("Hasil Segmentasi Perusahaan")
    st.dataframe(
        agg[["Nama_Perusahaan"] + FEATURES + ["cluster_kmeans", "cluster_dbscan"]],
        use_container_width=True
    )

# ---------------------------------------------------------
# TAB 3 – EVALUASI
# ---------------------------------------------------------
with tab3:
    def safe_metrics(X, labels):
        labels = np.array(labels)
        unique = set(labels.tolist())

        noise_ratio = float((labels == -1).sum() / len(labels))

        if len(unique) <= 1:
            return (np.nan, np.nan, np.nan, noise_ratio)

        if -1 in unique:
            mask = labels != -1
            if mask.sum() < 2 or len(set(labels[mask])) <= 1:
                return (np.nan, np.nan, np.nan, noise_ratio)
            X_use = X[mask]
            y_use = labels[mask]
        else:
            X_use = X
            y_use = labels

        sil = silhouette_score(X_use, y_use)
        dbi = davies_bouldin_score(X_use, y_use)
        ch = calinski_harabasz_score(X_use, y_use)

        return (sil, dbi, ch, noise_ratio)

    sil_km, dbi_km, ch_km, _ = safe_metrics(X_scaled, agg["cluster_kmeans"])
    sil_db, dbi_db, ch_db, noise_db = safe_metrics(X_scaled, agg["cluster_dbscan"])

    eval_df = pd.DataFrame([
        {
            "Algoritma": "K-Means",
            "Silhouette": sil_km,
            "Davies_Bouldin": dbi_km,
            "Calinski_Harabasz": ch_km,
            "Inertia (Loss)": kmeans.inertia_,
            "Noise Ratio": np.nan
        },
        {
            "Algoritma": "DBSCAN",
            "Silhouette": sil_db,
            "Davies_Bouldin": dbi_db,
            "Calinski_Harabasz": ch_db,
            "Inertia (Loss)": np.nan,
            "Noise Ratio": noise_db
        }
    ])

    st.subheader("Perbandingan Akhir K-Means vs DBSCAN")
    st.dataframe(eval_df, use_container_width=True)

# ---------------------------------------------------------
# TAB 4 – VISUALISASI
# ---------------------------------------------------------
with tab4:
    st.subheader("Scatter Plot (total_transaksi vs std_transaksi_bulanan)")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].scatter(
        agg["total_transaksi"],
        agg["std_transaksi_bulanan"],
        c=agg["cluster_kmeans"]
    )
    ax[0].set_title("K-Means")
    ax[0].set_xlabel("total_transaksi")
    ax[0].set_ylabel("std_transaksi_bulanan")

    ax[1].scatter(
        agg["total_transaksi"],
        agg["std_transaksi_bulanan"],
        c=agg["cluster_dbscan"]
    )
    ax[1].set_title("DBSCAN")
    ax[1].set_xlabel("total_transaksi")
    ax[1].set_ylabel("std_transaksi_bulanan")

    st.pyplot(fig)

    st.subheader("PCA 2D Visualization")

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_scaled)

    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))

    ax2[0].scatter(X2[:, 0], X2[:, 1], c=agg["cluster_kmeans"])
    ax2[0].set_title("PCA - K-Means")

    ax2[1].scatter(X2[:, 0], X2[:, 1], c=agg["cluster_dbscan"])
    ax2[1].set_title("PCA - DBSCAN")

    st.pyplot(fig2)

# ---------------------------------------------------------
# TAB 5 – DOWNLOAD
# ---------------------------------------------------------
with tab5:
    st.subheader("Unduh Hasil")

    def to_excel_bytes(df1, df2):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df1.to_excel(writer, index=False, sheet_name="hasil_segmentasi")
            df2.to_excel(writer, index=False, sheet_name="evaluasi")
        return output.getvalue()

    st.download_button(
        label="⬇️ Download hasil_segmentasi.xlsx",
        data=to_excel_bytes(agg, eval_df),
        file_name="hasil_segmentasi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
