# =========================================================
# STREAMLIT APP - CEISA COMPANY SEGMENTATION
# Judul: Analisis Pola Transaksi Kepabeanan untuk Segmentasi Keaktifan Perusahaan Mitra
# Metode: K-Means vs DBSCAN (Unsupervised Clustering)
# Input: Upload Excel -> Auto Update
# Output: Segmentasi, evaluasi, visualisasi, Top 5, download Excel
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
# CONFIG
# ---------------------------------------------------------
st.set_page_config(page_title="Segmentasi Keaktifan Perusahaan CEISA", layout="wide")

st.title("Analisis Pola Transaksi Kepabeanan untuk Segmentasi Keaktifan Perusahaan Mitra")
st.caption("Aplikasi Streamlit untuk segmentasi perusahaan mitra berdasarkan pola transaksi (K-Means vs DBSCAN).")

st.markdown(
    """
Aplikasi ini menerima **input data Excel** dari pengguna dan melakukan proses otomatis:
1) pembersihan data dan ekstraksi informasi waktu,  
2) agregasi transaksi per perusahaan,  
3) segmentasi menggunakan **K-Means** dan **DBSCAN**,  
4) evaluasi clustering dan visualisasi hasil,  
5) menampilkan **Top 5 perusahaan** paling aktif berdasarkan total transaksi,  
6) menyediakan file hasil untuk diunduh.
"""
)

# ---------------------------------------------------------
# SIDEBAR INPUT
# ---------------------------------------------------------
st.sidebar.header("Input Data")
uploaded_file = st.sidebar.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

st.sidebar.header("Parameter Model")
K = st.sidebar.slider("K-Means: Jumlah Cluster (K)", 2, 6, 3)
EPS = st.sidebar.slider("DBSCAN: eps", 0.1, 5.0, 0.9, 0.1)
MIN_SAMPLES = st.sidebar.slider("DBSCAN: min_samples", 1, 20, 2)

st.sidebar.header("Opsi Tampilan")
show_raw = st.sidebar.checkbox("Tampilkan data mentah", value=True)
show_notes = st.sidebar.checkbox("Tampilkan catatan evaluasi", value=True)

# ---------------------------------------------------------
# VALIDASI FILE
# ---------------------------------------------------------
if uploaded_file is None:
    st.info("⬅️ Upload file Excel di sidebar untuk memulai. Semua perhitungan akan otomatis update.")
    st.stop()

try:
    df_raw = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Gagal membaca file Excel: {e}")
    st.stop()

REQUIRED_COLS = ["Nama_Perusahaan", "Tanggal Ekspor"]
missing_cols = [c for c in REQUIRED_COLS if c not in df_raw.columns]
if missing_cols:
    st.error(f"Kolom wajib tidak ditemukan: {missing_cols}")
    st.info(f"Kolom yang tersedia: {df_raw.columns.tolist()}")
    st.stop()


# ---------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------
def preprocess_and_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mengubah data per transaksi -> per perusahaan.
    Output: 1 baris = 1 perusahaan.
    """
    df = df.copy()

    # Konversi tanggal
    df["Tanggal Ekspor"] = pd.to_datetime(df["Tanggal Ekspor"], errors="coerce")

    # Buang baris kosong penting
    df = df.dropna(subset=["Nama_Perusahaan", "Tanggal Ekspor"]).copy()

    # Rapikan nama perusahaan
    df["Nama_Perusahaan"] = df["Nama_Perusahaan"].astype(str).str.strip()

    # Ambil bulan
    df["bulan"] = df["Tanggal Ekspor"].dt.to_period("M")

    # Hitung transaksi per bulan per perusahaan
    monthly = (
        df.groupby(["Nama_Perusahaan", "bulan"])
          .size()
          .reset_index(name="transaksi_bulanan")
    )

    # Agregasi final: 1 perusahaan = 1 baris
    agg = monthly.groupby("Nama_Perusahaan").agg(
        total_transaksi=("transaksi_bulanan", "sum"),
        bulan_aktif=("bulan", "nunique"),
        rata_rata_per_bulan=("transaksi_bulanan", "mean"),
        std_transaksi_bulanan=("transaksi_bulanan", "std"),
    ).reset_index()

    # Jika hanya aktif 1 bulan -> std NaN -> isi 0
    agg["std_transaksi_bulanan"] = agg["std_transaksi_bulanan"].fillna(0)

    # Pastikan numerik (anti kasus aneh)
    for c in ["total_transaksi", "bulan_aktif", "rata_rata_per_bulan", "std_transaksi_bulanan"]:
        agg[c] = pd.to_numeric(agg[c], errors="coerce").fillna(0)

    return agg


def safe_clustering_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> tuple:
    """
    Metrik clustering (valid untuk unsupervised):
    - Silhouette (lebih besar lebih baik)
    - Davies-Bouldin (lebih kecil lebih baik)
    - Calinski-Harabasz (lebih besar lebih baik)

    Untuk DBSCAN: noise (-1) dikeluarkan jika memungkinkan.
    """
    labels = np.array(labels)
    unique = set(labels.tolist())

    # Harus ada >= 2 cluster efektif
    if len(unique) <= 1:
        return (np.nan, np.nan, np.nan)

    # Jika ada noise, buang noise untuk evaluasi
    if -1 in unique:
        mask = labels != -1
        if mask.sum() < 2 or len(set(labels[mask].tolist())) <= 1:
            return (np.nan, np.nan, np.nan)
        X_use = X_scaled[mask]
        y_use = labels[mask]
    else:
        X_use = X_scaled
        y_use = labels

    sil = silhouette_score(X_use, y_use)
    dbi = davies_bouldin_score(X_use, y_use)
    ch = calinski_harabasz_score(X_use, y_use)
    return (float(sil), float(dbi), float(ch))


def to_excel_bytes(seg_df: pd.DataFrame,
                   eval_df: pd.DataFrame,
                   cont_matrix: pd.DataFrame,
                   top5_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        seg_df.to_excel(writer, index=False, sheet_name="hasil_segmentasi")
        eval_df.to_excel(writer, index=False, sheet_name="perbandingan_akhir")
        cont_matrix.to_excel(writer, sheet_name="contingency_matrix")
        top5_df.to_excel(writer, index=False, sheet_name="top5_perusahaan")
    return output.getvalue()


# ---------------------------------------------------------
# PROCESS (AUTO UPDATE)
# ---------------------------------------------------------
agg = preprocess_and_aggregate(df_raw)

FEATURES = ["total_transaksi", "bulan_aktif", "rata_rata_per_bulan", "std_transaksi_bulanan"]

# Jika perusahaan terlalu sedikit, stop dengan pesan jelas
if agg.shape[0] < 2:
    st.error("Jumlah perusahaan < 2, clustering tidak dapat dilakukan. Coba upload data yang lebih banyak.")
    st.stop()

# Scaling
X = agg[FEATURES].values
X_scaled = StandardScaler().fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=K, random_state=42, n_init="auto")
labels_km = kmeans.fit_predict(X_scaled)
agg["cluster_kmeans"] = labels_km

# DBSCAN
dbscan = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES)
labels_db = dbscan.fit_predict(X_scaled)
agg["cluster_dbscan"] = labels_db

# Contingency matrix (confusion-like)
cont_matrix = pd.crosstab(
    agg["cluster_kmeans"],
    agg["cluster_dbscan"],
    rownames=["K-Means Cluster"],
    colnames=["DBSCAN Cluster"]
)

# Evaluasi (pengganti accuracy/loss)
sil_km, dbi_km, ch_km = safe_clustering_metrics(X_scaled, labels_km)
sil_db, dbi_db, ch_db = safe_clustering_metrics(X_scaled, labels_db)

noise_count = int((labels_db == -1).sum())
noise_ratio = float(noise_count / len(labels_db))

# jumlah cluster efektif DBSCAN (tanpa noise)
dbscan_clusters = sorted(set(labels_db.tolist()))
dbscan_clusters_no_noise = [c for c in dbscan_clusters if c != -1]

eval_df = pd.DataFrame([
    {
        "Algoritma": "K-Means",
        "n_clusters": len(set(labels_km.tolist())),
        "Silhouette": sil_km,
        "Davies_Bouldin": dbi_km,
        "Calinski_Harabasz": ch_km,
        "Inertia (Loss K-Means)": float(kmeans.inertia_),
        "Noise Ratio (DBSCAN)": np.nan
    },
    {
        "Algoritma": "DBSCAN",
        "n_clusters": len(dbscan_clusters_no_noise),
        "Silhouette": sil_db,
        "Davies_Bouldin": dbi_db,
        "Calinski_Harabasz": ch_db,
        "Inertia (Loss K-Means)": np.nan,
        "Noise Ratio (DBSCAN)": noise_ratio
    }
])

# Top 5 perusahaan
top5 = (
    agg.sort_values("total_transaksi", ascending=False)
       .head(5)
       .copy()
)

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Data", "🔗 Segmentasi", "📊 Evaluasi", "📈 Visualisasi", "⬇️ Unduh"])

# ---------------------------------------------------------
# TAB 1 - DATA
# ---------------------------------------------------------
with tab1:
    st.subheader("Ringkasan Data")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total baris data mentah", df_raw.shape[0])
    c2.metric("Total kolom data mentah", df_raw.shape[1])
    c3.metric("Jumlah perusahaan", agg.shape[0])

    if show_raw:
        st.subheader("Preview Data Mentah")
        st.dataframe(df_raw.head(30), use_container_width=True)

    st.subheader("Data Agregasi (1 baris = 1 perusahaan)")
    st.dataframe(agg, use_container_width=True)

# ---------------------------------------------------------
# TAB 2 - SEGMENTASI
# ---------------------------------------------------------
with tab2:
    st.subheader("Hasil Segmentasi Perusahaan")
    st.dataframe(
        agg[["Nama_Perusahaan"] + FEATURES + ["cluster_kmeans", "cluster_dbscan"]],
        use_container_width=True
    )

    st.subheader("Top 5 Perusahaan Paling Aktif (berdasarkan total transaksi)")
    st.dataframe(
        top5[["Nama_Perusahaan", "total_transaksi", "bulan_aktif", "std_transaksi_bulanan", "cluster_kmeans", "cluster_dbscan"]],
        use_container_width=True
    )

# ---------------------------------------------------------
# TAB 3 - EVALUASI
# ---------------------------------------------------------
with tab3:
    st.subheader("Perbandingan Akhir K-Means vs DBSCAN")
    st.dataframe(eval_df, use_container_width=True)

    if show_notes:
        st.caption(
            "Catatan evaluasi: Karena ini clustering (unsupervised), tidak ada accuracy/loss klasik. "
            "Sebagai pengganti digunakan Silhouette (besar lebih baik), Davies–Bouldin (kecil lebih baik), "
            "Calinski–Harabasz (besar lebih baik). K-Means memiliki Inertia (loss resmi), sedangkan DBSCAN "
            "dinilai juga dari Noise Ratio (lebih kecil lebih baik)."
        )

    st.subheader("Contingency Matrix (Confusion-like) K-Means vs DBSCAN")
    st.dataframe(cont_matrix, use_container_width=True)

    st.markdown("**Visualisasi Contingency Matrix**")
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
    ax_cm.imshow(cont_matrix.values)

    ax_cm.set_xticks(range(len(cont_matrix.columns)))
    ax_cm.set_xticklabels(cont_matrix.columns)
    ax_cm.set_yticks(range(len(cont_matrix.index)))
    ax_cm.set_yticklabels(cont_matrix.index)

    ax_cm.set_xlabel("DBSCAN Cluster")
    ax_cm.set_ylabel("K-Means Cluster")
    ax_cm.set_title("Contingency Matrix: K-Means vs DBSCAN")

    for i in range(cont_matrix.shape[0]):
        for j in range(cont_matrix.shape[1]):
            ax_cm.text(j, i, cont_matrix.values[i, j], ha="center", va="center")

    st.pyplot(fig_cm)

    st.subheader("DBSCAN Noise/Outlier")
    st.write(f"Noise (-1): **{noise_count}** dari **{len(labels_db)}** perusahaan (**{noise_ratio*100:.2f}%**)")
    fig_noise, ax_noise = plt.subplots(figsize=(5, 5))
    ax_noise.pie([len(labels_db) - noise_count, noise_count], labels=["Clustered", "Noise (-1)"], autopct="%1.1f%%")
    ax_noise.set_title("DBSCAN: Proporsi Clustered vs Noise")
    st.pyplot(fig_noise)

# ---------------------------------------------------------
# TAB 4 - VISUALISASI
# ---------------------------------------------------------
with tab4:
    st.subheader("Scatter Plot: total_transaksi vs std_transaksi_bulanan")
    fig_scatter, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].scatter(agg["total_transaksi"], agg["std_transaksi_bulanan"], c=agg["cluster_kmeans"])
    ax[0].set_title("K-Means")
    ax[0].set_xlabel("total_transaksi")
    ax[0].set_ylabel("std_transaksi_bulanan")

    ax[1].scatter(agg["total_transaksi"], agg["std_transaksi_bulanan"], c=agg["cluster_dbscan"])
    ax[1].set_title("DBSCAN")
    ax[1].set_xlabel("total_transaksi")
    ax[1].set_ylabel("std_transaksi_bulanan")

    st.pyplot(fig_scatter)
    st.markdown("---")


    # -------- TOP 5 CHART (FIXED - PASTI MUNCUL) --------
    st.subheader("Top 5 Perusahaan Paling Aktif (Total Transaksi)")

    # Kalau perusahaan < 5, tetap tampilkan sebanyak yang ada
    if top5.shape[0] == 0:
        st.warning("Top 5 tidak tersedia (data agregasi kosong).")
    else:
        fig_top5, ax_top5 = plt.subplots(figsize=(8, 5))
        ax_top5.barh(top5["Nama_Perusahaan"], top5["total_transaksi"])
        ax_top5.set_xlabel("Total Transaksi")
        ax_top5.set_ylabel("Nama Perusahaan")
        ax_top5.set_title("Top Perusahaan dengan Transaksi Kepabeanan Tertinggi")
        ax_top5.invert_yaxis()  # ranking 1 di atas
        st.pyplot(fig_top5)

        st.caption(
            "Grafik ini menampilkan perusahaan dengan total transaksi tertinggi pada periode data yang dianalisis. "
            "Perusahaan-perusahaan ini dapat dianggap sebagai mitra paling aktif berdasarkan frekuensi transaksi."
        )

    st.subheader("Visualisasi PCA 2D")
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X_scaled)

    fig_pca, axp = plt.subplots(1, 2, figsize=(12, 5))
    axp[0].scatter(X2[:, 0], X2[:, 1], c=agg["cluster_kmeans"])
    axp[0].set_title("PCA - K-Means")
    axp[0].set_xlabel("PC1")
    axp[0].set_ylabel("PC2")

    axp[1].scatter(X2[:, 0], X2[:, 1], c=agg["cluster_dbscan"])
    axp[1].set_title("PCA - DBSCAN")
    axp[1].set_xlabel("PC1")
    axp[1].set_ylabel("PC2")

    st.pyplot(fig_pca)

# ---------------------------------------------------------
# TAB 5 - UNDUH
# ---------------------------------------------------------
with tab5:
    st.subheader("Unduh Hasil")
    st.write("File Excel berisi: hasil segmentasi, perbandingan akhir, contingency matrix, dan top 5 perusahaan.")

    excel_bytes = to_excel_bytes(agg, eval_df, cont_matrix, top5)
    st.download_button(
        label="⬇️ Download hasil_segmentasi_dan_evaluasi.xlsx",
        data=excel_bytes,
        file_name="hasil_segmentasi_dan_evaluasi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
