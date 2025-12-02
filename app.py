import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Layout lebar
st.set_page_config(page_title="Dashboard Ekspor K-Means", layout="wide")

st.title("Analisis Tren Transaksi Ekspor dan Segmentasi Perusahaan\nMenggunakan K-Means Clustering")

# === SIDEBAR UPLOAD ===
st.sidebar.title("Unggah Data")
uploaded_file = st.sidebar.file_uploader(
    "Pilih file CSV atau Excel", type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("Silakan unggah file CSV / Excel dulu.")
    st.stop()

# pastikan kolom wajib ada
required_cols = ["Nama_Perusahaan", "FOB_USD", "Qty"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Kolom berikut tidak ditemukan di dataset: {missing}")
    st.stop()

 === BACA DATA ===
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

st.subheader("Preview Data")
st.dataframe(df.head())

st.write("Nama kolom di data:")
st.write(list(df.columns))

# pastikan kolom wajib ada
required_cols = ["Nama_Perusahaan", "FOB_USD", "Qty"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Kolom berikut tidak ditemukan di dataset: {missing}")
    st.stop()

# === BERSIHKAN & KONVERSI ANGKA ===
# FOB_USD di file kamu pakai koma sebagai pemisah ribuan, misal '59,113.80'
df["FOB_USD"] = (
    df["FOB_USD"]
    .astype(str)
    .str.replace(",", "", regex=False)  # buang koma ribuan
)
df["FOB_USD"] = pd.to_numeric(df["FOB_USD"], errors="coerce")

df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce")

# kalau masih ada NaN di dua kolom ini, ganti dengan median (supaya tidak error tapi tetap masuk perhitungan)
df["FOB_USD"] = df["FOB_USD"].fillna(df["FOB_USD"].median())
df["Qty"] = df["Qty"].fillna(df["Qty"].median())

# === PERUSAHAAN DENGAN TRANSAKSI TERBANYAK ===
st.subheader("Perusahaan yang Sering Melakukan Transaksi")

transaksi_perusahaan = (
    df.groupby("Nama_Perusahaan")
      .size()
      .reset_index(name="Jumlah_Transaksi")
      .sort_values("Jumlah_Transaksi", ascending=False)
)

st.dataframe(transaksi_perusahaan)

fig, ax = plt.subplots(figsize=(10, 6))
top10 = transaksi_perusahaan.head(10)
sns.barplot(
    data=top10,
    x="Jumlah_Transaksi",
    y="Nama_Perusahaan",
    ax=ax
)
ax.set_title("Top 10 Perusahaan dengan Jumlah Transaksi Terbanyak")
ax.set_xlabel("Jumlah Transaksi")
ax.set_ylabel("Nama Perusahaan")
plt.tight_layout()
st.pyplot(fig)

# === CLUSTERING K-MEANS ===
st.subheader("Proses Clustering Berdasarkan FOB_USD dan Qty")

features = ["FOB_USD", "Qty"]
X = df[features].copy()

# Standardisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
inertia = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(K_range, inertia, marker="o")
ax.set_xlabel("Jumlah Cluster (k)")
ax.set_ylabel("Inertia")
ax.set_title("Elbow Method")
st.pyplot(fig)

# untuk simpel, pakai k = 3 (boleh kamu jadikan input slider nanti)
k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

df_cluster = df.copy()
df_cluster["Cluster"] = cluster_labels

st.write("Contoh Hasil Clustering:")
st.dataframe(df_cluster[["Nama_Perusahaan", "FOB_USD", "Qty", "Cluster"]].head())

# Silhouette score
if k_optimal > 1:
    sil = silhouette_score(X_scaled, cluster_labels)
    st.write(f"Silhouette Score (k={k_optimal}): **{sil:.3f}**")

# Scatter plot cluster
fig, ax = plt.subplots(figsize=(5, 4))
sns.scatterplot(
    data=df_cluster,
    x="FOB_USD",
    y="Qty",
    hue="Cluster",
    palette="viridis",
    s=80,
    ax=ax,
)
ax.set_title("Visualisasi Cluster Berdasarkan FOB_USD dan Qty")
ax.set_xlabel("FOB_USD")
ax.set_ylabel("Qty")
plt.legend(title="Cluster")
st.pyplot(fig)

# === RINGKASAN RANGE SETIAP CLUSTER (UMUM) ===
st.subheader("Ringkasan Range Setiap Cluster")

# daftar fitur yang dipakai clustering
feature_cols = ["FOB_USD", "Qty"]

# hitung statistik per cluster (min, max, mean, median)
cluster_stats = (
    df_cluster
    .groupby("Cluster")[feature_cols]
    .agg(["min", "max", "mean", "median"])
    .round(2)
)

st.write("Tabel berikut menunjukkan rentang nilai (min–max) dan rata-rata untuk setiap cluster:")
st.dataframe(cluster_stats)

# opsional: ubah jadi format lebih rapi (1 level kolom)
cluster_stats_flat = cluster_stats.copy()
cluster_stats_flat.columns = [f"{col[0]}_{col[1]}" for col in cluster_stats.columns]
cluster_stats_flat = cluster_stats_flat.reset_index()

# tampilkan juga dalam bentuk teks ringkas per cluster
st.markdown("### Penjelasan Otomatis Per Cluster")

for _, row in cluster_stats_flat.iterrows():
    c = int(row["Cluster"])
    fob_min = row["FOB_USD_min"]
    fob_max = row["FOB_USD_max"]
    qty_min = row["Qty_min"]
    qty_max = row["Qty_max"]
    fob_mean = row["FOB_USD_mean"]
    qty_mean = row["Qty_mean"]

    st.markdown(
        f"""
**Cluster {c}**

- Range Qty: **{qty_min:.0f} – {qty_max:.0f}**
- Range FOB_USD: **{fob_min:,.2f} – {fob_max:,.2f}**
- Rata-rata Qty: **{qty_mean:.0f}**
- Rata-rata FOB_USD: **{fob_mean:,.2f}**

Interpretasi umum:
Cluster {c} merepresentasikan kelompok perusahaan dengan rata-rata nilai ekspor (FOB) sekitar **{fob_mean:,.2f} USD** 
dan rata-rata jumlah barang sekitar **{qty_mean:.0f} unit**, dengan variasi dari **{qty_min:.0f}–{qty_max:.0f}** untuk Qty 
dan **{fob_min:,.2f}–{fob_max:,.2f} USD** untuk FOB_USD.
"""
    )

