import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set page layout to wide
st.set_page_config(layout="wide")

# Sidebar - File Upload
st.sidebar.title("Unggah Data")
st.sidebar.info("Unggah file CSV atau Excel untuk analisis clustering")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV atau Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load data
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Menampilkan data yang diunggah
    st.title("Analisis Tren Transaksi Ekspor dan Segmentasi Perusahaan")
    st.write("Berikut adalah data yang diunggah:")
    st.dataframe(df.head())

    # Memastikan bahwa kolom yang digunakan ada dan valid
    st.write("Nama kolom yang tersedia dalam dataset:")
    st.write(df.columns)

    # Membersihkan data untuk analisis
    df['FOB_USD'] = pd.to_numeric(df['FOB_USD'], errors='coerce')
    df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')
    df = df.dropna(subset=['FOB_USD', 'Qty'])  # Menghapus NaN

    # Menampilkan perusahaan dengan transaksi terbanyak
    st.markdown("### Perusahaan yang Sering Melakukan Transaksi")
    transaksi_perusahaan = df.groupby('Nama Perusahaan').size().reset_index(name='Jumlah Transaksi')
    transaksi_perusahaan_sorted = transaksi_perusahaan.sort_values(by='Jumlah Transaksi', ascending=False)
    
    st.write("Berikut adalah perusahaan yang sering melakukan transaksi, diurutkan berdasarkan jumlah transaksi terbanyak:")
    st.dataframe(transaksi_perusahaan_sorted)

    # Preprocessing data untuk clustering
    st.markdown("### Proses Clustering")
    features = ["FOB_USD", "Qty"]
    df_clean = df[features].dropna()  # Remove missing values

    # Normalisasi data (standarisasi)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_clean)

    # Menentukan jumlah cluster menggunakan metode Elbow
    inertia = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)

    # Visualisasi Elbow Method
    st.markdown("### Metode Elbow untuk Menentukan Jumlah Cluster")
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), inertia, marker='o', linestyle='-', color='b')
    ax.set_title('Elbow Method untuk Menentukan Jumlah Cluster')
    ax.set_xlabel('Jumlah Cluster')
    ax.set_ylabel('Inertia')
    st.pyplot(fig)

    # Melakukan KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)  # Misalnya K=3 setelah evaluasi Elbow
    df_clean['Cluster'] = kmeans.fit_predict(scaled_features)

    # Menampilkan hasil clustering dalam bentuk tabel
    st.write("Hasil Clustering:")
    st.dataframe(df_clean.head())

    # Visualisasi hasil clustering
    st.markdown("### Visualisasi Hasil Clustering")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=df_clean['FOB_USD'], y=df_clean['Qty'], hue=df_clean['Cluster'], palette='viridis', s=100)
    ax.set_title("Visualisasi Clustering Perusahaan Berdasarkan Transaksi Ekspor")
    ax.set_xlabel("Nilai FOB (USD)")
    ax.set_ylabel("Jumlah Transaksi")
    plt.legend(title='Cluster', loc='upper right')
    st.pyplot(fig)

    # Menampilkan statistik cluster
    st.markdown("### Statistik Cluster")
    st.write(df_clean.groupby('Cluster').agg({
        'FOB_USD': ['mean', 'std', 'min', 'max'],
        'Qty': ['mean', 'std', 'min', 'max']
    }))

else:
    st.warning("Silakan unggah file CSV atau Excel terlebih dahulu.")
