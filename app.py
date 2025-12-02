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

    # Menangani data yang tidak valid (NaN) pada kolom 'FOB_USD' dan 'Qty'
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

    # Melakukan KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_clean['Cluster'] = kmeans.fit_predict(scaled_features)

    # Menampilkan hasil clustering dalam bentuk tabel
    st.write("Hasil Clustering:")
    st.dataframe(df_clean.head())

    # Visualisasi Pie Chart
    st.markdown("### Visualisasi Pie Chart Berdasarkan Cluster")
    cluster_counts = df_clean['Cluster'].value_counts()
    fig, ax = plt.subplots()
    ax.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3", len(cluster_counts)))
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig)

    # Visualisasi Bar Chart
    st.markdown("### Visualisasi Bar Chart Berdasarkan Cluster")
    cluster_summary = df_clean.groupby('Cluster').agg({'FOB_USD': 'mean', 'Qty': 'mean'}).reset_index()
    fig, ax = plt.subplots()
    sns.barplot(data=cluster_summary, x='Cluster', y='FOB_USD', palette='Set2')
    ax.set_title("Rata-rata Nilai Ekspor (FOB) per Cluster")
    st.pyplot(fig)

    # Menampilkan statistik
    st.markdown("### Statistik Cluster")
    st.write(df_clean.groupby('Cluster').agg({
        'FOB_USD': ['mean', 'std', 'min', 'max'],
        'Qty': ['mean', 'std', 'min', 'max']
    }))

    # Penjelasan untuk user
    st.markdown("""
    ### Penjelasan untuk User:

    **Pie Chart** menunjukkan distribusi persentase jumlah item yang masuk ke dalam masing-masing cluster. Setiap cluster berisi produk dengan karakteristik yang serupa.

    **Bar Chart** menampilkan rata-rata nilai FOB dari produk dalam setiap cluster. Ini memberi gambaran seberapa besar kontribusi ekspor dari masing-masing cluster.

    **Statistik Cluster** menunjukkan informasi lebih detail seperti rata-rata, deviasi standar, nilai minimum, dan maksimum dari nilai FOB dan jumlah ekspor untuk masing-masing cluster.

    Anda dapat menggunakan informasi ini untuk memahami produk mana yang memiliki kontribusi terbesar terhadap nilai ekspor dan produk mana yang membutuhkan perhatian lebih.
    """)
else:
    st.warning("Silakan unggah file CSV atau Excel terlebih dahulu.")
