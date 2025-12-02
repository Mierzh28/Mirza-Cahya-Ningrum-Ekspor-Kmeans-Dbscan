import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set page layout to wide
st.set_page_config(layout="wide")

# Title of the web app
st.title("Analisis Tren Transaksi Ekspor dan Segmentasi Perusahaan Menggunakan Algoritma K-Means Clustering")

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
    st.title("Data Ekspor")
    st.write("Berikut adalah data yang diunggah: ")
    st.dataframe(df.head())

    # Menampilkan penjelasan tentang data
    st.markdown("""
    Data yang diunggah berisi informasi tentang transaksi ekspor produk. Berikut adalah penjelasan untuk beberapa kolom penting:

    - **Tanggal Ekspor**: Tanggal ketika transaksi ekspor dilakukan.
    - **Nomor Aju**: Nomor referensi untuk transaksi.
    - **Nama Perusahaan**: Nama perusahaan yang melakukan transaksi.
    - **Uraian Barang**: Deskripsi barang yang diekspor.
    - **Jumlah (Qty)**: Jumlah barang yang diekspor.
    - **FOB (USD)**: Nilai ekspor dalam mata uang USD.
    
    ***Analisis ini akan mengelompokkan produk berdasarkan nilai FOB dan jumlah transaksi.***
    """)

    # Memeriksa apakah kolom 'FOB_USD' dan 'Qty' ada dan berisi data numerik
    if "FOB_USD" in df.columns and "Qty" in df.columns:
        # Mengonversi kolom 'FOB_USD' dan 'Qty' menjadi numerik, jika tidak bisa dikonversi, akan menjadi NaN
        df["FOB_USD"] = pd.to_numeric(df["FOB_USD"].replace({',': ''}, regex=True), errors='coerce')
        df["Qty"] = pd.to_numeric(df["Qty"].replace({',': ''}, regex=True), errors='coerce')

        # Mengganti nilai NaN dengan 0
        df["FOB_USD"].fillna(0, inplace=True)
        df["Qty"].fillna(0, inplace=True)

        # Memeriksa apakah ada nilai NaN di kolom yang digunakan
        if df["FOB_USD"].isnull().sum() > 0 or df["Qty"].isnull().sum() > 0:
            st.warning("Data Anda mengandung nilai yang tidak valid (NaN) atau tidak sesuai untuk analisis. Silakan periksa kembali data Anda.")
        else:
            # Preprocessing data
            st.markdown("### Proses Clustering")
            
            # Menyiapkan data untuk clustering
            df_clean = df[["FOB_USD", "Qty"]].dropna()  # Remove missing values

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
        st.warning("Pastikan data Anda memiliki kolom 'FOB_USD' dan 'Qty' yang valid.")

else:
    st.warning("Silakan unggah file CSV atau Excel terlebih dahulu.")
