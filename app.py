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
    try:
        # Load data
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # Menampilkan data yang diunggah
        st.write("Berikut adalah data yang diunggah:")
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

        # Periksa apakah kolom yang digunakan ada dan bertipe numerik
        if 'FOB_USD' not in df.columns or 'Qty' not in df.columns:
            st.error("Kolom 'FOB_USD' atau 'Qty' tidak ditemukan dalam data. Silakan periksa file Anda.")
        else:
            # Mengganti nilai 0 dengan rata-rata atau nilai lain yang valid
            mean_fob = df['FOB_USD'][df['FOB_USD'] > 0].mean()  # Menghitung rata-rata FOB_USD tanpa 0
            df['FOB_USD'] = df['FOB_USD'].replace(0, mean_fob)  # Ganti 0 dengan rata-rata

            mean_qty = df['Qty'][df['Qty'] > 0].mean()  # Menghitung rata-rata Qty tanpa 0
            df['Qty'] = df['Qty'].replace(0, mean_qty)  # Ganti 0 dengan rata-rata

            # Menghapus nilai yang tidak valid (NaN)
            df_clean = df[['FOB_USD', 'Qty']].dropna()

            # Melanjutkan analisis jika data valid
            if df_clean.empty:
                st.error("Data yang Anda unggah tidak memiliki data yang valid untuk analisis. Pastikan semua data numerik terisi.")
            else:
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

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
else:
    st.warning("Silakan unggah file CSV atau Excel terlebih dahulu.")
