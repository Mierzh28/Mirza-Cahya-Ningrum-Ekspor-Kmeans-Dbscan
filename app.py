import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Set page layout to wide
st.set_page_config(layout="wide")

# Menambahkan judul utama pada halaman
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
    st.write("Berikut adalah data yang diunggah:")
    st.dataframe(df.head())

    # Memeriksa kolom yang ada dalam dataset
    st.write("Nama kolom yang tersedia dalam dataset:")
    st.write(df.columns)  # Menampilkan nama-nama kolom yang ada

    # Membersihkan data untuk analisis
    df['FOB_USD'] = pd.to_numeric(df['FOB_USD'], errors='coerce')
    df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')
    
    # Menangani nilai 0 atau NaN pada kolom yang digunakan untuk clustering
    df['FOB_USD'] = df['FOB_USD'].replace(0, np.nan)
    df['Qty'] = df['Qty'].replace(0, np.nan)
    df = df.dropna(subset=['FOB_USD', 'Qty'])  # Menghapus NaN jika ada

    # Pastikan tidak ada NaN sebelum normalisasi
    if df['FOB_USD'].isnull().any() or df['Qty'].isnull().any():
        st.error("Terdapat nilai NaN dalam kolom 'FOB_USD' atau 'Qty'. Silakan periksa data Anda.")
    else:
        # Preprocessing data untuk clustering
        st.markdown("### Proses Clustering")
        features = ["FOB_USD", "Qty"]
        df_clean = df[features]  # Menggunakan data yang telah dibersihkan

        # Normalisasi data (standarisasi)
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_clean)

        # Menentukan jumlah cluster menggunakan metode Elbow
        inertia = []
        for i in range(1, 11):  # Cek cluster dari 1 sampai 10
            kmeans = KMeans(n_clusters=i, random_state=42)
            kmeans.fit(scaled_features)
            inertia.append(kmeans.inertia_)

        st.markdown("### Menentukan Jumlah Cluster (Optimal K) dengan Elbow Method")
        fig, ax = plt.subplots()
        ax.plot(range(1, 11), inertia, marker='o', linestyle='-', color='b')
        ax.set_title('Elbow Method untuk Menentukan Jumlah Cluster')
        ax.set_xlabel('Jumlah Cluster')
        ax.set_ylabel('Inertia')
        st.pyplot(fig)

        # Menentukan jumlah cluster optimal (misalnya 3) dan melakukan KMeans clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        df_clean['Cluster'] = kmeans.fit_predict(scaled_features)

        # Menampilkan hasil clustering dalam bentuk tabel
        st.write("Hasil Clustering:")
        st.dataframe(df_clean.head())

        # Visualisasi Hasil Clustering
        st.markdown("### Visualisasi Hasil Clustering")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df_clean, x='FOB_USD', y='Qty', hue='Cluster', palette='viridis', s=100)
        ax.set_title("Visualisasi Clustering Perusahaan Berdasarkan Transaksi Ekspor")
        ax.set_xlabel('Nilai FOB (USD)')
        ax.set_ylabel('Jumlah Transaksi')
        plt.legend(title='Cluster', loc='upper right')
        st.pyplot(fig)

        # Evaluasi Hasil Clustering menggunakan Silhouette Score
        sil_score = silhouette_score(scaled_features, df_clean['Cluster'])
        st.write(f'Silhouette Score: {sil_score:.3f}')

        # Visualisasi Perusahaan dengan Transaksi Terbanyak
        st.markdown("### Top 10 Perusahaan dengan Transaksi Terbanyak")
        company_transactions = df_clean['Nama_Perusahaan'].value_counts().reset_index()
        company_transactions.columns = ['Nama_Perusahaan', 'Jumlah_Transaksi']
        top_companies = company_transactions.head(10)

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(x='Jumlah_Transaksi', y='Nama_Perusahaan', data=top_companies, palette='viridis')
        ax.set_title("Top 10 Perusahaan dengan Transaksi Terbanyak")
        ax.set_xlabel('Jumlah Transaksi')
        ax.set_ylabel('Nama Perusahaan')
        plt.tight_layout()
        st.pyplot(fig)

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
