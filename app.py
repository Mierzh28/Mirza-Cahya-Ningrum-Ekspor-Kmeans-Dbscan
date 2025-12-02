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
        df["FOB_USD"] = pd.to_numeric(df["FOB_USD"], errors='coerce')
        df["Qty"] = pd.to_numeric(df["Qty"], errors='coerce')

        # Memeriksa apakah ada nilai NaN di kolom yang digunakan
        if df["FOB_USD"].isnull().sum() > 0 or df["Qty"].isnull().sum() > 0:
            st.warning("Data Anda mengandung nilai yang tida
