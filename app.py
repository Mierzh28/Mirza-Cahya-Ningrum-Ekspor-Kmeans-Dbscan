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

        # Mengonversi kolom yang harus numerik
        df['FOB_USD'] = pd.to_numeric(df['FOB_USD'], errors='coerce')
        df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')

        # Periksa jika ada nilai NaN setelah konversi
        if df['FOB_USD'].isnull().any() or df['Qty'].isnull().any():
            st.warning("Ada nilai yang tidak valid (NaN) pada kolom 'FOB_USD' atau 'Qty'. Nilai ini akan diganti dengan rata-rata.")

            # Ganti NaN dengan rata-rata kolom
            df['FOB_USD'].fillna(df['FOB_USD'].mean(), inplace=True)
            df['Qty'].fillna(df['Qty'].mean(), inplace=True)

        # Menghapus baris
