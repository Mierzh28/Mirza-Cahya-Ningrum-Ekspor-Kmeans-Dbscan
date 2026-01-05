import streamlit as st
import pandas as pd

st.set_page_config(page_title="CEISA Segmentation", layout="wide")

st.title("Analisis Pola Transaksi Kepabeanan untuk Segmentasi Keaktifan Perusahaan Mitra")
st.caption("Perbandingan K-Means vs DBSCAN")

st.sidebar.header("Input")

uploaded = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
run = st.sidebar.button("Run")

if uploaded is None:
    st.info("Upload file Excel di sidebar, lalu klik Run.")
    st.stop()

df = pd.read_excel(uploaded)

st.subheader("Preview Data")
st.write(df.head(20))
