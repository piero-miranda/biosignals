import io
import streamlit as st
import pandas as pd

# Cargar datos desde un archivo subido por el usuario
uploaded_file = st.file_uploader("Sube un archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Leer el archivo CSV
    signal_data = pd.read_csv(uploaded_file)
    st.write('Datos cargados:')
    st.write(signal_data.head())
