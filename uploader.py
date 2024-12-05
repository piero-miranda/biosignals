import os
import pandas as pd
import streamlit as st

file_path = '/workspaces/biosignals/max4.csv'
if os.path.exists(file_path):
    ecg_signal = pd.read_csv(file_path)
else:
    st.error(f"Error: El archivo {file_path} no se encuentra.")
