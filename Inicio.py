import streamlit as st

# Configurar el logo en el sidebar
full_logo_url = "https://raw.githubusercontent.com/piero-miranda/biosignals/53e8eec9636b4c91bb94b79d4a4c0237b642d171/logo_full.png"
st.image(full_logo_url, use_container_width=True)


# Bienvenida en el sidebar
st.markdown("""
**¡Bienvenido a MedWave!**
Navega por las opciones para explorar las funcionalidades de análisis de señales biomédicas como ECG, EMG y EEG.
""")

st.page_link("pages/1_ECG.py", label="ECG", icon="📈")
st.page_link("pages/2_EMG.py", label="EMG", icon="💪")
st.page_link("pages/3_EEG.py", label="EEG", icon="🧠")
st.page_link("pages/4_Procesar.py", label="Subir una señal propia", icon="🔬")
st.page_link("pages/5_Convertir.py", label="Convertir archivo de BITalino", icon="📂")
