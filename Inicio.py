import streamlit as st

# Configurar el logo en el sidebar
logo_main_url = "https://raw.githubusercontent.com/piero-miranda/biosignals/ea4531f80608cf0936c142e224aba9844a40b4aa/logo_final.png"
st.image(logo_main_url, use_container_width=True)


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
