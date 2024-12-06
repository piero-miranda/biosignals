import streamlit as st

# Configurar el logo en el sidebar
logo_url = "assets/LOGO_1.png"  # Ruta del logo
st.image(logo_url, use_container_width=True)

# Título general en el sidebar
st.title("Biosignals App")

# Bienvenida en el sidebar
st.markdown("""
**¡Bienvenido a Biosignals App!**
Navega por las opciones para explorar las funcionalidades de análisis de señales biomédicas como ECG, EMG y EEG.
""")

st.page_link("pages/1_ECG.py", label="ECG", icon="📈")
st.page_link("pages/2_EMG.py", label="EMG", icon="💪")
st.page_link("pages/3_EEG.py", label="EEG", icon="🧠")
st.page_link("pages/4_Procesar.py", label="Tratamiento de señales", icon="🔬")
st.page_link("pages/5_Convertir.py", label="Conversión TXT a CSV", icon="📂")
