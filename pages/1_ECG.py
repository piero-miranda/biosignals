import streamlit as st
import pandas as pd
import numpy as np
from funciones import plot_time_domain, extract_features, plot_dwt, plot_psd, plot_stft, plot_cwt, plot_fft

# Configurar el logo en el sidebar
short_logo_url = "https://raw.githubusercontent.com/piero-miranda/biosignals/53e8eec9636b4c91bb94b79d4a4c0237b642d171/short_logo.png"
st.sidebar.image(short_logo_url, use_container_width=True)
st.sidebar.markdown("""
**Sobre la señal:**
Se colocaron electrodos en la posición FP2 (según el sistema internacional 10-20), permitiendo medir actividad de ondas beta. 
Frecuencia de muestreo: **1000 Hz**.
""")

# Imagen de la posición de ECG en el sidebar
ecg_image_url = "https://raw.githubusercontent.com/piero-miranda/biosignals/c99d4f9dd012ebcc6d3eeb063cd7498f8fa1b0fc/posic_ecg.png"
st.sidebar.image(ecg_image_url, use_container_width=True)

st.sidebar.markdown("""
El participante del estudio se mantuvo en estado basal durante toda la grabación de la señal.
""")

ecg_url = "https://raw.githubusercontent.com/piero-miranda/biosignals/501a52199bd6310d678dde38157c885d15ee1b51/basald1-2.csv"

def ecg_page():
    st.title('Electrocardiograma (ECG)')
    st.markdown("Método que registra la actividad eléctrica del corazón a lo largo del tiempo para analizar su ritmo y detectar condiciones como arritmias o enfermedades cardíacas.")

    ecg_signal = pd.read_csv(ecg_url).iloc[:, 0].values
    sampling_rate = 1000

    start_time = st.slider('Tiempo de inicio (en segundos)', 0.0, len(ecg_signal) / sampling_rate, 0.0, 0.001)
    end_time = st.slider('Tiempo de fin (en segundos)', 0.0, len(ecg_signal) / sampling_rate, 10.0, 0.001)

    filter_signal = st.radio('Seleccione el tipo de señal', ['No Filtrada', 'Filtrada'])

    fig = plot_time_domain(ecg_signal, sampling_rate, "Señal ECG", start_time, end_time, filtered=(filter_signal == 'Filtrada'))
    st.pyplot(fig)

    selected_features = st.pills('Características', ['Valor Medio', 'Desviación Estándar', 'Entropía', 'Frecuencia Cardíaca (bpm)', 'SDNN'], selection_mode="multi")
    if st.button('Extraer características'):
        features = extract_features(ecg_signal, sampling_rate, start_time, end_time, selected_features, 'ECG')
        for feature, value in features.items():
            st.write(f'{feature}: {value:.4f}')

    if st.checkbox('Mostrar Transformada Wavelet Discreta (DWT)'):
        levels = st.slider('Selecciona el número de niveles de descomposición', 1, 6, 3)
        wavelet = st.selectbox('Selecciona el tipo de wavelet', ['db4', 'haar', 'sym5'])
        fig = plot_dwt(ecg_signal, wavelet=wavelet, levels=levels, fs=sampling_rate, start_time=start_time, end_time=end_time)
        st.pyplot(fig)

    if st.checkbox('Mostrar FFT (dB vs Frecuencia)'):
        freq_unit = st.radio("Selecciona la unidad de frecuencia", ["Hz", "rad/s"])
        fig = plot_fft(ecg_signal, fs=sampling_rate, start_time=start_time, end_time=end_time, freq_unit=freq_unit)
        st.pyplot(fig)

    if st.checkbox('Mostrar Densidad Espectral de Potencia (PSD)'):
        fig = plot_psd(ecg_signal, fs=sampling_rate, start_time=start_time, end_time=end_time)
        st.pyplot(fig)

    if st.checkbox('Mostrar STFT'):
        nperseg = st.slider('Segmentos para STFT', 64, 1024, 256)
        noverlap = st.slider('Superposición entre segmentos (STFT)', 0, nperseg - 1, 128)
        fig = plot_stft(ecg_signal, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap, start_time=start_time, end_time=end_time)
        st.pyplot(fig)

    if st.checkbox('Mostrar CWT'):
        wavelet = st.selectbox('Selecciona el tipo de wavelet (CWT)', ['cmor', 'mexh'])
        scales = st.slider('Selecciona las escalas máximas (CWT)', 1, 128, 64)
        fig = plot_cwt(ecg_signal, start_time=start_time, end_time=end_time, wavelet=wavelet, scales=np.arange(1, scales + 1), fs=sampling_rate)
        st.pyplot(fig)

if __name__ == "__main__":
    ecg_page()
