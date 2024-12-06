import streamlit as st
import pandas as pd
import numpy as np
from funciones import plot_time_domain, extract_features, plot_dwt, plot_spectrogram, plot_stft, plot_cwt

# Configurar el logo en el sidebar
logo_url = "assets/LOGO_1.png"  # Ruta del logo
st.sidebar.image(logo_url, use_container_width=True)

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
        fig = plot_dwt(ecg_signal, wavelet=wavelet, levels=levels, fs=sampling_rate)
        st.pyplot(fig)

    if st.checkbox('Mostrar Espectrograma (FFT)'):
        nperseg = st.slider('Segmentos para FFT', 64, 1024, 256)
        noverlap = st.slider('Superposición entre segmentos', 0, nperseg - 1, 128)
        fig = plot_spectrogram(ecg_signal, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
        st.pyplot(fig)
    
    if st.checkbox('Mostrar STFT'):
        nperseg = st.slider('Segmentos para STFT', 64, 1024, 256)
        noverlap = st.slider('Superposición entre segmentos (STFT)', 0, nperseg - 1, 128)
        fig = plot_stft(ecg_signal, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
        st.pyplot(fig)

    if st.checkbox('Mostrar CWT'):
        wavelet = st.selectbox('Selecciona el tipo de wavelet (CWT)', ['cmor', 'mexh'])
        scales = st.slider('Selecciona las escalas máximas (CWT)', 1, 128, 64)
        fig = plot_cwt(ecg_signal, wavelet=wavelet, scales=np.arange(1, scales + 1), fs=sampling_rate)
        st.pyplot(fig)

if __name__ == "__main__":
    ecg_page()
