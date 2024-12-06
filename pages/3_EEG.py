import streamlit as st
from funciones import plot_time_domain, extract_features, plot_dwt, plot_spectrogram, plot_stft, plot_cwt
import pandas as pd
import numpy as np

eeg_url = "https://raw.githubusercontent.com/piero-miranda/biosignals/1312455150b97d273907bf574acb0bad7588f830/eeg_final.csv"

# Configurar el logo en el sidebar
short_logo_url = "https://raw.githubusercontent.com/piero-miranda/biosignals/53e8eec9636b4c91bb94b79d4a4c0237b642d171/short_logo.png"
st.sidebar.image(short_logo_url, use_container_width=True)
st.sidebar.markdown("""
**Sobre la señal:**
Configuración Einthoven Lead I con electrodos en las clavículas e iliaca. 
Frecuencia de muestreo: **1000 Hz**.
""")

# Imagen de la posición de EEG en el sidebar
eeg_image_url = "https://raw.githubusercontent.com/piero-miranda/biosignals/c99d4f9dd012ebcc6d3eeb063cd7498f8fa1b0fc/posic_eeg.png"
st.sidebar.image(eeg_image_url, use_container_width=True)

st.sidebar.markdown("""
El participante del estudio hizo la acción de pestañear por intervalos cortos de tiempo.
""")

# Página EEG
def eeg_page():
    st.title('Electroencefalograma (EEG)')
    st.markdown("Técnica que registra la actividad eléctrica cerebral mediante electrodos colocados en el cuero cabelludo, utilizada para evaluar funciones cerebrales y detectar anomalías como epilepsia o trastornos del sueño.")

    eeg_signal = pd.read_csv(eeg_url).iloc[:, 0].values
    sampling_rate = 1000

    start_time = st.slider('Tiempo de inicio (en segundos)', 0.0, len(eeg_signal) / sampling_rate, 11.0, 0.001)
    end_time = st.slider('Tiempo de fin (en segundos)', 0.0, len(eeg_signal) / sampling_rate, 20.0, 0.001)

    filter_signal = st.radio('Seleccione el tipo de señal', ['No Filtrada', 'Filtrada'])

    fig = plot_time_domain(eeg_signal, sampling_rate, "Señal EEG", start_time, end_time, filtered=(filter_signal == 'Filtrada'))
    st.pyplot(fig)

    selected_features = st.pills(
        'Características',
        ['Valor Medio', 'Desviación Estándar', 'Entropía', 'Potencia Media', 'Pico-Amplitud', 'Número de Picos'],
        selection_mode="multi"
    )
    if st.button('Extraer características'):
        features = extract_features(eeg_signal, sampling_rate, start_time, end_time, selected_features, 'EEG')
        for feature, value in features.items():
            st.write(f'{feature}: {value:.4f}')

    if st.checkbox('Mostrar Transformada Wavelet Discreta (DWT)'):
        levels = st.slider('Selecciona el número de niveles de descomposición', 1, 6, 3)
        wavelet = st.selectbox('Selecciona el tipo de wavelet', ['db4', 'haar', 'sym5'])
        fig = plot_dwt(eeg_signal, wavelet=wavelet, levels=levels, fs=sampling_rate)
        st.pyplot(fig)

    if st.checkbox('Mostrar Espectrograma (FFT)'):
        nperseg = st.slider('Segmentos para FFT', 64, 1024, 256)
        noverlap = st.slider('Superposición entre segmentos', 0, nperseg - 1, 128)
        fig = plot_spectrogram(eeg_signal, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
        st.pyplot(fig)

    if st.checkbox('Mostrar STFT'):
        nperseg = st.slider('Segmentos para STFT', 64, 1024, 256)
        noverlap = st.slider('Superposición entre segmentos (STFT)', 0, nperseg - 1, 128)
        fig = plot_stft(eeg_signal, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
        st.pyplot(fig)

    if st.checkbox('Mostrar CWT'):
        wavelet = st.selectbox('Selecciona el tipo de wavelet (CWT)', ['cmor', 'mexh'])
        scales = st.slider('Selecciona las escalas máximas (CWT)', 1, 128, 64)
        fig = plot_cwt(eeg_signal, start_time=start_time, end_time=end_time, wavelet=wavelet, scales=np.arange(1, scales + 1), fs=sampling_rate)
        st.pyplot(fig)
    

if __name__ == "__main__":
    eeg_page()
