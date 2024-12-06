import streamlit as st
from funciones import plot_time_domain, extract_features, plot_dwt, plot_spectrogram, plot_stft, plot_cwt
import numpy as np
import pandas as pd

# Configurar el logo en el sidebar
logo_url = "https://raw.githubusercontent.com/piero-miranda/biosignals/ea4531f80608cf0936c142e224aba9844a40b4aa/short_logo.png"
st.sidebar.image(logo_url, use_container_width=True)

# Página Tratamiento de Señales
def signal_treatment_page():
    st.title('Tratamiento de Señales')
    st.markdown(
        """
        **Instrucciones para subir un archivo:**
        - El archivo debe estar en formato CSV.
        - El archivo debe contener una sola columna con los valores de la señal.
        - No se requiere una columna de timestamp.
        """
    )

    # Subir archivo CSV
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if uploaded_file:
        # Solicitar la frecuencia de muestreo
        sampling_rate = st.number_input(
            "Ingrese la frecuencia de muestreo (Hz):",
            min_value=1,
            max_value=100000,
            value=1000,
            step=1,
            help="Frecuencia de muestreo utilizada para capturar la señal."
        )

        # Leer y mostrar información del archivo subido
        try:
            signal_data = pd.read_csv(uploaded_file).iloc[:, 0].values
            st.write(f"Se cargaron {len(signal_data)} muestras.")
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            return

        # Seleccionar el tipo de señal
        signal_type = st.selectbox("Seleccione el tipo de señal", ["ECG", "EMG", "EEG"])

        # Selección de intervalo de tiempo para graficar
        start_time = st.slider('Tiempo de inicio (en segundos)', 0.0, len(signal_data) / sampling_rate, 0.0, 0.001)
        end_time = st.slider('Tiempo de fin (en segundos)', 0.0, len(signal_data) / sampling_rate, 10.0, 0.001)

        # Opción de "Señal Filtrada"
        filter_signal = st.radio('Seleccione el tipo de señal', ['No Filtrada', 'Filtrada'])

        # Graficar la señal en el intervalo seleccionado
        fig = plot_time_domain(signal_data, sampling_rate, "Señal", start_time, end_time, filtered=(filter_signal == 'Filtrada'))
        st.pyplot(fig)

        # Características según el tipo de señal
        if signal_type == "EMG":
            selected_features = st.pills(
                'Características',
                ['Valor Medio', 'Desviación Estándar', 'Entropía', 'MAV', 'VAR', 'ZCR', 'SSC'],
                selection_mode="multi"
            )
        elif signal_type == "ECG":
            selected_features = st.pills(
                'Características',
                ['Valor Medio', 'Desviación Estándar', 'Entropía', 'Frecuencia Cardíaca (bpm)', 'SDNN'],
                selection_mode="multi"
            )
        elif signal_type == "EEG":
            selected_features = st.pills(
                'Características',
                ['Valor Medio', 'Desviación Estándar', 'Entropía', 'Potencia Media', 'Pico-Amplitud', 'Número de Picos'],
                selection_mode="multi"
            )

        # Extraer características
        if st.button('Extraer características'):
            features = extract_features(signal_data, sampling_rate, start_time, end_time, selected_features, signal_type)
            for feature, value in features.items():
                st.write(f'{feature}: {value:.4f}')

        # Opcional: Mostrar transformada wavelet discreta (DWT)
        if st.checkbox('Mostrar Transformada Wavelet Discreta (DWT)'):
            levels = st.slider('Selecciona el número de niveles de descomposición', 1, 6, 3)
            wavelet = st.selectbox('Selecciona el tipo de wavelet', ['db4', 'haar', 'sym5'])
            fig = plot_dwt(signal_data, wavelet=wavelet, levels=levels, fs=sampling_rate)
            st.pyplot(fig)

        # Opcional: Mostrar espectrograma
        if st.checkbox('Mostrar Espectrograma (FFT)'):
            nperseg = st.slider('Segmentos para FFT', 64, 1024, 256)
            noverlap = st.slider('Superposición entre segmentos', 0, nperseg - 1, 128)
            fig = plot_spectrogram(signal_data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
            st.pyplot(fig)
        
        if st.checkbox('Mostrar STFT'):
            nperseg = st.slider('Segmentos para STFT', 64, 1024, 256)
            noverlap = st.slider('Superposición entre segmentos (STFT)', 0, nperseg - 1, 128)
            fig = plot_stft(signal_data, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
            st.pyplot(fig)

        if st.checkbox('Mostrar CWT'):
            wavelet = st.selectbox('Selecciona el tipo de wavelet (CWT)', ['cmor', 'mexh'])
            scales = st.slider('Selecciona las escalas máximas (CWT)', 1, 128, 64)
            fig = plot_cwt(signal_data, wavelet=wavelet, scales=np.arange(1, scales + 1), fs=sampling_rate)
            st.pyplot(fig)

# Renderizar la página automáticamente si se ejecuta directamente
if __name__ == "__main__":
    signal_treatment_page()