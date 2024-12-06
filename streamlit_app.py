import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram, find_peaks
import pywt

# URLs del permalink de GitHub (raw)
ecg_url = "https://raw.githubusercontent.com/piero-miranda/biosignals/501a52199bd6310d678dde38157c885d15ee1b51/basald1-2.csv"
emg_url = "https://raw.githubusercontent.com/piero-miranda/biosignals/501a52199bd6310d678dde38157c885d15ee1b51/max4.csv"


# Filtro pasabajas
def butter_lowpass_filter(data, cutoff, fs, order=4):
    padlen = 3 * order
    if len(data) <= padlen:
        st.warning(f"El segmento es demasiado corto para filtrar. Se necesitan al menos {padlen + 1} muestras.")
        return data

    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    try:
        y = filtfilt(b, a, data)
    except ValueError as e:
        st.error(f"Error al aplicar el filtro: {e}")
        return data
    return y

# Gráfico de dominio del tiempo
def plot_time_domain(signal, fs, title, start_time, end_time, filtered=False):
    N = len(signal)
    T = 1.0 / fs
    t = np.linspace(0.0, N * T, N)

    start_idx = int(start_time * fs)
    end_idx = int(end_time * fs)

    segment = signal[start_idx:end_idx]

    padlen = 3 * 4
    if len(segment) <= padlen:
        st.warning(f"El intervalo seleccionado es demasiado corto para procesar el filtro. Se necesitan al menos {padlen + 1} muestras.")
        filtered = False

    if filtered:
        segment = butter_lowpass_filter(segment, cutoff=30, fs=fs, order=4)
        title = "Señal Filtrada"

    fig, ax = plt.subplots(figsize=(15, 3))
    ax.plot(t[start_idx:end_idx], segment)
    ax.set_title(f'Señal en el dominio del tiempo - {title}')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Amplitud')
    ax.grid(True)
    ax.set_xlim([start_time, end_time])
    return fig

# Cálculo de características adicionales de EMG
def calculate_emg_features(segment, selected_features):
    features = {}
    if 'MAV' in selected_features:
        features['MAV'] = np.mean(np.abs(segment))
    if 'VAR' in selected_features:
        features['VAR'] = np.var(segment)
    if 'ZCR' in selected_features:
        zero_crossings = np.where(np.diff(np.sign(segment)))[0]
        features['ZCR'] = len(zero_crossings) / len(segment)
    if 'SSC' in selected_features:
        slope_changes = np.where(np.diff(np.sign(np.diff(segment))))[0]
        features['SSC'] = len(slope_changes) / len(segment)
    return features

# Cálculo de características adicionales de ECG
def calculate_ecg_features(segment, fs, selected_features):
    features = {}
    if 'Frecuencia Cardíaca (bpm)' in selected_features or 'SDNN' in selected_features:
        peaks, _ = find_peaks(segment, distance=fs * 0.6)
        rr_intervals = np.diff(peaks) / fs

        if len(rr_intervals) > 0:
            if 'Frecuencia Cardíaca (bpm)' in selected_features:
                features['Frecuencia Cardíaca (bpm)'] = 60 / np.mean(rr_intervals)
            if 'SDNN' in selected_features:
                features['SDNN'] = np.std(rr_intervals)
        else:
            if 'Frecuencia Cardíaca (bpm)' in selected_features:
                features['Frecuencia Cardíaca (bpm)'] = np.nan
            if 'SDNN' in selected_features:
                features['SDNN'] = np.nan
    return features

# Gráfico del espectrograma
def plot_spectrogram(signal, fs=1000, nperseg=256, noverlap=128):
    f, t, Sxx = spectrogram(signal, fs, nperseg=nperseg, noverlap=noverlap)
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
    fig.colorbar(cax, ax=ax, label='Densidad de potencia (dB)')
    ax.set_ylabel('Frecuencia (Hz)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_title('Espectrograma')
    return fig

# Gráfico de DWT
def plot_dwt(signal, wavelet='db4', levels=3, fs=1000):
    coeffs = pywt.wavedec(signal, wavelet, level=levels)
    fig, axs = plt.subplots(levels + 1, 1, figsize=(10, 8))
    fig.suptitle('Transformada Wavelet Discreta (DWT)', fontsize=16)

    for i, coeff in enumerate(coeffs):
        t = np.linspace(0, len(coeff) / fs, len(coeff))
        if i == 0:
            axs[i].plot(t, coeff, label="Aproximación")
            axs[i].set_title("Aproximación (A)")
        else:
            axs[i].plot(t, coeff, label=f"Detalle D{i}")
            axs[i].set_title(f"Detalle (D{i})")
        axs[i].legend()
        axs[i].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

# Gráfico del espectrograma
def plot_spectrogram(signal, fs=1000, nperseg=256, noverlap=128):
    f, t, Sxx = spectrogram(signal, fs, nperseg=nperseg, noverlap=noverlap)
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
    fig.colorbar(cax, ax=ax, label='Densidad de potencia (dB)')
    ax.set_ylabel('Frecuencia (Hz)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_title('Espectrograma')
    return fig

# Función para calcular todas las características
def extract_features(signal, fs, start_time, end_time, selected_features, signal_type):
    start_idx = int(start_time * fs)
    end_idx = int(end_time * fs)
    segment = signal[start_idx:end_idx]

    features = {}
    if 'Valor Medio' in selected_features:
        features['Valor Medio'] = np.mean(segment)
    if 'Desviación Estándar' in selected_features:
        features['Desviación Estándar'] = np.std(segment)
    if 'Entropía' in selected_features:
        entropy = -np.sum(np.log2(np.abs(segment) / np.sum(np.abs(segment))) * (np.abs(segment) / np.sum(np.abs(segment))))
        features['Entropía'] = entropy

    if signal_type == 'EMG':
        features.update(calculate_emg_features(segment, selected_features))
    elif signal_type == 'ECG':
        features.update(calculate_ecg_features(segment, fs, selected_features))

    return features

# Página EMG
def emg_page():
    st.title('Electromiograma (EMG)')
    st.markdown("Técnica que mide la actividad eléctrica generada por los músculos durante la contracción o el reposo, utilizada para evaluar su función y diagnosticar trastornos neuromusculares.")

    emg_signal = pd.read_csv(emg_url).iloc[:, 0].values
    sampling_rate = 1000

    start_time = st.slider('Tiempo de inicio (en segundos)', 0.0, len(emg_signal) / sampling_rate, 0.0, 0.001)
    end_time = st.slider('Tiempo de fin (en segundos)', 0.0, len(emg_signal) / sampling_rate, 10.0, 0.001)

    filter_signal = st.radio('Seleccione el tipo de señal', ['No Filtrada', 'Filtrada'])

    fig = plot_time_domain(emg_signal, sampling_rate, "Señal EMG", start_time, end_time, filtered=(filter_signal == 'Filtrada'))
    st.pyplot(fig)

    selected_features = st.pills('Características', ['Valor Medio', 'Desviación Estándar', 'Entropía', 'MAV', 'VAR', 'ZCR', 'SSC'], selection_mode="multi")
    if st.button('Extraer características'):
        features = extract_features(emg_signal, sampling_rate, start_time, end_time, selected_features, 'EMG')
        for feature, value in features.items():
            st.write(f'{feature}: {value:.4f}')

    if st.checkbox('Mostrar Transformada Wavelet Discreta (DWT)'):
        levels = st.slider('Selecciona el número de niveles de descomposición', 1, 6, 3)
        wavelet = st.selectbox('Selecciona el tipo de wavelet', ['db4', 'haar', 'sym5'])
        fig = plot_dwt(emg_signal, wavelet=wavelet, levels=levels, fs=sampling_rate)
        st.pyplot(fig)

    if st.checkbox('Mostrar Espectrograma (FFT)'):
        nperseg = st.slider('Segmentos para FFT', 64, 1024, 256)
        noverlap = st.slider('Superposición entre segmentos', 0, nperseg - 1, 128)
        fig = plot_spectrogram(emg_signal, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
        st.pyplot(fig)

# Página ECG
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

def signal_treatment_page():
    st.title('Tratamiento de Señales')

    # Subir archivo CSV
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if uploaded_file:
        signal_data = pd.read_csv(uploaded_file).iloc[:, 0].values
        sampling_rate = 1000
        st.write(f"Se cargaron {len(signal_data)} muestras.")
        
        # Seleccionar el tipo de señal
        signal_type = st.selectbox("Seleccione el tipo de señal", ["ECG", "EMG"])
        
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


# Menú lateral
menu = st.sidebar.radio("Selecciona una página", ["ECG", "EMG", "Tratamiento de señales"])
if menu == "ECG":
    ecg_page()
elif menu == "EMG":
    emg_page()
else:
    signal_treatment_page()
