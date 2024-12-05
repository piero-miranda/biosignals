import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Función para cargar la señal desde el archivo CSV
def load_signal(csv_file):
    try:
        data = pd.read_csv(csv_file)
        signal = data.iloc[:, 0].values  # Seleccionar la primera columna directamente
        return signal
    except Exception as e:
        print(f"Error al cargar la señal desde {csv_file}: {e}")
        return None

# Función para graficar la señal en el dominio del tiempo
def plot_time_domain(signal, fs, title, start_time, end_time, filtered=False):
    N = len(signal)
    T = 1.0 / fs
    t = np.linspace(0.0, N * T, N)

    # Seleccionar el intervalo de tiempo
    start_idx = int(start_time * fs)
    end_idx = int(end_time * fs)

    # Subset de la señal
    segment = signal[start_idx:end_idx]

    if filtered:
        # Aplicar filtro pasabajas si el usuario selecciona "Señal filtrada"
        segment = butter_lowpass_filter(segment, cutoff=30, fs=fs, order=4)  # Cambié `cutoff_frequency` por `cutoff`
        title = "Señal Filtrada"

    # Graficar la señal
    plt.figure(figsize=(15, 3))
    plt.plot(t[start_idx:end_idx], segment)
    plt.title(f'Señal en el dominio del tiempo - {title}')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.xlim([start_time, end_time])  # Focalizar en el tramo seleccionado
    return plt

# Función de filtro pasabajas
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs  # Frecuencia de Nyquist
    normal_cutoff = cutoff / nyquist  # Calcular frecuencia normalizada
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # Diseño del filtro
    y = filtfilt(b, a, data)  # Aplicar el filtro pasabajas
    return y

# Función para calcular características de la señal
def extract_features(signal, fs, start_time, end_time, selected_features):
    start_idx = int(start_time * fs)
    end_idx = int(end_time * fs)
    segment = signal[start_idx:end_idx]

    # Características a calcular
    features = {}

    # Valor Medio
    if 'Valor Medio' in selected_features:
        features['Valor Medio'] = np.mean(segment)

    # Desviación Estándar
    if 'Desviación Estándar' in selected_features:
        features['Desviación Estándar'] = np.std(segment)

    # Entropía (entropía de Shannon)
    if 'Entropía' in selected_features:
        entropy = -np.sum(np.log2(np.abs(segment) / np.sum(np.abs(segment))) * (np.abs(segment) / np.sum(np.abs(segment))))
        features['Entropía'] = entropy

    return features

# Página EMG
def emg_page():
    st.title('Electromiograma (EMG)')

    # Ruta del archivo CSV para ECG (ya cargado por defecto)
    file_path = '/workspaces/biosignals/max4.csv'
    ecg_signal = load_signal(file_path)
    sampling_rate = 1000  # 1000 Hz

    # Selección del intervalo de tiempo para graficar
    start_time = st.slider('Tiempo de inicio (en segundos)', 0.0, len(ecg_signal)/sampling_rate, 0.0, 0.001)
    end_time = st.slider('Tiempo de fin (en segundos)', 0.0, len(ecg_signal)/sampling_rate, 10.0, 0.001)

    # Opción de "Señal Filtrada" usando st.radio
    filter_signal = st.radio('Seleccione el tipo de señal', ['No Filtrada', 'Filtrada'])

    # Graficar la señal ECG en el intervalo seleccionado
    fig = plot_time_domain(ecg_signal, sampling_rate, "Señal ECG", start_time, end_time, filtered=(filter_signal == 'Filtrada'))
    st.pyplot(fig)

    # Selección de características para extraer utilizando st.multiselect
    selected_features = st.pills('Características',['Valor Medio', 'Desviación Estándar', 'Entropía'], selection_mode="multi")


    # Extraer características
    if st.button('Extraer características'):
        features = extract_features(ecg_signal, sampling_rate, start_time, end_time, selected_features)
        for feature, value in features.items():
            st.write(f'{feature}: {value:.4f}')

# Página ECG
def ecg_page():
    st.title('Electrocardiograma (ECG)')

    # Ruta del archivo CSV para ECG (ya cargado por defecto)
    file_path = '/workspaces/biosignals/basald1-2.csv'
    ecg_signal = load_signal(file_path)
    sampling_rate = 1000  # 1000 Hz

    # Selección del intervalo de tiempo para graficar
    start_time = st.slider('Tiempo de inicio (en segundos)', 0.0, len(ecg_signal)/sampling_rate, 0.0, 0.001)
    end_time = st.slider('Tiempo de fin (en segundos)', 0.0, len(ecg_signal)/sampling_rate, 10.0, 0.001)

    # Opción de "Señal Filtrada" usando st.radio
    filter_signal = st.radio('Seleccione el tipo de señal', ['No Filtrada', 'Filtrada'])

    # Graficar la señal ECG en el intervalo seleccionado
    fig = plot_time_domain(ecg_signal, sampling_rate, "Señal ECG", start_time, end_time, filtered=(filter_signal == 'Filtrada'))
    st.pyplot(fig)

    # Selección de características para extraer utilizando st.multiselect
    selected_features = st.pills('Características',['Valor Medio', 'Desviación Estándar', 'Entropía'], selection_mode="multi")


    # Extraer características
    if st.button('Extraer características'):
        features = extract_features(ecg_signal, sampling_rate, start_time, end_time, selected_features)
        for feature, value in features.items():
            st.write(f'{feature}: {value:.4f}')

# Página Tratamiento de Señales
def signal_treatment_page():
    st.title('Tratamiento de Señales')

    # Subir archivo CSV
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
    if uploaded_file:
        signal_data = load_signal(uploaded_file)
        sampling_rate = 1000  # Puedes ajustar esto según el archivo subido
        st.write(f"Se cargaron {len(signal_data)} muestras.")
        
        # Selección de intervalo de tiempo para graficar
        start_time = st.slider('Tiempo de inicio (en segundos)', 0.0, len(signal_data)/sampling_rate, 0.0, 0.001)
        end_time = st.slider('Tiempo de fin (en segundos)', 0.0, len(signal_data)/sampling_rate, 10.0, 0.001)

        # Opción de "Señal Filtrada"
        filter_signal = st.radio('Seleccione el tipo de señal', ['No Filtrada', 'Filtrada'])

        # Graficar la señal en el intervalo seleccionado
        fig = plot_time_domain(signal_data, sampling_rate, "Señal", start_time, end_time, filtered=(filter_signal == 'Filtrada'))
        st.pyplot(fig)

        # Selección de características para extraer utilizando st.multiselect
        selected_features = st.pills('Características',['Valor Medio', 'Desviación Estándar', 'Entropía'], selection_mode="multi")

        # Extraer características
        if st.button('Extraer características'):
            features = extract_features(signal_data, sampling_rate, start_time, end_time, selected_features)
            for feature, value in features.items():
                st.write(f'{feature}: {value:.4f}')

# Menú lateral para navegar entre las páginas
menu = st.sidebar.radio("Selecciona una página", ["EMG", "ECG", "Tratamiento de señales"])

if menu == "EMG":
    emg_page()
elif menu == "ECG":
    ecg_page()
else:
    signal_treatment_page()
