import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# URLs del permalink de GitHub (raw)
ecg_url = "https://raw.githubusercontent.com/piero-miranda/biosignals/501a52199bd6310d678dde38157c885d15ee1b51/basald1-2.csv"
emg_url = "https://raw.githubusercontent.com/piero-miranda/biosignals/501a52199bd6310d678dde38157c885d15ee1b51/max4.csv"

# Cargar los archivos CSV desde los permalinks


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

    # Validar longitud mínima del segmento antes de aplicar el filtro
    padlen = 3 * 4  # 3 * order del filtro
    if len(segment) <= padlen:
        st.warning(f"El intervalo seleccionado es demasiado corto para procesar el filtro. Se necesitan al menos {padlen + 1} muestras.")
        filtered = False  # Evitar filtrado

    if filtered:
        # Aplicar filtro pasabajas
        segment = butter_lowpass_filter(segment, cutoff=30, fs=fs, order=4)
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

from scipy.signal import butter, filtfilt
def butter_lowpass_filter(data, cutoff, fs, order=4):
    # Verificar si la longitud del segmento es suficiente
    padlen = 3 * order  # Este es el valor mínimo necesario para filtrar correctamente
    if len(data) <= padlen:
        st.warning(f"El segmento es demasiado corto para filtrar. Se necesitan al menos {padlen + 1} muestras.")
        return data  # Devolver la señal sin filtrar

    nyquist = 0.5 * fs  # Frecuencia de Nyquist
    normal_cutoff = cutoff / nyquist  # Frecuencia normalizada

    # Coefficients del filtro pasabajas
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    # Intentar aplicar el filtro
    try:
        y = filtfilt(b, a, data)
    except ValueError as e:
        st.error(f"Error al aplicar el filtro: {e}")
        return data  # Devolver la señal sin filtrar en caso de error

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

def emg_page():
    st.title('Electromiograma (EMG)')

    # Cargar los datos desde la URL
    emg_signal = pd.read_csv(emg_url).iloc[:, 0].values  # Usar solo la primera columna
    sampling_rate = 1000  # 1000 Hz

    # Selección del intervalo de tiempo
    start_time = st.slider('Tiempo de inicio (en segundos)', 0.0, len(emg_signal)/sampling_rate, 0.0, 0.001)
    end_time = st.slider('Tiempo de fin (en segundos)', 0.0, len(emg_signal)/sampling_rate, 10.0, 0.001)

    # Validar longitud mínima del intervalo
    min_interval = 0.02  # 20 ms mínimo
    if end_time - start_time < min_interval:
        st.warning(f"El intervalo seleccionado es muy corto. Ajustando a un mínimo de {min_interval} segundos.")
        end_time = start_time + min_interval

    # Opción de "Señal Filtrada"
    filter_signal = st.radio('Seleccione el tipo de señal', ['No Filtrada', 'Filtrada'])

    # Graficar la señal
    fig = plot_time_domain(emg_signal, sampling_rate, "Señal EMG", start_time, end_time, filtered=(filter_signal == 'Filtrada'))
    if fig:
        st.pyplot(fig)

    # Características
    selected_features = st.pills('Características',['Valor Medio', 'Desviación Estándar', 'Entropía'], selection_mode="multi")
    if st.button('Extraer características'):
        features = extract_features(emg_signal, sampling_rate, start_time, end_time, selected_features)
        for feature, value in features.items():
            st.write(f'{feature}: {value:.4f}')


def ecg_page():
    st.title('Electrocardiograma (ECG)')

    # Cargar los datos desde la URL
    ecg_signal = pd.read_csv(ecg_url).iloc[:, 0].values  # Usar solo la primera columna
    sampling_rate = 1000  # 1000 Hz

    # Selección del intervalo de tiempo
    start_time = st.slider('Tiempo de inicio (en segundos)', 0.0, len(ecg_signal)/sampling_rate, 0.0, 0.001)
    end_time = st.slider('Tiempo de fin (en segundos)', 0.0, len(ecg_signal)/sampling_rate, 10.0, 0.001)

    # Validar longitud mínima del intervalo
    min_interval = 0.02  # 20 ms mínimo
    if end_time - start_time < min_interval:
        st.warning(f"El intervalo seleccionado es muy corto. Ajustando a un mínimo de {min_interval} segundos.")
        end_time = start_time + min_interval

    # Opción de "Señal Filtrada"
    filter_signal = st.radio('Seleccione el tipo de señal', ['No Filtrada', 'Filtrada'])

    # Graficar la señal
    fig = plot_time_domain(ecg_signal, sampling_rate, "Señal ECG", start_time, end_time, filtered=(filter_signal == 'Filtrada'))
    if fig:
        st.pyplot(fig)

    # Características
    selected_features = st.pills('Características',['Valor Medio', 'Desviación Estándar', 'Entropía'], selection_mode="multi")

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
menu = st.sidebar.radio("Selecciona una página", ["ECG", "EMG", "Tratamiento de señales"])

if menu == "ECG":
    ecg_page()
elif menu == "EMG":
    emg_page()
else:
    signal_treatment_page()
