import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram, find_peaks
import pywt
from scipy.signal import stft
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

# URLs del permalink de GitHub (raw)
ecg_url = "https://raw.githubusercontent.com/piero-miranda/biosignals/501a52199bd6310d678dde38157c885d15ee1b51/basald1-2.csv"
emg_url = "https://raw.githubusercontent.com/piero-miranda/biosignals/501a52199bd6310d678dde38157c885d15ee1b51/max4.csv"
eeg_url = "https://raw.githubusercontent.com/piero-miranda/biosignals/1312455150b97d273907bf574acb0bad7588f830/eeg_final.csv"

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
    ax.set_ylabel('Amplitud (mV)')
    ax.grid(True)
    ax.set_xlim([start_time, end_time])
    return fig

def plot_spectrogram(signal, fs=1000, nperseg=256, noverlap=128, start_time=None, end_time=None):
    """
    Graficar espectrograma de una señal, considerando un intervalo de tiempo.
    Parameters:
    - signal: Señal de entrada.
    - fs: Frecuencia de muestreo.
    - nperseg: Número de puntos por segmento.
    - noverlap: Superposición entre segmentos.
    - start_time: Tiempo de inicio (en segundos).
    - end_time: Tiempo de fin (en segundos).
    """
    # Extraer segmento de interés
    if start_time is not None and end_time is not None:
        start_idx = int(start_time * fs)
        end_idx = int(end_time * fs)
        signal = signal[start_idx:end_idx]
    
    # Generar el espectrograma
    f, t, Sxx = spectrogram(signal, fs, nperseg=nperseg, noverlap=noverlap)
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
    fig.colorbar(cax, ax=ax, label='Densidad de potencia (dB)')
    ax.set_ylabel('Frecuencia (Hz)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_title('Espectrograma')
    return fig


def plot_stft(signal, fs=1000, nperseg=256, noverlap=128, start_time=None, end_time=None):
    """
    Graficar STFT de una señal, considerando un intervalo de tiempo.
    Parameters:
    - signal: Señal de entrada.
    - fs: Frecuencia de muestreo.
    - nperseg: Número de puntos por segmento.
    - noverlap: Superposición entre segmentos.
    - start_time: Tiempo de inicio (en segundos).
    - end_time: Tiempo de fin (en segundos).
    """
    # Extraer segmento de interés
    if start_time is not None and end_time is not None:
        start_idx = int(start_time * fs)
        end_idx = int(end_time * fs)
        signal = signal[start_idx:end_idx]
    
    # Generar STFT
    f, t, Zxx = stft(signal, fs, nperseg=nperseg, noverlap=noverlap)
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='viridis')
    fig.colorbar(cax, ax=ax, label='Amplitud')
    ax.set_ylabel('Frecuencia (Hz)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_title('STFT (Transformada de Fourier de Tiempo Corto)')
    return fig


def plot_dwt(signal, wavelet='db4', levels=3, fs=1000, start_time=None, end_time=None):
    """
    Graficar DWT de una señal, considerando un intervalo de tiempo.
    Parameters:
    - signal: Señal de entrada.
    - wavelet: Tipo de wavelet.
    - levels: Número de niveles de descomposición.
    - fs: Frecuencia de muestreo.
    - start_time: Tiempo de inicio (en segundos).
    - end_time: Tiempo de fin (en segundos).
    """
    # Extraer segmento de interés
    if start_time is not None and end_time is not None:
        start_idx = int(start_time * fs)
        end_idx = int(end_time * fs)
        signal = signal[start_idx:end_idx]
    
    # Calcular la DWT
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


def plot_cwt(signal, start_time, end_time, wavelet='cmor', scales=None, fs=1000):
    """
    Graficar CWT de un segmento de señal.
    
    Parameters:
    - signal: Señal de entrada.
    - start_time: Tiempo de inicio del segmento (en segundos).
    - end_time: Tiempo de fin del segmento (en segundos).
    - wavelet: Tipo de wavelet (por ejemplo, 'cmor', 'mexh').
    - scales: Escalas para la transformada wavelet.
    - fs: Frecuencia de muestreo.
    """
    import pywt

    if scales is None:
        scales = np.arange(1, 128)

    # Seleccionar el segmento de la señal
    start_idx = int(start_time * fs)
    end_idx = int(end_time * fs)
    segment = signal[start_idx:end_idx]

    # Calcular la CWT del segmento
    coefficients, frequencies = pywt.cwt(segment, scales, wavelet, 1 / fs)

    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.pcolormesh(
        np.arange(len(segment)) / fs + start_time,  # Ajustar el eje de tiempo al segmento
        frequencies,
        np.abs(coefficients),
        shading='gouraud',
        cmap='viridis'
    )
    fig.colorbar(cax, ax=ax, label='Amplitud')
    ax.set_ylabel('Frecuencia (Hz)')
    ax.set_xlabel('Tiempo (s)')
    ax.set_title('CWT (Transformada Wavelet Continua)')
    return fig


# Cálculo de características adicionales de EMG
def calculate_emg_features(segment, selected_features):
    features = {}
    if 'MAV' in selected_features:
        features['MAV'] = np.mean(np.abs(segment))  # Media de la amplitud rectificada
    if 'VAR' in selected_features:
        features['VAR'] = np.var(segment)  # Varianza de la señal
    if 'ZCR' in selected_features:
        zero_crossings = np.where(np.diff(np.sign(segment)))[0]
        features['ZCR'] = len(zero_crossings) / len(segment)  # Tasa de cruces por cero
    if 'SSC' in selected_features:
        slope_changes = np.where(np.diff(np.sign(np.diff(segment))))[0]
        features['SSC'] = len(slope_changes) / len(segment)  # Cambios significativos en la pendiente
    return features

# Cálculo de características adicionales de ECG
def calculate_ecg_features(segment, fs, selected_features):
    features = {}
    if 'Frecuencia Cardíaca (bpm)' in selected_features or 'SDNN' in selected_features:
        peaks, _ = find_peaks(segment, distance=fs * 0.6)  # Detectar picos R
        rr_intervals = np.diff(peaks) / fs  # Intervalos R-R en segundos

        if len(rr_intervals) > 0:
            if 'Frecuencia Cardíaca (bpm)' in selected_features:
                features['Frecuencia Cardíaca (bpm)'] = 60 / np.mean(rr_intervals)  # Calcular BPM
            if 'SDNN' in selected_features:
                features['SDNN'] = np.std(rr_intervals)  # Desviación estándar de los intervalos R-R
        else:
            if 'Frecuencia Cardíaca (bpm)' in selected_features:
                features['Frecuencia Cardíaca (bpm)'] = np.nan
            if 'SDNN' in selected_features:
                features['SDNN'] = np.nan
    return features

# Cálculo de características adicionales de EEG
def calculate_eeg_features(segment, selected_features):
    features = {}
    if 'Potencia Media' in selected_features:
        features['Potencia Media'] = np.mean(segment**2)  # Potencia media
    if 'Pico-Amplitud' in selected_features:
        features['Pico-Amplitud'] = np.max(segment)  # Amplitud máxima
    if 'Número de Picos' in selected_features:
        peaks, _ = find_peaks(segment, height=0)  # Detectar picos
        features['Número de Picos'] = len(peaks)  # Número total de picos
    return features

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

    # Otras características específicas por tipo
    if signal_type == 'EMG':
        features.update(calculate_emg_features(segment, selected_features))
    elif signal_type == 'ECG':
        features.update(calculate_ecg_features(segment, fs, selected_features))
    elif signal_type == 'EEG':
        features.update(calculate_eeg_features(segment, selected_features))

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
    
    if st.checkbox('Mostrar STFT'):
        nperseg = st.slider('Segmentos para STFT', 64, 1024, 256)
        noverlap = st.slider('Superposición entre segmentos (STFT)', 0, nperseg - 1, 128)
        fig = plot_stft(emg_signal, fs=sampling_rate, nperseg=nperseg, noverlap=noverlap)
        st.pyplot(fig)

    if st.checkbox('Mostrar CWT'):
        wavelet = st.selectbox('Selecciona el tipo de wavelet (CWT)', ['cmor', 'mexh'])
        scales = st.slider('Selecciona las escalas máximas (CWT)', 1, 128, 64)
        fig = plot_cwt(emg_signal, wavelet=wavelet, scales=np.arange(1, scales + 1), fs=sampling_rate)
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

# Página EEG
def eeg_page():
    st.title('Electroencefalograma (EEG)')
    st.markdown("Técnica que registra la actividad eléctrica cerebral mediante electrodos colocados en el cuero cabelludo, utilizada para evaluar funciones cerebrales y detectar anomalías como epilepsia o trastornos del sueño.")

    eeg_signal = pd.read_csv(eeg_url).iloc[:, 0].values
    sampling_rate = 1000

    start_time = st.slider('Tiempo de inicio (en segundos)', 0.0, len(eeg_signal) / sampling_rate, 0.0, 0.001)
    end_time = st.slider('Tiempo de fin (en segundos)', 0.0, len(eeg_signal) / sampling_rate, 10.0, 0.001)

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
        fig = plot_cwt(eeg_signal, wavelet=wavelet, scales=np.arange(1, scales + 1), fs=sampling_rate)
        st.pyplot(fig)
    
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

def process_bitalino_file(file_content):
    """
    Procesa un archivo BITalino cargado como texto y extrae la última columna de datos.
    
    Parameters:
    - file_content: Contenido del archivo TXT cargado como BytesIO.
    
    Returns:
    - DataFrame con los datos procesados.
    """
    # Leer el archivo línea por línea
    lines = file_content.decode("utf-8").split("\n")
    
    # Filtrar solo las líneas de datos, ignorando las primeras 3 líneas (cabecera)
    data_lines = [line.strip() for line in lines if line.strip()][3:]
    
    # Dividir las líneas en columnas basadas en espacios o tabulaciones
    data = [line.split() for line in data_lines]
    
    # Crear un DataFrame con los datos
    df = pd.DataFrame(data)
    
    # Mantener solo la última columna
    df = df.iloc[:, -1]
    
    return df

def bitalino_converter_page():
    """
    Página para convertir un archivo TXT de BITalino a un archivo CSV descargable.
    """
    st.title("Convertidor de Archivos BITalino")
    st.markdown("Sube un archivo TXT generado por BITalino y conviértelo automáticamente a un archivo CSV descargable.")
    
    # Subir el archivo TXT
    uploaded_file = st.file_uploader("Sube tu archivo TXT de BITalino", type=["txt"])
    
    if uploaded_file:
        try:
            # Procesar el archivo
            df = process_bitalino_file(uploaded_file.getvalue())
            
            # Mostrar una vista previa de los datos procesados
            st.subheader("Vista previa de los datos procesados")
            st.dataframe(df.head(10))  # Muestra las primeras 10 filas
            
            # Generar archivo CSV descargable
            csv = df.to_csv(index=False, header=False).encode("utf-8")
            st.download_button(
                label="Descargar archivo CSV",
                data=csv,
                file_name="bitalino_procesado.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")

def home_page():
    """Página de inicio con el resumen y las descripciones técnicas."""
    st.title("Bienvenido a Biosignals")
    st.markdown(
        """
        ### Descripción General
        **Biosignals** es una aplicación diseñada para el análisis, visualización y procesamiento de señales biomédicas 
        como EEG, ECG y EMG. A través de esta plataforma, los usuarios pueden:
        
        - Analizar señales electroencefalográficas (EEG) para estudiar la actividad cerebral.
        - Examinar señales electromiográficas (EMG) para evaluar la función muscular.
        - Explorar señales electrocardiográficas (ECG) para analizar la actividad cardíaca.
        - Convertir archivos de texto generados por BITalino en formato CSV para facilitar su uso.
        
        ### Adquisición de Señales
        Las señales utilizadas en esta aplicación han sido adquiridas con sensores BITalino, configurados para registrar 
        muestras a una frecuencia de muestreo de **1000 Hz**. A continuación, se describen los procedimientos y configuraciones específicas:
        
        #### EEG (Electroencefalografía)
        - **Electrodos:** Sensor EEG colocado en la posición **FP2** (frente, encima del ojo derecho) según el sistema internacional 10-20.
        - **Objetivo:** Medición de ondas beta, indicativas de pensamiento activo.
        
        #### EMG (Electromiografía)
        - **Electrodos:** Dos electrodos de medición colocados a lo largo de las fibras musculares en el músculo **bíceps braquial**, y un electrodo de referencia en el hueso (codo o clavícula).
        - **Objetivo:** Evaluación de la actividad muscular durante la contracción.
        
        #### ECG (Electrocardiografía)
        - **Electrodos:** Configuración de **Einthoven Lead I**, con el electrodo positivo (rojo) en la clavícula izquierda, el negativo (negro) en la clavícula derecha y el de referencia (blanco) en la cresta ilíaca.
        - **Objetivo:** Registro de la actividad eléctrica cardíaca para análisis del ritmo y condiciones del corazón.

        ### ¿Qué puedes hacer con esta aplicación?
        Explora las distintas páginas disponibles en el menú lateral:
        - **EEG:** Analiza señales cerebrales para estudiar ondas beta y actividad cognitiva.
        - **EMG:** Evalúa la función muscular y registra contracciones específicas.
        - **ECG:** Explora la actividad eléctrica del corazón y analiza intervalos R-R.
        - **Tratamiento de señales:** Procesa tus señales subidas en formato CSV.
        - **Conversión TXT a CSV:** Convierte archivos de texto generados por BITalino a un formato CSV utilizable.
        """
    )

