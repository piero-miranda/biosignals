import streamlit as st
from funciones import process_bitalino_file

# Configurar el logo en el sidebar
logo_url = "assets/LOGO_1.png"  # Ruta del logo
st.sidebar.image(logo_url, use_container_width=True)

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

if __name__ == "__main__":
    bitalino_converter_page()