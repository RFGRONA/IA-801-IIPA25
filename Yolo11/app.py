import streamlit as st
from PIL import Image
import os
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile

def mostrar_cabecera():
    """
    Muestra la cabecera personalizada de la aplicación con escudos y títulos.
    """
    ruta_escudo = os.path.join("assets", "escudo.png")
    ruta_logo = os.path.join("assets", "logo.png")
    col1, col2, col3 = st.columns([1, 2, 1], vertical_alignment="center")

    with col1:
        try:
            escudo = Image.open(ruta_escudo)
            st.image(escudo, width=80)
        except FileNotFoundError:
            st.warning(f"No se encontró {ruta_escudo}")

    with col2:
        st.markdown("<h2 style='text-align: center; margin-bottom: 0px;'>Inteligencia Artificial 801 - IIPA 2025</h2>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; margin-top: 0px; margin-bottom: 0px;'>Ingeniería de Sistemas y Computación</h4>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; margin-top: 5px;'>Yohan Leon, Oscar Barbosa, Gabriel Martinez</p>", unsafe_allow_html=True)

    with col3:
        try:
            logo = Image.open(ruta_logo)
            st.image(logo, width=180)
        except FileNotFoundError:
            st.warning(f"No se encontró {ruta_logo}")
    
    st.divider()

@st.cache_resource
def cargar_modelo(nombre_modelo_app):
    """
    Carga el modelo YOLO seleccionado dinámicamente.
    Usa los archivos .pt locales que están en la carpeta raíz.
    """
    model_map = {
        "YOLOv8n": "./yolov8n.pt",   
        "YOLOv9c": "./yolov9c.pt",   
        "YOLOv10b": "./yolov10b.pt", 
        "YOLOv11n": "./yolov11n.pt",
        "YOLOv11s": "./yolov11s.pt",
        "YOLOv11m": "./yolov11m.pt",
    }
    
    model_file = model_map.get(nombre_modelo_app)
    
    if model_file:
        with st.spinner(f"Cargando modelo local '{model_file}'... ⏳"):
            try:
                model = YOLO(model_file)
                return model
            except Exception as e:
                st.error(f"Error al cargar el modelo {model_file}: {e}")
                return None
    else:
        st.error("Modelo no encontrado en el mapeo.")
        return None

def main():
    # --- 1. Configuración de la Página ---
    st.set_page_config(
        page_title="Detector de Objetos YOLO",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # --- 2. Mostrar Cabecera ---
    mostrar_cabecera()
    st.header("Detector de Objetos con YOLO")

    # --- 3. Definir el Layout (Columnas) ---
    col_controles, col_visualizacion = st.columns([1, 2]) # Col. izquierda (1) más pequeña que la derecha (2)

    # Variable para guardar el JSON (si aplica)
    json_para_mostrar = None

    # --- 4. Panel de Control (Columna Izquierda) ---
    with col_controles:
        st.subheader("Panel de Control")
        
        modelo_seleccionado = st.selectbox(
            "Seleccione el modelo de YOLO:",
            ("YOLOv8n", "YOLOv9c", "YOLOv10b", "YOLOv11n", "YOLOv11s", "YOLOv11m"),
            help="Elija el modelo que desea usar para la detección."
        )
        
        uploaded_file = st.file_uploader(
            "Cargue una imagen o video",
            type=["jpg", "jpeg", "png", "mp4"],
            help="Soporta imágenes (JPG, PNG) y videos (MP4)."
        )
        
        mostrar_dataframe = st.checkbox(
            "Mostrar datos de detección (JSON)",
            help="Muestra los datos crudos (solo para imágenes)."
        )

    # --- 5. Cargar Modelo ---
    model = cargar_modelo(modelo_seleccionado)
    
    if model is None:
        st.error("No se pudo cargar el modelo. Por favor, verifique los archivos .pt.")
        return # Detener la ejecución si el modelo no cargó

    # --- 6. Área de Visualización (Columna Derecha) ---
    with col_visualizacion:
        st.subheader("Resultado")
        
        if uploaded_file is not None:
            # --- LÓGICA DE IMAGEN ---
            if uploaded_file.type.startswith('image'):
                st.info("Procesando imagen... 🖼️")
                image = Image.open(uploaded_file)
                
                results = model.predict(image, verbose=False) 
                annotated_image_bgr = results[0].plot()
                annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
                
                st.image(annotated_image_rgb, caption="Imagen Procesada", use_column_width=True)
                
                # Guardamos el JSON para mostrarlo luego
                json_para_mostrar = results[0].to_json()

            # --- LÓGICA DE VIDEO ---
            elif uploaded_file.type.startswith('video'):
                st.info("Procesando video... 🎬")
                
                stframe = st.empty() # Placeholder para el video
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                video_path = tfile.name
                
                try:
                    cap = cv2.VideoCapture(video_path)
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        results = model.predict(frame, verbose=False)
                        annotated_frame = results[0].plot()
                        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        
                        stframe.image(annotated_frame_rgb, caption=f"Procesando...", use_column_width=True)

                    st.success("¡Video procesado exitosamente!")

                except Exception as e:
                    st.error(f"Error procesando el video: {e}")
                
                finally:
                    cap.release()
                    tfile.close()
                    os.remove(video_path)
        
        else:
            st.info("Cargue un archivo en el panel de la izquierda para comenzar.")

    # --- 7. Mostrar JSON (Debajo de las columnas) ---
    st.divider()
    
    if mostrar_dataframe and json_para_mostrar is not None:
        st.subheader("Datos de Detección (JSON)")
        st.json(json_para_mostrar)
    elif mostrar_dataframe and uploaded_file is not None:
        st.warning("La salida JSON solo está disponible para el procesamiento de imágenes estáticas.")

# --- Punto de entrada principal ---
if __name__ == "__main__":
    main()