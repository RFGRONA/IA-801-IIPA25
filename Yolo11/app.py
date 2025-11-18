import streamlit as st
from PIL import Image
import os
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import time  # Importamos time
import psutil
import gc
from collections import deque  # Importamos deque

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
    
    # Inicialización de st.session_state.metrics (Solo una vez al inicio)
    if 'metrics' not in st.session_state:
        st.session_state.metrics = {
            'fps_values': deque(maxlen=100),  # Últimos 100 FPS
            'memory_values': deque(maxlen=100),
            'detection_counts': deque(maxlen=100),
            'start_time': None,
            'frame_count': 0
        }

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
        
        # 🟢 VERIFICACIÓN CLAVE: Asegura que haya un archivo cargado
        if uploaded_file is not None:
            
            # --- LÓGICA DE IMAGEN ---
            if uploaded_file.type.startswith('image'):
                st.info("Procesando imagen... 🖼️")
                image = Image.open(uploaded_file)
                
                results = model.predict(image, verbose=False) 
                annotated_image_bgr = results[0].plot()
                annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
                
                st.image(annotated_image_rgb, caption="Imagen Procesada", width='stretch')
                
                # Guardamos el JSON para mostrarlo luego
                json_para_mostrar = results[0].to_json()

            # --- LÓGICA DE VIDEO (CON MÉTRICAS INTEGRALES) ---
            elif uploaded_file.type.startswith('video'):
                st.info("Procesando video... 🎬")
                
                # NUEVO: Sección para mostrar métricas en tiempo real
                metrics_placeholder = st.empty()
                progress_bar = st.progress(0)
                stframe = st.empty() # Placeholder para el video (movido aquí)
                
                # Creación del archivo temporal
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                video_path = tfile.name
                
                # Reiniciar contadores de frames al inicio del procesamiento de un nuevo video
                st.session_state.metrics['frame_count'] = 0 
                st.session_state.metrics['fps_values'].clear()
                st.session_state.metrics['memory_values'].clear()
                st.session_state.metrics['detection_counts'].clear()

                try:
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    st.session_state.metrics['start_time'] = time.time()
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Medir tiempo inicial
                        start_time = time.time()
                        
                        # Procesamiento del modelo
                        results = model.predict(frame, verbose=False)
                        annotated_frame = results[0].plot()
                        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        
                        # Cálculo de FPS
                        end_time = time.time()
                        fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
                        
                        # Actualizar métricas
                        st.session_state.metrics['fps_values'].append(fps)
                        st.session_state.metrics['memory_values'].append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB
                        st.session_state.metrics['detection_counts'].append(len(results[0].boxes) if results[0].boxes else 0)
                        st.session_state.metrics['frame_count'] += 1
                        
                        # Mostrar métricas en tiempo real
                        current_frame = st.session_state.metrics['frame_count']
                        # Usar max() para evitar errores si el deque está vacío al inicio
                        avg_fps = np.mean(st.session_state.metrics['fps_values']) if st.session_state.metrics['fps_values'] else 0
                        avg_memory = np.mean(st.session_state.metrics['memory_values']) if st.session_state.metrics['memory_values'] else 0
                        avg_detecciones = np.mean(st.session_state.metrics['detection_counts']) if st.session_state.metrics['detection_counts'] else 0
                        
                        with metrics_placeholder.container():
                            st.markdown(f"""
                            **Métricas en Tiempo Real:**
                            - FPS actual: **{fps:.1f}**
                            - FPS promedio: {avg_fps:.1f}
                            - Memoria usada: {avg_memory:.1f} MB
                            - Detecciones por frame: {avg_detecciones:.1f}
                            - Frames procesados: {current_frame}/{total_frames}
                            """)
                        
                        progress_bar.progress(current_frame / total_frames)
                        
                        # Mostrar frame
                        stframe.image(annotated_frame_rgb, caption=f"Procesando... Frame {current_frame}", width='stretch')

                    # Mostrar resumen final
                    show_final_metrics()
                    
                    st.success("¡Video procesado exitosamente!")

                except Exception as e:
                    st.error(f"Error procesando el video: {e}")
                
                finally:
                    # Liberación de recursos
                    if 'cap' in locals() and cap.isOpened():
                        cap.release()
                    tfile.close()
                    os.remove(video_path)
        
        else: # Si uploaded_file es None
            st.info("Cargue un archivo en el panel de la izquierda para comenzar.")

    # --- 7. Mostrar JSON (Debajo de las columnas) ---
    st.divider()
    
    if mostrar_dataframe and json_para_mostrar is not None:
        st.subheader("Datos de Detección (JSON)")
        st.json(json_para_mostrar)
    elif mostrar_dataframe and uploaded_file is not None:
        st.warning("La salida JSON solo está disponible para el procesamiento de imágenes estáticas.")

def show_final_metrics():
    """Muestra un resumen completo de las métricas después del procesamiento"""
    if st.session_state.metrics['frame_count'] > 0:
        st.subheader("📊 Resumen de Métricas de Estabilidad")
        
        total_time = time.time() - st.session_state.metrics['start_time']
        fps_values = list(st.session_state.metrics['fps_values'])
        memory_values = list(st.session_state.metrics['memory_values'])
        detection_counts = list(st.session_state.metrics['detection_counts'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("FPS Promedio", f"{np.mean(fps_values):.1f}")
            st.metric("FPS Mínimo", f"{np.min(fps_values):.1f}")
            st.metric("FPS Máximo", f"{np.max(fps_values):.1f}")
            
        with col2:
            st.metric("Memoria Promedio", f"{np.mean(memory_values):.1f} MB")
            st.metric("Memoria Máxima", f"{np.max(memory_values):.1f} MB")
            st.metric("Variación Memoria", f"{np.max(memory_values) - np.min(memory_values):.1f} MB")
            
        with col3:
            st.metric("Detecciones Promedio", f"{np.mean(detection_counts):.1f}")
            st.metric("Tiempo Total", f"{total_time:.1f} seg")
            st.metric("Frames Procesados", st.session_state.metrics['frame_count'])
        
        # Análisis de estabilidad
        memory_change = (memory_values[-1] - memory_values[0]) if len(memory_values) > 1 else 0
        stability_status = "✅ ESTABLE" if memory_change < 50 else "⚠️ INESTABLE"
        
        st.info(f"**Análisis de Estabilidad:** {stability_status}")
        st.write(f"Cambio en uso de memoria: {memory_change:.1f} MB")
        
        # Preparar datos para exportar
        metrics_data = {
            'fps_promedio': np.mean(fps_values),
            'fps_min': np.min(fps_values),
            'fps_max': np.max(fps_values),
            'memoria_promedio_mb': np.mean(memory_values),
            'memoria_max_mb': np.max(memory_values),
            'detecciones_promedio': np.mean(detection_counts),
            'tiempo_total_seg': total_time,
            'total_frames': st.session_state.metrics['frame_count']
        }
        
        st.download_button(
            label="📥 Descargar Métricas en JSON",
            data=str(metrics_data),
            file_name="metricas_estabilidad.json",
            mime="application/json"
        )

# --- Punto de entrada principal ---
if __name__ == "__main__":
    main()