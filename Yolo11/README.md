# 🤖 Detector de Objetos con YOLO (v8-v11) y Streamlit

Este proyecto es una aplicación web interactiva construida con Streamlit que permite cargar imágenes o videos y procesarlos con diferentes modelos de la familia YOLO (You Only Look Once) para realizar detección de objetos.

-----

## 🚀 Características Principales

  * **Interfaz Gráfica Interactiva:** Creada con Streamlit, con un panel de control a la izquierda y el visualizador a la derecha.
  * **Selector de Modelos:** Permite elegir dinámicamente entre `YOLOv8n`, `YOLOv9c`, `YOLOv10b` y `YOLOv11n`.
  * **Detección en Imágenes:** Soporta la carga de archivos `.jpg`, `.jpeg` y `.png` para análisis estático.
  * **Procesamiento de Video:** Soporta la carga de archivos `.mp4` y los procesa cuadro a cuadro.
  * **Visualización de Datos:** Muestra opcionalmente los resultados de la detección (para imágenes) en formato JSON.
  * **Tema Claro:** Configurado por defecto para una mejor visibilidad.

-----

## 🛠️ Configuración y Montaje (Instalación)

Sia estos pasos para poner en marcha el proyecto en su máquina local.

### 1\. Estructura del Proyecto

Asegúrese de que el proyecto tenga la siguiente estructura de archivos:

```
YOLO11/
│
├── .streamlit/
│   └── config.toml         <-- (Configura el tema claro)
│
├── assets/
│   ├── escudo.png          <-- (Logo 1 para la cabecera)
│   └── logo.png            <-- (Logo 2 para la cabecera)
│
├── .venv/                  <-- (Entorno virtual)
│
├── app.py                  <-- (El código principal de la app)
├── requirements.txt        <-- (Las librerías de Python)
│
├── yolov8n.pt              <-- (¡Importante! Archivo de modelo)
├── yolov9c.pt              <-- (¡Importante! Archivo de modelo)
├── yolov10b.pt             <-- (¡Importante! Archivo de modelo)
└── yolov11n.pt             <-- (¡Importante! Archivo de modelo)
```

### 2\. Prerrequisitos

  * [Python 3.10+](https://www.python.org/)
  * **Archivos de Modelos (`.pt`):** Esta aplicación carga los modelos desde archivos locales. **Debes descargar** los archivos `yolov8n.pt`, `yolov9c.pt`, `yolov10b.pt` y `yolov11n.pt` y **colocarlos en la raíz del proyecto**, como se muestra en la estructura de carpetas.

### 3\. Pasos de Instalación

1.  **Clona o descarga el repositorio:**

    ```bash
    # (Si estás usando git)
    git clone https://github.com/RFGRONA/IA-801-IIPA25
    cd Yolo11
    ```

2.  **Crea y activa un entorno virtual:**

    ```bash
    # Crear el entorno
    python -m venv .venv

    # Activar en Windows
    .\.venv\Scripts\activate

    # Activar en macOS/Linux
    source .venv/bin/activate
    ```

3.  **Crea el archivo `requirements.txt`:**
    Copia y pega el siguiente contenido en el archivo `requirements.txt`:

    ```txt
    streamlit
    ultralytics
    opencv-python-headless
    Pillow
    ```

4.  **Instala las dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Crea el archivo `config.toml`:**
    Para asegurar el tema claro, crear la carpeta `.streamlit` y dentro el archivo `config.toml` con este contenido:

    ```toml
    [theme]
    base="light"
    ```

-----

## 💡 Cómo Usar el Programa

Una vez que se tenga todo instalado y los archivos `.pt` estén en su lugar, ejecutar la aplicación es muy sencillo:

1.  **Inicia el servidor de Streamlit:**
    Asegúrate de que el entorno virtual esté activado y, desde la raíz del proyecto (`Yolo11/`), ejecutar:

    ```bash
    streamlit run app.py
    ```

2.  **Abre la aplicación:**
    El navegador web se abrirá automáticamente en una pestaña (usualmente `http://localhost:8501`).

3.  **Interactúa con la App:**

      * **Panel Izquierdo (Controles):**
          * Usar el **selector** para elegir el modelo YOLO que desea probar.
          * Usar el **cargador de archivos** para subir una imagen (`.jpg`, `.png`) o un video (`.mp4`).
          * Marca la casilla **"Mostrar datos de detección (JSON)"** si desea ver los resultados crudos (solo funciona para imágenes).
      * **Panel Derecho (Resultado):**
          * La imagen o video procesado (con las cajas de detección dibujadas) aparecerá en esta área.
      * **Panel Inferior (JSON):**
          * Si la casilla está marcada y se procesó una imagen, los datos JSON aparecerán en la parte inferior de la página.