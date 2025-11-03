# interfaz.py (Versión con motor de animación rediseñado)

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from PIL import Image, ImageTk
import random
import sys
import os
import numpy as np
from collections import OrderedDict
import threading
import queue
import itertools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

from backpropagation import MLP
from procesador_datos import cargar_y_convertir_dataset, convertir_imagen_individual
from kernels import KERNELS
from procesador_datos import convolve_2d_manual

def resource_path(relative_path):
    """ Obtiene la ruta absoluta al recurso, funciona para desarrollo y para PyInstaller """
    try:
        # PyInstaller crea una carpeta temporal y almacena la ruta en _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Si _MEIPASS no existe, estamos en modo de desarrollo normal
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class EditorTargetsDialog(tk.Toplevel):
    def __init__(self, parent, contenido_inicial):
        super().__init__(parent)
        self.title("Editor de Patrones de Salida (targets.txt)")
        self.geometry("400x300")
        self.resultado = None # Para almacenar el texto guardado
        
        self.transient(parent)
        self.grab_set()

        # Crear un Frame principal con padding
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Widget de texto con scrollbar
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill="both", expand=True)
        
        self.editor_texto = tk.Text(text_frame, wrap="word", height=10, width=40)
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.editor_texto.yview)
        self.editor_texto['yscrollcommand'] = scrollbar.set
        
        scrollbar.pack(side="right", fill="y")
        self.editor_texto.pack(side="left", fill="both", expand=True)
        
        self.editor_texto.insert("1.0", contenido_inicial)

        # Frame para los botones
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=(10, 0))
        
        btn_guardar = ttk.Button(button_frame, text="Guardar y Cerrar", command=self.guardar)
        btn_guardar.pack(side="right", padx=5)
        
        btn_cancelar = ttk.Button(button_frame, text="Cancelar", command=self.destroy)
        btn_cancelar.pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.wait_window(self)

    def guardar(self):
        # El .strip() al final elimina cualquier línea vacía extra que el Text widget suele añadir
        self.resultado = self.editor_texto.get("1.0", tk.END).strip()
        self.destroy()

class OpcionesDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Opciones de Entrenamiento"); self.result = "stop"; self.transient(parent); self.grab_set()
        ttk.Label(self, text="El entrenamiento está en pausa.", font=("Arial", 14)).pack(pady=20, padx=20)
        frame_botones = ttk.Frame(self); frame_botones.pack(pady=10, padx=20, fill="x")
        btn_continue = ttk.Button(frame_botones, text="Continuar Entrenamiento", command=lambda: self.set_result("continue")); btn_continue.pack(side="left", expand=True, padx=5)
        btn_stop = ttk.Button(frame_botones, text="Detener Entrenamiento", command=lambda: self.set_result("stop")); btn_stop.pack(side="left", expand=True, padx=5)
        btn_restart = ttk.Button(frame_botones, text="Reiniciar +1 Neurona Oculta", command=lambda: self.set_result("restart")); btn_restart.pack(side="left", expand=True, padx=5)
        self.protocol("WM_DELETE_WINDOW", lambda: self.set_result("stop")); self.wait_window(self)
    def set_result(self, result): self.result = result; self.destroy()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MLP con Backpropagation - IIPA 2025"); self.geometry("1400x850")
        self.protocol("WM_DELETE_WINDOW", self.cerrar_aplicacion)
        
        self.mlp_actual = None
        self.historial_mse_train, self.historial_mse_val = [], []
        self.historial_matrices = [] 
        self.epoca_inicial_bloque = 0
        self.X_train, self.Y_train, self.X_val, self.Y_val = [], [], [], []
        self.nombres_clases = []
        self.hilo_entrenamiento = None; self.entrenamiento_cancelado = False
        self.animacion_activa = False
        
        self.linea_mse_train, self.linea_mse_val = None, None
        self.linea_precision_train, self.linea_precision_val = None, None

        self.ruta_dataset = tk.StringVar(value="./dataset")
        self.ruta_targets = tk.StringVar(value="./targets.txt")
        self.ruta_dataset_preprocesamiento = tk.StringVar() 

        self.crear_cabecera()
        self.notebook = ttk.Notebook(self)
        self.tab_entrenamiento = ttk.Frame(self.notebook)
        self.tab_preprocesamiento = ttk.Frame(self.notebook) 
        self.tab_uso = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_preprocesamiento, text="Pre-procesamiento") 
        self.notebook.add(self.tab_entrenamiento, text="Entrenamiento")
        self.notebook.add(self.tab_uso, text="Uso de la Red")
        
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        
        self.mlp_uso = None; 
        self.clases_info = OrderedDict()
        self.clases_info_uso = OrderedDict()
        self.cola_gui = queue.Queue()
        
        self.img_muestra_original = None
        self.img_muestra_procesada = None
        self.var_escala = tk.StringVar(value="48x48")
        self.var_cantidad_aumentos = tk.StringVar(value="1")
        self.var_color = tk.StringVar(value="gris") 
        self.var_padding = tk.BooleanVar(value=True)
        self.entries_filtro_manual_vars = []  
        self.entries_filtro_manual_widgets = []
        
        self.crear_tab_entrenamiento()
        self.crear_tab_preprocesamiento()
        self.crear_tab_uso()
        self.procesar_cola_gui()
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        self.rutas_imagenes_totales = []

    def cerrar_aplicacion(self):
        self.entrenamiento_cancelado = True
        self.animacion_activa = False
        if self.hilo_entrenamiento and self.hilo_entrenamiento.is_alive(): self.hilo_entrenamiento.join(timeout=1)
        self.destroy()
        self.quit()

    def crear_cabecera(self):
        frame_cabecera = ttk.Frame(self, height=100); frame_cabecera.pack(fill="x", padx=10, pady=5)
        frame_cabecera.grid_columnconfigure(0, weight=1); frame_cabecera.grid_columnconfigure(1, weight=2); frame_cabecera.grid_columnconfigure(2, weight=1)
        label_escudo = ttk.Label(frame_cabecera)
        try:
            ruta_escudo = resource_path("escudo.png")
            img_escudo = Image.open(ruta_escudo).resize((60, 80), Image.Resampling.LANCZOS)
            foto_escudo = ImageTk.PhotoImage(img_escudo); label_escudo.image = foto_escudo; label_escudo.config(image=foto_escudo)
        except Exception: label_escudo.config(text="[Escudo]")
        label_escudo.grid(row=0, column=0, sticky="w")
        texto_info = ("Inteligencia Artificial 801 - IIPA 2025\nIngeniería de Sistemas y Computación\n"
                      "Yohan Leon, Oscar Barbosa, Gabriel Martinez")
        ttk.Label(frame_cabecera, text=texto_info, justify="center", font=("Arial", 12)).grid(row=0, column=1)
        label_logo = ttk.Label(frame_cabecera)
        try:
            ruta_logo = resource_path("logo.png")
            img_logo = Image.open(ruta_logo).resize((160, 80), Image.Resampling.LANCZOS)
            foto_logo = ImageTk.PhotoImage(img_logo); label_logo.image = foto_logo; label_logo.config(image=foto_logo)
        except Exception: label_logo.config(text="[Logo]")
        label_logo.grid(row=0, column=2, sticky="e")
        
    def crear_tab_entrenamiento(self):
        frame_izquierdo = ttk.Frame(self.tab_entrenamiento, width=400); frame_izquierdo.pack(side="left", fill="y", padx=10, pady=10); frame_izquierdo.pack_propagate(False)
        frame_graficas = ttk.Frame(self.tab_entrenamiento); frame_graficas.pack(side="left", expand=True, fill="both")
        frame_config = ttk.LabelFrame(frame_izquierdo, text="Configuración del Modelo MLP"); frame_config.pack(fill="x", pady=5)
        
        ttk.Button(frame_config, text="Seleccionar Carpeta Dataset", command=lambda: self.ruta_dataset.set(filedialog.askdirectory(initialdir="."))).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Label(frame_config, textvariable=self.ruta_dataset, wraplength=200).grid(row=0, column=1, columnspan=2, sticky="w")

        frame_targets = ttk.Frame(frame_config)
        frame_targets.grid(row=1, column=0, columnspan=3, sticky="ew")
        ttk.Button(frame_targets, text="Seleccionar Targets", command=lambda: self.ruta_targets.set(filedialog.askopenfilename(initialdir="."))).pack(side="left", padx=5, pady=5)
        ttk.Button(frame_targets, text="Editar Patrones", command=self.editar_patrones_salida).pack(side="left")
        ttk.Label(frame_targets, textvariable=self.ruta_targets, wraplength=150).pack(side="left", padx=5)

        ttk.Label(frame_config, text="Neuronas Capa Oculta:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.neuronas_ocultas_var = tk.IntVar(value=15); ttk.Entry(frame_config, textvariable=self.neuronas_ocultas_var, width=10).grid(row=2, column=1, sticky="w", padx=5)
        ttk.Label(frame_config, text="Tasa de Aprendizaje (\u03B1):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.tasa_aprendizaje_var = tk.StringVar(value="0.1"); ttk.Entry(frame_config, textvariable=self.tasa_aprendizaje_var, width=10).grid(row=3, column=1, sticky="w", padx=5)
        ttk.Label(frame_config, text="MSE Deseado:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.error_deseado_var = tk.StringVar(value="0.01"); ttk.Entry(frame_config, textvariable=self.error_deseado_var, width=10).grid(row=4, column=1, sticky="w", padx=5)
        ttk.Label(frame_config, text="Semilla Aleatoria (0=random):").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.semilla_var = tk.IntVar(value=0); ttk.Entry(frame_config, textvariable=self.semilla_var, width=10).grid(row=5, column=1, sticky="w", padx=5)
        self.momentum_activado = tk.BooleanVar(value=True); self.momentum_var = tk.StringVar(value="0.9")
        ttk.Checkbutton(frame_config, text="Activar Momentum:", variable=self.momentum_activado).grid(row=6, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(frame_config, textvariable=self.momentum_var, width=10).grid(row=6, column=1, sticky="w", padx=5)

        ttk.Label(frame_config, text="Épocas por bloque:").grid(row=7, column=0, sticky="w", padx=5, pady=5)
        self.epocas_bloque_var = tk.IntVar(value=500)
        ttk.Entry(frame_config, textvariable=self.epocas_bloque_var, width=10).grid(row=7, column=1, sticky="w", padx=5)

        ttk.Label(frame_config, text="Activación Oculta:").grid(row=8, column=0, sticky="w", padx=5, pady=5)
        self.act_oculta_var = tk.StringVar(value="relu")
        ttk.Combobox(frame_config, textvariable=self.act_oculta_var, 
                     values=["relu", "sigmoide"], width=10, state="readonly").grid(row=8, column=1, sticky="w", padx=5)
        
        ttk.Label(frame_config, text="Activación Salida:").grid(row=9, column=0, sticky="w", padx=5, pady=5)
        self.act_salida_var = tk.StringVar(value="sigmoide")
        ttk.Combobox(frame_config, textvariable=self.act_salida_var, 
                     values=["sigmoide", "relu"], width=10, state="readonly").grid(row=9, column=1, sticky="w", padx=5)

        ttk.Label(frame_config, text="División Dataset (% Entr.):").grid(row=10, column=0, sticky="w", padx=5, pady=5)
        
        self.division_var = tk.IntVar(value=80)
        self.division_label_var = tk.StringVar(value=f"{self.division_var.get()}% / {100-self.division_var.get()}%")
        
        frame_slider = ttk.Frame(frame_config)
        frame_slider.grid(row=10, column=1, columnspan=2, sticky="ew") # <-- Fila 10
        slider = ttk.Scale(frame_slider, from_=50, to=95, orient="horizontal", variable=self.division_var, command=lambda value: self.division_label_var.set(f"{int(float(value))}% / {100-int(float(value))}%"))
        slider.pack(side="left", expand=True, fill="x")
        ttk.Label(frame_slider, textvariable=self.division_label_var, width=10).pack(side="left")

        self.btn_iniciar = ttk.Button(frame_izquierdo, text="Iniciar Entrenamiento", command=self.iniciar_entrenamiento_nuevo); self.btn_iniciar.pack(pady=10, fill="x")
        self.btn_cancelar = ttk.Button(frame_izquierdo, text="Cancelar Entrenamiento", command=self.detener_entrenamiento, state="disabled"); self.btn_cancelar.pack(pady=5, fill="x")
        self.label_animacion = ttk.Label(frame_izquierdo, text="", font=("Arial", 10, "italic"))
        self.label_animacion.pack(pady=5)
        frame_consola = ttk.LabelFrame(frame_izquierdo, text="Consola de Entrenamiento")
        frame_consola.pack(fill="both", expand=True, pady=10)
        self.log_consola = tk.Text(frame_consola, height=10, state="disabled")
        self.log_consola.pack(fill="both", expand=True, padx=5, pady=5)

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(9, 7), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_graficas); self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.limpiar_graficas()
    
    def editar_patrones_salida(self):
        ruta_archivo = self.ruta_targets.get()
        if not ruta_archivo:
            messagebox.showwarning("Sin archivo", "Primero selecciona un archivo de targets.")
            return

        contenido_actual = ""
        try:
            with open(ruta_archivo, 'r') as f:
                contenido_actual = f.read()
        except FileNotFoundError:
            logging.warning(f"El archivo '{ruta_archivo}' no existe. Se creará uno nuevo.")
            # Texto por defecto para un archivo nuevo
            contenido_actual = ("# Define tus clases y sus vectores de salida aquí.\n"
                                "# Formato: NombreClase, valor1, valor2, ...\n"
                                "A, 0.1, 0.1, 0.9\n"
                                "E, 0.1, 0.9, 0.1\n"
                                "I, 0.1, 0.9, 0.9\n"
                                "O, 0.9, 0.1, 0.1\n"
                                "U, 0.9, 0.1, 0.9\n")
        
        dialogo = EditorTargetsDialog(self, contenido_actual)
        
        if dialogo.resultado is not None: # Si el usuario guardó
            try:
                with open(ruta_archivo, 'w') as f:
                    f.write(dialogo.resultado)
                messagebox.showinfo("Guardado", f"El archivo '{ruta_archivo}' se ha guardado correctamente.")
            except Exception as e:
                messagebox.showerror("Error al Guardar", f"No se pudo guardar el archivo:\n{e}")

    def iniciar_entrenamiento_nuevo(self):
        self.limpiar_graficas()
        self.log_consola.config(state="normal")
        self.log_consola.delete("1.0", tk.END)
        self.log_consola.config(state="disabled")
        try:
            porcentaje_entrenamiento = self.division_var.get() / 100.0

            # --- MODIFICADO: Llamada a la nueva función de carga ---
            # Esta función ahora detecta n_in y modo (L/RGB) automáticamente
            self.X_train, self.Y_train, self.X_val, self.Y_val, n_in, n_out, _, self.rutas_imagenes_totales = cargar_y_convertir_dataset(
                self.ruta_dataset.get(), 
                self.ruta_targets.get(),
                porcentaje_entrenamiento,
                semilla=self.semilla_var.get()
            )
            
            # --- MODIFICADO: Comprobación robusta de n_in ---
            if not self.X_train or n_in <= 0: 
                messagebox.showerror("Error de Carga", "No se cargaron datos de entrenamiento.\n\nVerifique que:\n1. La 'ruta_dataset' apunte a un dataset PROCESADO.\n2. El 'targets.txt' coincida con las carpetas.\n3. Todas las imágenes en el dataset tengan las MISMAS dimensiones y MODO (Gris/RGB).")
                return

            self.clases_info.clear()
            with open(self.ruta_targets.get(), 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip(): continue
                    # Corregimos posible error de comas en nombres de carpetas
                    parts = [p.strip() for p in line.strip().split(',')]; self.clases_info[parts[0].replace(',', '.')] = [float(val) for val in parts[1:]]
            self.nombres_clases = list(self.clases_info.keys())
            neuronas_ocultas = self.neuronas_ocultas_var.get()
            
            # Esta llamada ya era flexible y es correcta
            self.mlp_actual = MLP(
                n_in, 
                neuronas_ocultas, 
                n_out,
                activacion_oculta=self.act_oculta_var.get(),
                activacion_salida=self.act_salida_var.get(),
                semilla=self.semilla_var.get()
            )

            resumen_inicial = (
                f"--- INICIO DEL ENTRENAMIENTO ---\n"
                f" Arquitectura de la Red:\n"
                f"   - Neuronas de Entrada: {n_in}\n" # <-- Este valor ahora es dinámico
                f"   - Neuronas Ocultas:    {neuronas_ocultas} (Activación: {self.act_oculta_var.get()})\n"
                f"   - Neuronas de Salida:  {n_out} (Activación: {self.act_salida_var.get()})\n"
                f" Hiperparámetros:\n"
                f"   - Tasa de Aprendizaje (α): {self.tasa_aprendizaje_var.get()}\n"
                f"   - Momentum (η):              {self.momentum_var.get() if self.momentum_activado.get() else 'Desactivado'}\n"
                f"   - MSE Deseado:             {self.error_deseado_var.get()}\n"
                f" Dataset: {self.ruta_dataset.get()}\n" # <-- NUEVO: Mostrar qué dataset se usa
                f"   - Patrones Totales:      {len(self.X_train) + len(self.X_val)}\n"
                f"   - División:              {self.division_var.get()}% Entrenamiento / {100-self.division_var.get()}% Validación\n"
                f"   - Patrones Entrenamiento: {len(self.X_train)}\n"
                f"   - Patrones Validación:    {len(self.X_val)}\n"
                f"----------------------------------"
            )
            self.log_to_console(resumen_inicial)

            self.continuar_entrenamiento()
        except Exception as e:
            messagebox.showerror("Error al Iniciar", f"Ocurrió un error al iniciar el entrenamiento: {e}")
            logging.error("Error al iniciar entrenamiento", exc_info=True)

    def continuar_entrenamiento(self):
        self.btn_iniciar.config(state="disabled")
        self.btn_cancelar.config(state="normal")
        self.entrenamiento_cancelado = False
        
        self.label_animacion.config(text=f"Iniciando entrenamiento...")
        self.hilo_entrenamiento = threading.Thread(target=self._hilo_entrenamiento_bloque, daemon=True)
        self.hilo_entrenamiento.start()

    def _hilo_entrenamiento_bloque(self):
        try:
            def reportar_progreso(epoca_actual):
                self.cola_gui.put(("progress_update", epoca_actual))
            
            epoca, h_mse_train, h_mse_val, h_matrices, log_b, completo = self.mlp_actual.entrenar_bloque(
                X_train=self.X_train, Y_train=self.Y_train,
                X_val=self.X_val, Y_val=self.Y_val,
                clases_info=self.clases_info,
                tasa_aprendizaje=float(self.tasa_aprendizaje_var.get()),
                error_deseado=float(self.error_deseado_var.get()),
                momentum=float(self.momentum_var.get()) if self.momentum_activado.get() else 0.0,
                epoca_inicio=self.epoca_inicial_bloque,
                max_epocas_bloque=self.epocas_bloque_var.get(),
                cancel_event=lambda: self.entrenamiento_cancelado,
                progress_callback=reportar_progreso
            )
            
            # <--- CAMBIO: El diccionario de resultado ahora incluye ambos historiales de MSE ---
            resultado = {
                "epoca_final": epoca,
                "historial_mse_train_bloque": h_mse_train,
                "historial_mse_val_bloque": h_mse_val,
                "historial_matrices_bloque": h_matrices,
                "log_del_bloque": log_b
            }

            self.cola_gui.put(("entrenamiento_finalizado" if completo else "bloque_finalizado", resultado))
        except Exception as e:
            logging.error(f"Excepción en hilo: {e}", exc_info=True)
            self.cola_gui.put(("show_error", ("Error en Entrenamiento", str(e))))

    def procesar_cola_gui(self):
        try:
            mensaje, datos = self.cola_gui.get_nowait()
            
            # --- NUEVO: Manejar fin de generación ---
            if mensaje == "generation_complete":
                procesados, dest_root = datos
                messagebox.showinfo("Generación Completa", f"Se generaron {procesados} imágenes en:\n{dest_root}")
            # --- FIN NUEVO ---
            
            elif mensaje == "show_error":
                self.animar_carga(stop=True)
                messagebox.showerror(datos[0], datos[1])
                # Detener si fue un error de entrenamiento
                if "Entrenamiento" in datos[0]:
                    self.detener_entrenamiento()
            elif mensaje == "progress_update":
                self.animar_carga(stop=True) # Detener animacion de 'procesando'
                self.label_animacion.config(text=f"Entrenando... Época {datos}")
            elif mensaje == "log_message":
                self.log_to_console(datos) 
            elif mensaje in ["bloque_finalizado", "entrenamiento_finalizado"]:
                self.animar_carga(stop=True)
                if datos["log_del_bloque"]:
                    self.log_to_console("\n".join(datos["log_del_bloque"]))
                
                callback = self.mostrar_dialogo_opciones if mensaje == "bloque_finalizado" else self.mostrar_mensaje_final
                self.animar_graficas(datos, callback)
        except queue.Empty: pass
        finally: self.after(100, self.procesar_cola_gui)

    def mostrar_dialogo_opciones(self):
        dialog = OpcionesDialog(self); resultado = dialog.result
        if resultado == "continue": self.continuar_entrenamiento()
        elif resultado == "stop": self.detener_entrenamiento()
        elif resultado == "restart":
            self.neuronas_ocultas_var.set(self.neuronas_ocultas_var.get() + 1)
            messagebox.showinfo("Reinicio", f"Reiniciando con {self.neuronas_ocultas_var.get()} neuronas ocultas.")
            self.iniciar_entrenamiento_nuevo()

    def mostrar_mensaje_final(self):
        messagebox.showinfo("Éxito", f"Entrenamiento completado en {self.epoca_inicial_bloque} épocas.")
        self.detener_entrenamiento()

    def detener_entrenamiento(self):
        self.entrenamiento_cancelado = True; self.animar_carga(stop=True)
        
        if self.mlp_actual:
            logging.info("Guardando estado actual del modelo...")
            self.mlp_actual.guardar_modelo(clases_info=self.clases_info)
            self.label_animacion.config(text="Entrenamiento detenido. ¡Modelo guardado!")
        else:
            self.label_animacion.config(text="Entrenamiento detenido.")

        self.dibujar_estado_final_graficas()
        
        # <--- CAMBIO: Resumen final ahora incluye métricas de validación ---
        if self.historial_mse_train and self.historial_mse_val:
            mse_train_final = self.historial_mse_train[-1]
            mse_val_final = self.historial_mse_val[-1]
            # La matriz final se calcula sobre el conjunto de validación para el reporte más fiel
            matriz_final_val = self._calcular_matriz_confusion_estatica(self.X_val, self.Y_val)
            precision_final_val = np.trace(matriz_final_val) / len(self.X_val) if self.X_val else 0
            
            resumen_final = (
                f"\n--- FIN DEL ENTRENAMIENTO ---\n"
                f" Resultados Finales (Época {self.epoca_inicial_bloque}):\n"
                f"   - MSE Entrenamiento: {mse_train_final:.6f}\n"
                f"   - MSE Validación:    {mse_val_final:.6f}\n"
                f"   - Precisión Validación:  {precision_final_val:.2%}\n"
                f"----------------------------------"
            )
            self.log_to_console(resumen_final)

        self.btn_iniciar.config(state="normal"); self.btn_cancelar.config(state="disabled")

    # --- CAMBIO: Nueva función helper para escribir en la consola ---
    def log_to_console(self, message):
        """Inserta un mensaje en el widget de la consola de forma segura."""
        self.log_consola.config(state="normal")
        self.log_consola.insert(tk.END, message + "\n")
        self.log_consola.see(tk.END)
        self.log_consola.config(state="disabled")

    def animar_carga(self, stop=False):
        if stop:
            self.label_animacion.config(text="Cálculos finalizados.")

    def limpiar_graficas(self):
        self.epoca_inicial_bloque = 0
        self.historial_mse_train, self.historial_mse_val = [], []
        self.historial_matrices = []
        self.linea_mse_train, self.linea_mse_val = None, None
        self.linea_precision_train, self.linea_precision_val = None, None
        
        for ax in [self.ax1, self.ax2, self.ax3]: ax.clear(); ax.grid(True)
        
        # Gráfica de Error (ax1) se mantiene como estaba originalmente
        self.ax1.set_title("MSE vs. Épocas"); self.ax1.set_xlabel("Época"); self.ax1.set_ylabel("MSE")
        self.ax1.legend(["Entrenamiento", "Validación"], loc="upper right")

        self.ax2.set_title("Precisión Validación vs Épocas (cada 25 épocas)")
        self.ax2.set_xlabel("Época")
        self.ax2.set_ylabel("Precisión")
        self.ax3.set_title("Matriz de Confusión (Validación)"); self.ax3.set_xlabel("Predicción"); self.ax3.set_ylabel("Real")
        self.ax3.set_xticks([]); self.ax3.set_yticks([])
        self.canvas.draw()
        
    def animar_graficas(self, datos_resultado, on_done_callback):
        epoca_final_anterior = self.epoca_inicial_bloque
        self.epoca_inicial_bloque = datos_resultado["epoca_final"]
        self.historial_mse_train.extend(datos_resultado["historial_mse_train_bloque"])
        self.historial_mse_val.extend(datos_resultado["historial_mse_val_bloque"])
        self.historial_matrices.extend(datos_resultado["historial_matrices_bloque"])
        
        eje_x_nuevo = range(epoca_final_anterior + 1, self.epoca_inicial_bloque + 1)
        
        self.animacion_activa = True
        self._bucle_animacion(
            eje_x_nuevo,
            datos_resultado["historial_mse_train_bloque"],
            datos_resultado["historial_mse_val_bloque"],
            datos_resultado["historial_matrices_bloque"],
            0,
            on_done_callback
        )

    def _bucle_animacion(self, x_data, mse_train_data, mse_val_data, matrices_data, index, on_done_callback):
        if not self.animacion_activa or index >= len(x_data):
            if self.animacion_activa: self.dibujar_estado_final_graficas(); self.after(100, on_done_callback)
            self.animacion_activa = False; return

        step = 25; delay_ms = 20; MUESTRAS_MATRIZ = 25
        
        if not self.linea_mse_train: self.linea_mse_train, = self.ax1.plot([], [], 'b-', label='Entrenamiento')
        if not self.linea_mse_val: self.linea_mse_val, = self.ax1.plot([], [], 'r-', label='Validación')
        if not self.linea_precision_train: self.linea_precision_train, = self.ax2.plot([], [], 'b-', label='Entrenamiento')
        if not self.linea_precision_val: self.linea_precision_val, = self.ax2.plot([], [], 'r-', label='Validación')
        
        next_index = min(index + step, len(x_data))

        # Actualizar MSE
        self.linea_mse_train.set_data(np.append(self.linea_mse_train.get_xdata(), x_data[index:next_index]), np.append(self.linea_mse_train.get_ydata(), mse_train_data[index:next_index]))
        self.linea_mse_val.set_data(np.append(self.linea_mse_val.get_xdata(), x_data[index:next_index]), np.append(self.linea_mse_val.get_ydata(), mse_val_data[index:next_index]))
        
        # Actualizar Precisión
        if matrices_data:
            precision_val_segmento = []
            for i in range(index, next_index):
                epoca_actual = x_data[i]
                indice_matriz = (epoca_actual - 1) // MUESTRAS_MATRIZ
                if indice_matriz < len(matrices_data):
                    matriz = matrices_data[indice_matriz]
                    precision = np.trace(matriz) / len(self.X_val) if len(self.X_val) > 0 else 0
                    precision_val_segmento.append(precision)
                else:
                    if len(self.linea_precision_val.get_ydata()) > 0:
                        precision_val_segmento.append(self.linea_precision_val.get_ydata()[-1])
                    else:
                        precision_val_segmento.append(0)

            self.linea_precision_val.set_data(
                np.append(self.linea_precision_val.get_xdata(), x_data[index:next_index]),
                np.append(self.linea_precision_val.get_ydata(), precision_val_segmento)
            )

        # Mostrar Matriz de Confusión
        if matrices_data:
            indice_matriz_display = (x_data[index] - 1) // MUESTRAS_MATRIZ
            if indice_matriz_display < len(matrices_data):
                matriz_a_mostrar = matrices_data[indice_matriz_display]
                self.dibujar_matriz_confusion_estatica(matriz_a_mostrar, epoca_actual=x_data[index])

        self.ax1.relim(); self.ax1.autoscale_view(); self.ax1.legend()
        self.ax2.relim(); self.ax2.autoscale_view() 
        self.canvas.draw()
        
        self.after(delay_ms, self._bucle_animacion, x_data, mse_train_data, mse_val_data, matrices_data, next_index, on_done_callback)

    def _calcular_matriz_confusion_estatica(self, X_data, Y_data):
        matriz = np.zeros((len(self.nombres_clases), len(self.nombres_clases)))
        if not X_data: return matriz
        
        for x, y_real_vec in zip(X_data, Y_data):
            prediccion_vec = np.array(self.mlp_actual.predecir(x))
            target_vectors = list(self.clases_info.values())
            distancias_real = [np.linalg.norm(np.array(y_real_vec) - np.array(tv)) for tv in target_vectors]
            idx_real = np.argmin(distancias_real)
            distancias_pred = [np.linalg.norm(prediccion_vec - np.array(tv)) for tv in target_vectors]
            idx_pred = np.argmin(distancias_pred)
            matriz[idx_real, idx_pred] += 1
        return matriz 

    def dibujar_matriz_confusion_estatica(self, matriz, epoca_actual=None):
        """Toma una matriz (pre-calculada) y la dibuja en el tercer eje (ax3)."""
        self.ax3.clear()
        if epoca_actual:
            self.ax3.set_title(f"Matriz de Confusión (Época {epoca_actual})")
        else:
            self.ax3.set_title("Matriz de Confusión (Validación)")
        
        n_clases = len(self.nombres_clases)
        self.ax3.matshow(matriz, cmap=plt.cm.Blues, alpha=0.7)
        self.ax3.set_xticks(np.arange(n_clases))
        self.ax3.set_yticks(np.arange(n_clases))
        self.ax3.set_xticklabels(self.nombres_clases)
        self.ax3.set_yticklabels(self.nombres_clases)
        
        # Escribe los números dentro de cada celda
        for i in range(matriz.shape[0]):
            for j in range(matriz.shape[1]):
                self.ax3.text(x=j, y=i, s=int(matriz[i, j]), va='center', ha='center', size='large')
    
    def dibujar_estado_final_graficas(self):
        if not self.historial_mse_train: return
        
        eje_x = range(1, len(self.historial_mse_train) + 1)
        
        # Dibujar las líneas de MSE
        if not self.linea_mse_train: self.linea_mse_train, = self.ax1.plot(eje_x, self.historial_mse_train, 'b-', label='Entrenamiento')
        else: self.linea_mse_train.set_data(eje_x, self.historial_mse_train)
        if not self.linea_mse_val: self.linea_mse_val, = self.ax1.plot(eje_x, self.historial_mse_val, 'r-', label='Validación')
        else: self.linea_mse_val.set_data(eje_x, self.historial_mse_val)

        # Dibujar la línea de Precisión
        eje_x_prec = []
        precision_val = []
        for i, epoca in enumerate(eje_x):
            if (epoca % 25 == 0 or epoca == len(eje_x)) and self.historial_matrices:
                indice_matriz = min( (epoca - 1) // 25, len(self.historial_matrices) - 1)
                matriz = self.historial_matrices[indice_matriz]
                acc = np.trace(matriz) / len(self.X_val) if len(self.X_val) > 0 else 0
                eje_x_prec.append(epoca)
                precision_val.append(acc)
        
        if not self.linea_precision_val:
            self.linea_precision_val, = self.ax2.plot(eje_x_prec, precision_val, 'r-o', label='Validación')
        else:
            self.linea_precision_val.set_data(eje_x_prec, precision_val)
            self.linea_precision_val.set_linestyle('-')
            self.linea_precision_val.set_marker('o')

        # Dibujar la Matriz final
        if self.X_val:
            matriz_final = self._calcular_matriz_confusion_estatica(self.X_val, self.Y_val)
            self.dibujar_matriz_confusion_estatica(matriz_final, epoca_actual=len(self.historial_mse_train))
        
        self.ax1.relim(); self.ax1.autoscale_view(); self.ax1.legend()
        self.ax2.relim(); self.ax2.autoscale_view() 
        self.canvas.draw()

    def on_tab_changed(self, event):
        selected_tab_index = self.notebook.index(self.notebook.select())
        if selected_tab_index == 1:
            self.cargar_recursos_uso()
            
    def crear_tab_uso(self):
        self.frame_uso_principal = ttk.Frame(self.tab_uso)
        self.frame_uso_principal.pack(fill="both", expand=True)

        self.canvas_red = tk.Canvas(self.frame_uso_principal, bg="white")
        self.canvas_red.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        frame_controles = ttk.Frame(self.frame_uso_principal, width=300)
        frame_controles.pack(side="left", fill="y", padx=10, pady=10)
        frame_controles.pack_propagate(False)
        
        self.btn_predecir_imagen = ttk.Button(frame_controles, text="Seleccionar y Predecir Imagen (.png)", command=self.predecir_imagen, state="disabled")
        self.btn_predecir_imagen.pack(pady=10, fill="x")

        self.btn_predecir_aleatoria = ttk.Button(frame_controles, text="Probar con Imagen Aleatoria", command=self.probar_imagen_aleatoria, state="disabled")
        self.btn_predecir_aleatoria.pack(pady=5, fill="x")

        frame_imagen = ttk.LabelFrame(frame_controles, text="Imagen a Predecir")
        frame_imagen.pack(fill="x", pady=10)
        self.label_imagen_predecida = ttk.Label(frame_imagen, text="\nCargue una imagen\n", style="Card.TLabel", anchor="center")
        self.label_imagen_predecida.pack(pady=20, padx=20)

        self.label_prediccion_binaria = ttk.Label(frame_controles, text="Salida: [?]", font=("Courier", 12))
        self.label_prediccion_binaria.pack(pady=20)
        
        frame_traduccion = ttk.LabelFrame(frame_controles, text="Predicción")
        frame_traduccion.pack(fill="x", pady=10)
        self.label_prediccion_letra = ttk.Label(frame_traduccion, text="?", font=("Arial", 36, "bold"), anchor="center")
        self.label_prediccion_letra.pack(pady=20, fill="x")

    def cargar_recursos_uso(self):
        logging.info("Cambiando a la pestaña de Uso. Recargando modelo y targets...")
        self.clases_info_uso.clear()
        modelo_cargado = False

        try:
            # Ahora esperamos recibir el modelo y la info de las clases
            self.mlp_uso, clases_info_cargadas = MLP.cargar_modelo("modelo_mlp.json")
            
            if self.mlp_uso and clases_info_cargadas:
                modelo_cargado = True
                # Usamos la info de clases que vino CON el modelo
                self.clases_info_uso = clases_info_cargadas
                logging.info("Modelo y datos de clases cargados exitosamente desde 'modelo_mlp.json'.")
                self.dibujar_red_uso()
            else:
                # Si algo falla, lo notificamos
                messagebox.showerror("Error", "No se pudo cargar 'modelo_mlp.json' o no contiene información de clases.", parent=self.tab_uso)

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar 'modelo_mlp.json': {e}", parent=self.tab_uso)
        
        if modelo_cargado:
            self.btn_predecir_imagen.config(state="normal")
            # Activa también el botón de predicción aleatoria si existe
            if hasattr(self, 'btn_predecir_aleatoria'):
                self.btn_predecir_aleatoria.config(state="normal")
        else:
            self.btn_predecir_imagen.config(state="disabled")
            if hasattr(self, 'btn_predecir_aleatoria'):
                self.btn_predecir_aleatoria.config(state="disabled")

    def predecir_imagen(self):
        if not self.mlp_uso or not self.clases_info_uso:
            messagebox.showwarning("Recursos no cargados", "Asegúrese de que el modelo y los targets estén cargados.")
            return
            
        ruta = filedialog.askopenfilename(filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if not ruta: return
        
        try:
            img = Image.open(ruta)
            photo = self._crear_imagen_previsualizacion(img)
            self.label_imagen_predecida.config(image=photo)
            self.label_imagen_predecida.image = photo 
            
            vector_entrada = convertir_imagen_individual(ruta)
            if len(vector_entrada) != self.mlp_uso.neuronas_entrada:
                messagebox.showerror("Error", f"La imagen no tiene el tamaño correcto. Se esperaba un vector de {self.mlp_uso.neuronas_entrada} píxeles.")
                return
            
            # Usamos el método público .predecir(), que devuelve una lista de Python
            salidas_finales = self.mlp_uso.predecir(vector_entrada) 
            
            # Ahora la siguiente línea funciona porque 'salidas_finales' es una lista normal
            self.label_prediccion_binaria.config(text=f"Salida: {[round(s, 2) for s in salidas_finales]}")

            # El resto de la lógica no cambia
            prediccion_vec = np.array(salidas_finales)
            target_vectors = list(self.clases_info_uso.values())
            nombres_clases = list(self.clases_info_uso.keys())

            distancias = [np.linalg.norm(prediccion_vec - np.array(tv)) for tv in target_vectors]
            idx_pred_correcto = np.argmin(distancias)
            letra_predicha = nombres_clases[idx_pred_correcto]
            
            self.label_prediccion_letra.config(text=letra_predicha)

        except Exception as e:
            messagebox.showerror("Error al Predecir", str(e))
    
    def dibujar_red_uso(self, salidas_ocultas=None, salidas_finales=None):
        self.canvas_red.delete("all")
        if not self.mlp_uso: return
        w, h = self.canvas_red.winfo_width(), self.canvas_red.winfo_height()
        x_in, x_hidden, x_out = w * 0.1, w * 0.5, w * 0.9
        self.canvas_red.create_oval(x_in-20, h/2-20, x_in+20, h/2+20, fill="lightgray"); self.canvas_red.create_text(x_in, h/2, text=f"{self.mlp_uso.neuronas_entrada}\nEntradas")
        y_step_h = h / (self.mlp_uso.neuronas_ocultas + 1)
        for j in range(self.mlp_uso.neuronas_ocultas):
            y_h = y_step_h * (j + 1)
            self.canvas_red.create_oval(x_hidden-15, y_h-15, x_hidden+15, y_h+15, fill="lightblue")
            for i in range(self.mlp_uso.neuronas_entrada): self.canvas_red.create_line(x_in, h/2, x_hidden, y_h, fill="gray")
        y_step_o = h / (self.mlp_uso.neuronas_salida + 1)
        for k in range(self.mlp_uso.neuronas_salida):
            y_o = y_step_o * (k + 1)
            self.canvas_red.create_oval(x_out-15, y_o-15, x_out+15, y_o+15, fill="lightgreen")
            for j in range(self.mlp_uso.neuronas_ocultas):
                y_h = y_step_h * (j + 1)
                self.canvas_red.create_line(x_hidden, y_h, x_out, y_o, fill="gray")
                self.canvas_red.create_text((x_hidden+x_out)/2, (y_h+y_o)/2, text=f"{self.mlp_uso.pesos_ho[k][j]:.1f}", font=("Arial", 7))

    def probar_imagen_aleatoria(self):
        """
        Selecciona una imagen al azar del dataset cargado y ejecuta la predicción.
        """
        if not (hasattr(self, 'rutas_imagenes_totales') and self.rutas_imagenes_totales):
            messagebox.showwarning("Sin Datos", "Primero debes cargar un dataset en la pestaña de Entrenamiento.")
            return
        
        # Escoge una ruta de imagen al azar
        ruta_aleatoria = random.choice(self.rutas_imagenes_totales)
        print(f"Probando con imagen aleatoria: {ruta_aleatoria}")

        # Ahora, el resto del código es idéntico al de predecir_imagen
        try:
            img = Image.open(ruta_aleatoria)
            photo = self._crear_imagen_previsualizacion(img)
            self.label_imagen_predecida.config(image=photo)
            self.label_imagen_predecida.image = photo 
            
            vector_entrada = convertir_imagen_individual(ruta_aleatoria)
            if len(vector_entrada) != self.mlp_uso.neuronas_entrada:
                messagebox.showerror("Error de Tamaño", f"La imagen no tiene el tamaño correcto. Se esperaba un vector de {self.mlp_uso.neuronas_entrada} píxeles.")
                return
            
            # Realizar la predicción
            salidas_finales = self.mlp_uso.predecir(vector_entrada) # predecir ya devuelve una lista
            self.label_prediccion_binaria.config(text=f"Salida: {[round(s, 2) for s in salidas_finales]}")

            # Traducir la predicción a una clase
            prediccion_vec = np.array(salidas_finales)
            target_vectors = list(self.clases_info_uso.values())
            nombres_clases = list(self.clases_info_uso.keys())

            distancias = [np.linalg.norm(prediccion_vec - np.array(tv)) for tv in target_vectors]
            idx_pred_clase = np.argmin(distancias)
            letra_predicha = nombres_clases[idx_pred_clase]
            
            self.label_prediccion_letra.config(text=letra_predicha)

        except Exception as e:
            messagebox.showerror("Error al Predecir", str(e))
    
    def _crear_imagen_previsualizacion(self, pil_image):
        """
        Crea una imagen de previsualización con zoom para imágenes pequeñas
        y encoge las imágenes grandes.
        """
        img_copy = pil_image.copy()
        w, h = img_copy.size
        
        # Tamaño máximo del contenedor
        max_w, max_h = 300, 300 

        if w < 100 and h < 100:
            # --- Es una imagen pequeña (ej. 48x48) ---
            # Aplicamos zoom x3 pixelado (NEAREST)
            # Como en tu captura, (48, 48) -> (144, 144)
            zoom = 3
            img_copy = img_copy.resize((w * zoom, h * zoom), Image.Resampling.NEAREST)
        else:
            # --- Es una imagen grande ---
            # La encogemos para que quepa en (300, 300) manteniendo la proporción (LANCZOS)
            img_copy.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

        return ImageTk.PhotoImage(img_copy)

    def crear_tab_preprocesamiento(self):
        # --- MODIFICADO: UI Corregida (Puntos 2, 4, 5 de la tanda anterior) ---
        main_paned_window = ttk.PanedWindow(self.tab_preprocesamiento, orient="horizontal")
        main_paned_window.pack(fill="both", expand=True, padx=10, pady=10)
        
        frame_controles = ttk.Frame(main_paned_window, width=450); frame_controles.pack_propagate(False)
        main_paned_window.add(frame_controles, weight=1)

        frame_carga = ttk.LabelFrame(frame_controles, text="A. Cargar Dataset Base")
        frame_carga.pack(fill="x", padx=10, pady=10)
        btn_cargar_base = ttk.Button(frame_carga, text="Seleccionar Carpeta (Base)", command=self._seleccionar_carpeta_base)
        btn_cargar_base.pack(fill="x", padx=5, pady=5)
        label_ruta_pre = ttk.Label(frame_carga, textvariable=self.ruta_dataset_preprocesamiento, wraplength=430)
        label_ruta_pre.pack(padx=5, pady=(0, 5))

        frame_acciones = ttk.LabelFrame(frame_controles, text="B. Previsualizar")
        frame_acciones.pack(fill="x", padx=10, pady=10)
        self.btn_cargar_muestra = ttk.Button(frame_acciones, text="Cargar Imagen de Muestra", command=self._cargar_imagen_muestra)
        self.btn_cargar_muestra.pack(side="left", padx=5, pady=5, expand=True)
        self.btn_previsualizar = ttk.Button(frame_acciones, text="Previsualizar Cambios", command=self._actualizar_preview, state="disabled")
        self.btn_previsualizar.pack(side="left", padx=5, pady=5, expand=True)

        frame_transform = ttk.LabelFrame(frame_controles, text="C. Transformación")
        frame_transform.pack(fill="x", padx=10, pady=5)
        ttk.Label(frame_transform, text="Reescalar (Ancho x Alto):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(frame_transform, textvariable=self.var_escala, width=10).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame_transform, text="Rotaciones por filtro:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.entry_cantidad_aumentos = ttk.Entry(frame_transform, textvariable=self.var_cantidad_aumentos, width=10)
        self.entry_cantidad_aumentos.grid(row=1, column=1, padx=5, pady=5, sticky="w")

        ttk.Label(frame_transform, text="Color:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        frame_color_radios = ttk.Frame(frame_transform)
        frame_color_radios.grid(row=2, column=1, columnspan=2, sticky="w")
        ttk.Radiobutton(frame_color_radios, text="Escala de Grises", variable=self.var_color, value="gris").pack(side="left", padx=5)
        ttk.Radiobutton(frame_color_radios, text="Color (RGB)", variable=self.var_color, value="color").pack(side="left", padx=5)

        frame_filtro_main = ttk.LabelFrame(frame_controles, text="D. Filtros (Selección Múltiple)")
        frame_filtro_main.pack(fill="x", padx=10, pady=5)
        frame_filtro_container = ttk.Frame(frame_filtro_main)
        frame_filtro_container.pack(fill="x", expand=True, padx=5, pady=5)
        
        frame_listbox = ttk.Frame(frame_filtro_container)
        frame_listbox.pack(side="left", fill="y", expand=True, padx=(0, 5))
        ttk.Label(frame_listbox, text="Seleccionar Filtros:").pack(anchor="w")
        scrollbar_listbox = ttk.Scrollbar(frame_listbox, orient="vertical")
        self.filtro_listbox = tk.Listbox(frame_listbox, selectmode="extended", 
                                         yscrollcommand=scrollbar_listbox.set, 
                                         exportselection=False, height=8)
        scrollbar_listbox.config(command=self.filtro_listbox.yview)
        scrollbar_listbox.pack(side="right", fill="y")
        self.filtro_listbox.pack(side="left", fill="both", expand=True)
        for k in KERNELS.keys(): self.filtro_listbox.insert(tk.END, k)
        self.filtro_listbox.insert(tk.END, "Manual")
        self.filtro_listbox.bind("<<ListboxSelect>>", self._on_filtro_seleccionado)
        
        frame_manual = ttk.Frame(frame_filtro_container)
        frame_manual.pack(side="left", fill="y", padx=(5, 0))
        ttk.Label(frame_manual, text="Filtro Manual (5x5):").pack(anchor="w")
        grid_frame = ttk.Frame(frame_manual)
        grid_frame.pack(anchor="center")
        
        self.entries_filtro_manual_vars = []
        self.entries_filtro_manual_widgets = []
        for r in range(5):
            row_vars = []; row_widgets = []
            for c in range(5):
                val = "0";
                if r == 2 and c == 2: val = "1"
                entry_var = tk.StringVar(value=val)
                entry = ttk.Entry(grid_frame, textvariable=entry_var, width=5, justify="center", state="disabled")
                entry.grid(row=r, column=c, padx=1, pady=1)
                row_vars.append(entry_var); row_widgets.append(entry)
            self.entries_filtro_manual_vars.append(row_vars)
            self.entries_filtro_manual_widgets.append(row_widgets)
        
        self.check_padding = ttk.Checkbutton(frame_filtro_main, text="Activar Padding (Mantener tamaño)", variable=self.var_padding, onvalue=True, offvalue=False)
        self.check_padding.pack(anchor="w", padx=5, pady=(10, 5))
        self.var_padding.set(True) 

        frame_generacion = ttk.LabelFrame(frame_controles, text="E. Generar Nuevo Dataset")
        frame_generacion.pack(fill="x", padx=10, pady=10, side="bottom")
        
        # --- MODIFICADO: Renombrar botón (Punto 1) ---
        self.btn_generar_pipeline = ttk.Button(frame_generacion, text="Generar (Por Filtro Individual)", command=self._generar_dataset_individual, state="disabled")
        self.btn_generar_pipeline.pack(fill="x", padx=5, pady=5)
        
        self.btn_generar_auto = ttk.Button(frame_generacion, text="Generar (Combinatorio)", command=self._generar_auto, state="disabled")
        self.btn_generar_auto.pack(fill="x", padx=5, pady=5)
        
        # Panel de Previsualización (Derecha)
        preview_paned_window = ttk.PanedWindow(main_paned_window, orient="vertical")
        main_paned_window.add(preview_paned_window, weight=2)
        frame_imagenes = ttk.LabelFrame(preview_paned_window, text="Previsualización de Imagen")
        preview_paned_window.add(frame_imagenes, weight=1)
        frame_imagenes.rowconfigure(1, weight=1); frame_imagenes.columnconfigure(0, weight=1); frame_imagenes.columnconfigure(1, weight=1)
        ttk.Label(frame_imagenes, text="Original", font=("Arial", 12, "bold")).grid(row=0, column=0, pady=5)
        ttk.Label(frame_imagenes, text="Procesada", font=("Arial", 12, "bold")).grid(row=0, column=1, pady=5)
        self.label_img_original = ttk.Label(frame_imagenes, text="Cargue una imagen de muestra", anchor="center", relief="solid")
        self.label_img_original.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.label_img_procesada = ttk.Label(frame_imagenes, text="Los cambios aparecerán aquí", anchor="center", relief="solid")
        self.label_img_procesada.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        frame_matrices = ttk.LabelFrame(preview_paned_window, text="Previsualización de Matriz (Valores 0-255)")
        preview_paned_window.add(frame_matrices, weight=1)
        frame_mat_orig = ttk.Frame(frame_matrices); frame_mat_orig.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scroll_mat_orig_v = ttk.Scrollbar(frame_mat_orig, orient="vertical"); scroll_mat_orig_h = ttk.Scrollbar(frame_mat_orig, orient="horizontal")
        self.text_matriz_original = tk.Text(frame_mat_orig, height=10, state="disabled", font=("Courier", 9), wrap="none", yscrollcommand=scroll_mat_orig_v.set, xscrollcommand=scroll_mat_orig_h.set)
        scroll_mat_orig_v.config(command=self.text_matriz_original.yview); scroll_mat_orig_h.config(command=self.text_matriz_original.xview)
        scroll_mat_orig_v.pack(side="right", fill="y"); scroll_mat_orig_h.pack(side="bottom", fill="x"); self.text_matriz_original.pack(fill="both", expand=True)
        frame_mat_proc = ttk.Frame(frame_matrices); frame_mat_proc.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        scroll_mat_proc_v = ttk.Scrollbar(frame_mat_proc, orient="vertical"); scroll_mat_proc_h = ttk.Scrollbar(frame_mat_proc, orient="horizontal")
        self.text_matriz_procesada = tk.Text(frame_mat_proc, height=10, state="disabled", font=("Courier", 9), wrap="none", yscrollcommand=scroll_mat_proc_v.set, xscrollcommand=scroll_mat_proc_h.set)
        scroll_mat_proc_v.config(command=self.text_matriz_procesada.yview); scroll_mat_proc_h.config(command=self.text_matriz_procesada.xview)
        scroll_mat_proc_v.pack(side="right", fill="y"); scroll_mat_proc_h.pack(side="bottom", fill="x"); self.text_matriz_procesada.pack(fill="both", expand=True)

    def _on_filtro_seleccionado(self, event):
        """(MODIFICADO v3 - Punto 2)
        Habilita o deshabilita la grilla del filtro manual.
        """
        try:
            selected_indices = self.filtro_listbox.curselection()
            selected_filtros = [self.filtro_listbox.get(i) for i in selected_indices]

            # --- MODIFICADO: Usar la lista de widgets (Punto 2) ---
            if "Manual" in selected_filtros:
                # Habilitar la grilla
                for r in range(5):
                    for c in range(5):
                        self.entries_filtro_manual_widgets[r][c].config(state="normal")
            else:
                # Deshabilitar la grilla
                for r in range(5):
                    for c in range(5):
                        self.entries_filtro_manual_widgets[r][c].config(state="disabled")
            # --- FIN MODIFICADO ---
        except Exception as e:
            logging.error(f"Error en _on_filtro_seleccionado: {e}")


    def _cargar_imagen_muestra(self):
        """(MODIFICADO v5 - Punto 1 y 2) 
        Carga una imagen de muestra desde la ruta de pre-procesamiento.
        - Ahora usa os.walk (recursivo) para encontrar TODAS las imágenes.
        - Muestra la matriz original VERDADERA inmediatamente.
        """
        
        source_root = self.ruta_dataset_preprocesamiento.get()
        if not source_root or not os.path.isdir(source_root):
            messagebox.showwarning("Sin Dataset", "Presiona 'Seleccionar Carpeta (Base)' para cargar un dataset.")
            return

        # --- MODIFICADO: Lógica de carga RECURSIVA (Punto 1) ---
        imagenes_disponibles = []
        try:
            for dirpath, _, filenames in os.walk(source_root):
                for filename in filenames:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                        imagenes_disponibles.append(os.path.join(dirpath, filename))
        except Exception as e:
            messagebox.showerror("Error al leer carpeta", f"No se pudo leer la carpeta:\n{source_root}\nError: {e}")
            return
        # --- FIN MODIFICADO ---

        if not imagenes_disponibles:
            messagebox.showwarning("Sin Imágenes", f"No se encontraron imágenes (.png, .jpg) en:\n{source_root}\n(Ni en sus subcarpetas)")
            return
            
        ruta_muestra = random.choice(imagenes_disponibles)
        
        try:
            self.img_muestra_original = Image.open(ruta_muestra)
            
            # --- INICIO DE CORRECCIÓN (Punto 2: Matriz) ---
            # Mostrar la imagen original
            img_tk_original = self._crear_imagen_previsualizacion(self.img_muestra_original)
            self.label_img_original.config(image=img_tk_original, text="")
            self.label_img_original.image = img_tk_original
            
            # Mostrar la matriz original VERDADERA
            self._mostrar_matriz(self.text_matriz_original, np.array(self.img_muestra_original))
            # --- FIN DE CORRECCIÓN ---
            
            # Activar botones
            self.btn_previsualizar.config(state="normal")
            self.btn_generar_pipeline.config(state="normal")
            self.btn_generar_auto.config(state="normal")
            
            # Ejecutar la primera previsualización (que solo actualizará el panel derecho)
            self._actualizar_preview()
            
        except Exception as e:
            messagebox.showerror("Error al Cargar Muestra", f"No se pudo cargar la imagen de muestra:\n{ruta_muestra}\nError: {e}")

    def _actualizar_preview(self):
        """(MODIFICADA) Pasa el booleano de rotación."""
        if not self.img_muestra_original:
            messagebox.showwarning("Sin Imagen", "Primero carga una imagen de muestra.")
            return

        logging.info("Actualizando previsualización...")
        try:
            settings = self._leer_controles_pipeline()
            if settings is None: return

            kernels_en_cadena = list(settings['kernels_dict'].values())
            
            # --- MODIFICADO (Punto 2) ---
            # Si el usuario pide > 1 aumentos, mostrar un ejemplo rotado.
            aplicar_rot_preview = (settings['cantidad_aumentos'] > 1)
            
            self.img_muestra_procesada = self._aplicar_pipeline_a_imagen(
                self.img_muestra_original, 
                settings, 
                kernel_list=kernels_en_cadena,
                aplicar_rotacion=aplicar_rot_preview # <-- MODIFICADO
            )
            # --- FIN MODIFICADO ---

            img_tk_procesada = self._crear_imagen_previsualizacion(self.img_muestra_procesada)
            self.label_img_procesada.config(image=img_tk_procesada, text=""); self.label_img_procesada.image = img_tk_procesada
            self._mostrar_matriz(self.text_matriz_procesada, np.array(self.img_muestra_procesada))
            
            self._mostrar_matriz(self.text_matriz_original, np.array(self.img_muestra_original))
            logging.info(f"Previsualización actualizada.")
            
        except Exception as e:
            logging.error(f"Error al actualizar preview: {e}", exc_info=True)
            messagebox.showerror("Error de Previsualización", f"No se pudo aplicar la transformación: {e}")

    def _mostrar_matriz(self, text_widget, matriz):
        """(MODIFICADA) Helper para mostrar una matriz NumPy completa en un widget de Texto."""
        text_widget.config(state="normal")
        text_widget.delete("1.0", tk.END)
        
        info = ""
        if matriz.ndim == 3:
            matriz_display = matriz[:, :, 0]
            info = f"Mostrando 1er Canal (Rojo) - Dimensiones: {matriz.shape}\n"
        else:
            matriz_display = matriz
            info = f"Mostrando Canal Único (Gris) - Dimensiones: {matriz.shape}\n"
        
        text_widget.insert(tk.END, info + "-"*20 + "\n")
        
        # --- MODIFICADO: Imprimir matriz completa (Punto 3) ---
        matriz_str = np.array2string(
            matriz_display, 
            precision=0, 
            suppress_small=True,
            threshold=np.inf, # <-- Fuerza a NumPy a no truncar
            max_line_width=np.inf # <-- Evita saltos de línea
        )
        text_widget.insert(tk.END, matriz_str)
        text_widget.config(state="disabled")


    def _generar_dataset_individual(self):
        """(MODIFICADO) Botón principal: Inicia el HILO para "Generar por Filtro Individual"."""
        logging.info("Iniciando generación de dataset (Individual)...")
        source_root = self.ruta_dataset_preprocesamiento.get()
        if not source_root or not os.path.isdir(source_root):
            messagebox.showerror("Error", "La ruta del Dataset 'Base' no es válida. Selecciónala en el Paso A."); return
        dest_root = filedialog.askdirectory(title="Selecciona una carpeta de destino para el nuevo dataset")
        if not dest_root: return
        
        settings = self._leer_controles_pipeline()
        if settings is None: return 
        if not settings['kernels_dict']:
             messagebox.showwarning("Nada que hacer", "No has seleccionado ningún filtro en la lista D.")
             return

        # --- MODIFICADO: Mover a Hilo (Punto 3) ---
        self.notebook.select(self.tab_entrenamiento) # Cambiar a la pestaña de log
        self.label_animacion.config(text="Generando dataset (individual)...") # Mostrar estado
        
        # Iniciar el hilo de trabajo
        threading.Thread(
            target=self._hilo_generar_individual, 
            args=(settings, source_root, dest_root), 
            daemon=True
        ).start()

    def _hilo_generar_individual(self, settings, source_root, dest_root):
        """(MODIFICADO v5) Hilo de trabajo para la generación individual.
        - Ya NO guarda la 'original' (Punto 1).
        - Añade bucle de 'cantidad_aumentos' (Punto 2).
        """
        try:
            self.cola_gui.put(("log_message", f"--- Iniciando generación (Individual) en: {dest_root} ---"))
            procesados_total = 0
            
            # --- MODIFICADO (Punto 2) ---
            cantidad_aumentos = settings['cantidad_aumentos']
            # Determinar si se aplica rotación. Si N=1, no rotar. Si N>1, rotar.
            aplicar_rotacion = (cantidad_aumentos > 1)
            self.cola_gui.put(("log_message", f"Generando {cantidad_aumentos} aumentos por imagen/filtro."))
            if aplicar_rotacion:
                 self.cola_gui.put(("log_message", "(Rotación aleatoria activada)"))
            # --- FIN MODIFICADO ---

            # Iterar por cada filtro seleccionado
            for nombre_kernel, kernel in settings['kernels_dict'].items():
                self.cola_gui.put(("log_message", f"  Procesando con filtro: {nombre_kernel}"))
                
                for dirpath, _, filenames in os.walk(source_root):
                    valid_files = [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if not valid_files:
                        continue 
                    
                    relative_dir = os.path.relpath(dirpath, source_root)
                    self.cola_gui.put(("log_message", f"    -> Aplicando a carpeta: {relative_dir} ({len(valid_files)} imágenes)"))

                    for filename in valid_files:
                        source_path = os.path.join(dirpath, filename)
                        current_settings = settings.copy() 
                        kernel_list = [kernel] 
                        
                        relative_path = os.path.relpath(source_path, source_root)
                        relative_dir = os.path.dirname(relative_path)
                        base_filename, ext = os.path.splitext(os.path.basename(relative_path))
                        
                        dest_dir_final = os.path.join(dest_root, relative_dir)
                        os.makedirs(dest_dir_final, exist_ok=True)
                        
                        # --- MODIFICADO: Bucle de Aumentos (Punto 2) ---
                        for i in range(cantidad_aumentos):
                            try:
                                nombre_filtro_limpio = nombre_kernel.replace(" ", "-").replace("(", "").replace(")", "")
                                # Nuevo nombre: Pez1_Sobel_aug_0.jpg
                                dest_filename_filtro = f"{base_filename}_{nombre_filtro_limpio}_aug{i}{ext}"
                                dest_path_filtro = os.path.join(dest_dir_final, dest_filename_filtro)
                                
                                with Image.open(source_path) as img:
                                    img_procesada_filtro = self._aplicar_pipeline_a_imagen(
                                        img, 
                                        current_settings, 
                                        kernel_list=kernel_list, 
                                        aplicar_rotacion=aplicar_rotacion
                                    )
                                    img_procesada_filtro.save(dest_path_filtro)
                                    procesados_total += 1
                            except Exception as e:
                                logging.warning(f"No se pudo procesar {source_path} con {nombre_kernel} (aug {i}): {e}")
                        # --- FIN MODIFICADO ---
            
            self.cola_gui.put(("log_message", f"Generadas {procesados_total} imágenes."))
            self.cola_gui.put(("log_message", f"--- Generación (Individual) completada ---"))
            self.cola_gui.put(("generation_complete", (procesados_total, dest_root)))

        except Exception as e:
            logging.error(f"Error fatal en _hilo_generar_individual: {e}", exc_info=True)
            self.cola_gui.put(("show_error", ("Error en Generación", str(e))))

    def _generar_auto(self):
        """(MODIFICADO) Botón principal: Inicia el HILO para "Generar Combinatorio"."""
        logging.info("Iniciando generación automática (Combinatoria)...")
        
        source_root = self.ruta_dataset_preprocesamiento.get()
        if not source_root or not os.path.isdir(source_root):
            messagebox.showerror("Error", "Primero selecciona una 'Carpeta (Base)' en el Paso A."); return
        dest_root = filedialog.askdirectory(title="Selecciona una carpeta de destino para el dataset aumentado")
        if not dest_root: return

        base_settings = self._leer_controles_pipeline()
        if base_settings is None: return
        
        # --- MODIFICADO: Leer el dict (Punto 2) ---
        kernels_dict = base_settings['kernels_dict']
        if not kernels_dict:
            messagebox.showerror("Error", "No has seleccionado ningún filtro en la lista 'D. Filtro' para combinar."); return
        
        # Generar combinaciones de los NOMBRES y KERNELS
        kernel_items = list(kernels_dict.items())
        combinaciones = []
        for r in range(1, len(kernel_items) + 1):
            combinaciones.extend(list(itertools.combinations(kernel_items, r)))
        
        total_combinaciones = len(combinaciones)
        
        if not messagebox.askyesno("Confirmar Generación",
            f"Has seleccionado {len(kernels_dict)} filtros.\n\n"
            f"Esto generará {total_combinaciones} combinaciones de filtros "
            f"(ej. F1, F2, F1+F2, etc.) por CADA imagen original.\n\n"
            f"¿Deseas continuar?"):
            return

        # --- MODIFICADO: Mover a Hilo (Punto 3) ---
        self.notebook.select(self.tab_entrenamiento) # Cambiar a la pestaña de log
        self.label_animacion.config(text="Generando dataset (combinatorio)...") # Mostrar estado

        # Iniciar el hilo de trabajo
        threading.Thread(
            target=self._hilo_generar_auto, 
            args=(base_settings, source_root, dest_root, combinaciones, total_combinaciones), 
            daemon=True
        ).start()

    def _hilo_generar_auto(self, base_settings, source_root, dest_root, combinaciones, total_combinaciones):
        """(MODIFICADO v5) Hilo de trabajo para la generación combinatoria.
        - Ya NO guarda la 'original' (Punto 1).
        - Añade bucle de 'cantidad_aumentos' (Punto 2).
        """
        try:
            self.cola_gui.put(("log_message", f"--- Iniciando generación combinatoria ({total_combinaciones} comb.) en: {dest_root} ---"))
            procesados_total = 0
            
            # --- MODIFICADO (Punto 2) ---
            cantidad_aumentos = base_settings['cantidad_aumentos']
            aplicar_rotacion = (cantidad_aumentos > 1)
            self.cola_gui.put(("log_message", f"Generando {cantidad_aumentos} aumentos por imagen/combinación."))
            if aplicar_rotacion:
                 self.cola_gui.put(("log_message", "(Rotación aleatoria activada)"))
            # --- FIN MODIFICADO ---
            
            # Recolectar imágenes
            source_images_paths = []
            for dirpath, _, filenames in os.walk(source_root):
                for filename in filenames:
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        source_images_paths.append(os.path.join(dirpath, filename))
            self.cola_gui.put(("log_message", f"Encontradas {len(source_images_paths)} imágenes base para procesar."))

            # Iterar por cada COMBINACIÓN
            for i, combo_tuplas in enumerate(combinaciones):
                combo_nombre = "_".join([t[0].replace(" ", "-").replace("(", "").replace(")", "") for t in combo_tuplas])
                combo_kernels = [t[1] for t in combo_tuplas] 

                self.cola_gui.put(("log_message", f"  Procesando Combinación {i+1}/{total_combinaciones} ({combo_nombre})..."))

                # Agrupar por carpeta para loguear
                imagenes_por_carpeta = {}
                for s_path in source_images_paths:
                    rel_dir = os.path.dirname(os.path.relpath(s_path, source_root))
                    if rel_dir not in imagenes_por_carpeta: imagenes_por_carpeta[rel_dir] = []
                    imagenes_por_carpeta[rel_dir].append(s_path)
                
                # Iterar por cada CARPETA
                for relative_dir, image_paths in imagenes_por_carpeta.items():
                    self.cola_gui.put(("log_message", f"    -> Aplicando a carpeta: {relative_dir} ({len(image_paths)} imágenes)"))
                    
                    # Iterar por cada IMAGEN
                    for source_path in image_paths:
                        current_settings = base_settings.copy()
                        base_filename, ext = os.path.splitext(os.path.basename(source_path))
                        dest_dir_final = os.path.join(dest_root, relative_dir)
                        os.makedirs(dest_dir_final, exist_ok=True)

                        # --- MODIFICADO: Bucle de Aumentos (Punto 2) ---
                        for j in range(cantidad_aumentos):
                            try:
                                # Nuevo nombre: Pez1_Enfoque_Sobel_aug_0.jpg
                                dest_filename_combo = f"{base_filename}_{combo_nombre}_aug{j}{ext}"
                                dest_path_combo = os.path.join(dest_dir_final, dest_filename_combo)
                                
                                with Image.open(source_path) as img:
                                    img_procesada_combo = self._aplicar_pipeline_a_imagen(
                                        img, 
                                        current_settings, 
                                        kernel_list=combo_kernels,
                                        aplicar_rotacion=aplicar_rotacion
                                    )
                                    img_procesada_combo.save(dest_path_combo)
                                    procesados_total += 1
                            except Exception as e:
                                logging.warning(f"No se pudo procesar (auto) {source_path} con {combo_nombre} (aug {j}): {e}")
                        # --- FIN MODIFICADO ---

            self.cola_gui.put(("log_message", f"Generadas {procesados_total} imágenes."))
            self.cola_gui.put(("log_message", f"--- Generación combinatoria completada ---"))
            self.cola_gui.put(("generation_complete", (procesados_total, dest_root)))

        except Exception as e:
            logging.error(f"Error fatal en _hilo_generar_auto: {e}", exc_info=True)
            self.cola_gui.put(("show_error", ("Error en Generación", str(e))))

    def _leer_controles_pipeline(self):
        """
        Lee la configuración de la UI. Lee la cantidad de aumentos.
        """
        settings = {}
        try:
            w_str, h_str = self.var_escala.get().lower().split('x')
            settings['escala'] = (int(w_str), int(h_str))
            if settings['escala'][0] <= 0 or settings['escala'][1] <= 0:
                raise ValueError("Dimensiones deben ser positivas")
        except Exception:
            messagebox.showerror("Error de Formato", "El formato de reescalado debe ser 'Ancho x Alto', ej: '50x50'.")
            return None

        # --- MODIFICADO (Punto 2) ---
        try:
            settings['cantidad_aumentos'] = int(self.var_cantidad_aumentos.get())
            if settings['cantidad_aumentos'] < 1:
                raise ValueError("La cantidad debe ser 1 o más")
        except Exception:
            messagebox.showerror("Error de Formato", "La 'Cantidad de Aumentos' debe ser un número entero (ej. 1, 5, 10).")
            return None
        # --- FIN MODIFICADO ---

        settings['modo_color'] = self.var_color.get()
        settings['usar_padding'] = self.var_padding.get()

        settings['kernels_dict'] = OrderedDict() 
        selected_indices = self.filtro_listbox.curselection()
        selected_filtros = [self.filtro_listbox.get(i) for i in selected_indices]
        
        for nombre_filtro in selected_filtros:
            if nombre_filtro == "Manual":
                try:
                    kernel_list = [[float(var.get()) for var in row] for row in self.entries_filtro_manual_vars]
                    settings['kernels_dict']["Manual"] = np.array(kernel_list)
                except Exception as e:
                    messagebox.showerror("Error de Filtro Manual", f"Asegúrate de que todos los valores del filtro manual sean números.\nError: {e}")
                    return None
            else:
                settings['kernels_dict'][nombre_filtro] = KERNELS.get(nombre_filtro)
        
        return settings

    def _aplicar_pipeline_a_imagen(self, pil_img, settings, kernel_list, aplicar_rotacion=False):
        """
        Aplica el pipeline. Acepta un booleano para aplicar rotación.
        """
        procesada_pil = pil_img.copy()
        procesada_pil = procesada_pil.resize(settings['escala'], Image.Resampling.LANCZOS)
        
        # --- MODIFICADO (Punto 2) ---
        if aplicar_rotacion:
            angulo = random.uniform(-15.0, 15.0)
            procesada_pil = procesada_pil.rotate(angulo, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=0)
        # --- FIN MODIFICADO ---

        if settings['modo_color'] == 'gris':
            if procesada_pil.mode != 'L': procesada_pil = procesada_pil.convert('L')
        elif settings['modo_color'] == 'color':
            if procesada_pil.mode != 'RGB': procesada_pil = procesada_pil.convert('RGB')

        matriz_procesada = np.array(procesada_pil)
        
        for kernel in kernel_list:
            if kernel is not None:
                matriz_procesada = convolve_2d_manual(matriz_procesada, kernel, settings['usar_padding'])
        
        if matriz_procesada.ndim == 3:
            return Image.fromarray(matriz_procesada, 'RGB')
        else:
            return Image.fromarray(matriz_procesada, 'L')

    def _seleccionar_carpeta_base(self):
        """
        Manejador robusto para el botón 'Seleccionar Carpeta (Base)'.
        Obtiene el directorio y verifica que sea válido antes de asignarlo
        a la variable de pre-procesamiento.
        """
        # Pedimos el directorio
        directorio = filedialog.askdirectory(initialdir=".")
        
        # filedialog devuelve un string, o un string vacío/None si se cancela.
        # Verificamos que obtuvimos un string y que no está vacío.
        if directorio and isinstance(directorio, str):
            directorio_limpio = directorio.strip() # Limpiar espacios extra
            if os.path.isdir(directorio_limpio):
                self.ruta_dataset_preprocesamiento.set(directorio_limpio)
                logging.info(f"Carpeta base de pre-procesamiento seleccionada: {directorio_limpio}")
            else:
                logging.warning(f"Se seleccionó una ruta inválida: {directorio_limpio}")
        else:
            # User canceló o se obtuvo un resultado inválido
            logging.warning("Selección de carpeta cancelada.")
            
if __name__ == "__main__":
    app = App()
    app.mainloop()