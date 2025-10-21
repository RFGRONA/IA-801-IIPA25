# interfaz.py (Versión con motor de animación rediseñado)

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
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
        
        # <--- CAMBIO: historiales para las métricas de validación ---
        self.mlp_actual = None
        self.historial_mse_train, self.historial_mse_val = [], []
        self.historial_matrices = [] # Guardará las matrices de validación
        self.epoca_inicial_bloque = 0
        self.X_train, self.Y_train, self.X_val, self.Y_val = [], [], [], []
        self.nombres_clases = []
        self.hilo_entrenamiento = None; self.entrenamiento_cancelado = False
        self.animacion_activa = False
        
        # <--- CAMBIO: Nuevas líneas para las gráficas de validación ---
        self.linea_mse_train, self.linea_mse_val = None, None
        self.linea_precision_train, self.linea_precision_val = None, None
        
        self.crear_cabecera()
        self.notebook = ttk.Notebook(self); self.tab_entrenamiento = ttk.Frame(self.notebook); self.tab_uso = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_entrenamiento, text="Entrenamiento"); self.notebook.add(self.tab_uso, text="Uso de la Red")
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        
        self.mlp_uso = None; 
        self.clases_info = OrderedDict()
        self.clases_info_uso = OrderedDict()
        self.cola_gui = queue.Queue()
        self.crear_tab_entrenamiento(); self.crear_tab_uso(); self.procesar_cola_gui()
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
        self.ruta_dataset = tk.StringVar(value="./dataset"); self.ruta_targets = tk.StringVar(value="./targets.txt")
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

        ttk.Label(frame_config, text="División Dataset (% Entr.):").grid(row=8, column=0, sticky="w", padx=5, pady=5)
        self.division_var = tk.IntVar(value=80)
        self.division_label_var = tk.StringVar(value=f"{self.division_var.get()}% / {100-self.division_var.get()}%")
        
        frame_slider = ttk.Frame(frame_config)
        frame_slider.grid(row=8, column=1, columnspan=2, sticky="ew")
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

            self.X_train, self.Y_train, self.X_val, self.Y_val, n_in, n_out, _, self.rutas_imagenes_totales = cargar_y_convertir_dataset(
                self.ruta_dataset.get(), 
                self.ruta_targets.get(),
                porcentaje_entrenamiento,
                semilla=self.semilla_var.get()
            )
            
            if not self.X_train: messagebox.showerror("Error", "No se cargaron datos de entrenamiento."); return

            self.clases_info.clear()
            with open(self.ruta_targets.get(), 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip(): continue
                    parts = [p.strip() for p in line.strip().split(',')]; self.clases_info[parts[0]] = [float(val) for val in parts[1:]]
            self.nombres_clases = list(self.clases_info.keys())
            neuronas_ocultas = self.neuronas_ocultas_var.get()
            self.mlp_actual = MLP(n_in, neuronas_ocultas, n_out, self.semilla_var.get())

            resumen_inicial = (
                f"--- INICIO DEL ENTRENAMIENTO ---\n"
                f" Arquitectura de la Red:\n"
                f"   - Neuronas de Entrada: {n_in}\n"
                f"   - Neuronas Ocultas:    {neuronas_ocultas}\n"
                f"   - Neuronas de Salida:  {n_out}\n"
                f" Hiperparámetros:\n"
                f"   - Tasa de Aprendizaje (α): {self.tasa_aprendizaje_var.get()}\n"
                f"   - Momentum (η):              {self.momentum_var.get() if self.momentum_activado.get() else 'Desactivado'}\n"
                f"   - MSE Deseado:             {self.error_deseado_var.get()}\n"
                f" Dataset:\n"
                f"   - Patrones Totales:      {len(self.X_train) + len(self.X_val)}\n"
                f"   - División:              {self.division_var.get()}% Entrenamiento / {100-self.division_var.get()}% Validación\n"
                f"   - Patrones Entrenamiento: {len(self.X_train)}\n"
                f"   - Patrones Validación:    {len(self.X_val)}\n"
                f"----------------------------------"
            )
            self.log_to_console(resumen_inicial)

            self.continuar_entrenamiento()
        except Exception as e:
            messagebox.showerror("Error al Iniciar", str(e))

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
            self.animar_carga(stop=True)
            if mensaje == "show_error":
                messagebox.showerror(datos[0], datos[1]); self.detener_entrenamiento()
            elif mensaje == "progress_update":
                self.label_animacion.config(text=f"Entrenando... Época {datos}")
            elif mensaje == "log_message":
                self.log_to_console(datos) 
            elif mensaje in ["bloque_finalizado", "entrenamiento_finalizado"]:
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
        Crea una imagen de previsualización con tamaño dinámico.
        - Agranda imágenes pequeñas (como las vocales) de forma nítida.
        - Encoge imágenes grandes manteniendo su proporción.
        """
        # Definimos el tamaño máximo del contenedor de la previsualización
        max_width = 150
        max_height = 200
        
        original_width, original_height = pil_image.size
        
        # --- LÓGICA PARA IMÁGENES PEQUEÑAS (ej. vocales de 7x5) ---
        if original_width < 50 and original_height < 50:
            # Agrandamos la imagen para que sea visible, usando NEAREST para mantener los píxeles nítidos.
            # Un factor de 20 hará que una imagen de 5x7 se vea como 100x140.
            zoom_factor = 20 
            new_width = original_width * zoom_factor
            new_height = original_height * zoom_factor
            # Asegurarnos de que no exceda el máximo
            if new_width > max_width or new_height > max_height:
                zoom_factor = min(max_width // original_width, max_height // original_height)
            
            preview_img = pil_image.resize(
                (original_width * zoom_factor, original_height * zoom_factor),
                Image.Resampling.NEAREST
            )
        # --- LÓGICA PARA IMÁGENES GRANDES ---
        else:
            # Hacemos una copia para no modificar la imagen original
            preview_img = pil_image.copy()
            # El método .thumbnail() es perfecto: reduce la imagen para que quepa en el
            # cuadro (max_width, max_height) MANTENIENDO LA PROPORCIÓN.
            preview_img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)

        return ImageTk.PhotoImage(preview_img)
            
if __name__ == "__main__":
    app = App()
    app.mainloop()