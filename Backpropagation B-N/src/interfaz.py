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
        
        self.mlp_actual = None; self.historial_mse = []; self.epoca_inicial_bloque = 0
        self.X_train, self.Y_train = [], []; self.nombres_clases = []
        self.hilo_entrenamiento = None; self.entrenamiento_cancelado = False
        self.linea_mse = None; self.linea_precision = None
        self.animacion_activa = False
        
        self.crear_cabecera()
        self.notebook = ttk.Notebook(self); self.tab_entrenamiento = ttk.Frame(self.notebook); self.tab_uso = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_entrenamiento, text="Entrenamiento"); self.notebook.add(self.tab_uso, text="Uso de la Red")
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        
        # --- CORRECCIÓN AQUÍ ---
        self.mlp_uso = None
        # Esta variable es para el entrenamiento
        self.clases_info = OrderedDict() 
        self.clases_info_uso = OrderedDict()
        
        self.cola_gui = queue.Queue()

        self.crear_tab_entrenamiento(); self.crear_tab_uso(); self.procesar_cola_gui()
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

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
            img_escudo = Image.open("escudo.png").resize((60, 80), Image.Resampling.LANCZOS)
            foto_escudo = ImageTk.PhotoImage(img_escudo); label_escudo.image = foto_escudo; label_escudo.config(image=foto_escudo)
        except Exception: label_escudo.config(text="[Escudo]")
        label_escudo.grid(row=0, column=0, sticky="w")
        texto_info = ("Inteligencia Artificial 801 - IIPA 2025\nIngeniería de Sistemas y Computación\n"
                      "Yohan Leon, Oscar Barbosa, Gabriel Martinez")
        ttk.Label(frame_cabecera, text=texto_info, justify="center", font=("Arial", 12)).grid(row=0, column=1)
        label_logo = ttk.Label(frame_cabecera)
        try:
            img_logo = Image.open("logo.png").resize((160, 80), Image.Resampling.LANCZOS)
            foto_logo = ImageTk.PhotoImage(img_logo); label_logo.image = foto_logo; label_logo.config(image=foto_logo)
        except Exception: label_logo.config(text="[Logo]")
        label_logo.grid(row=0, column=2, sticky="e")
        
    def crear_tab_entrenamiento(self):
        # (código sin cambios)
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
            self.X_train, self.Y_train, n_in, n_out, _ = cargar_y_convertir_dataset(self.ruta_dataset.get(), self.ruta_targets.get())
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
                f"   - Patrones de Entrenamiento: {len(self.X_train)}\n"
                f"----------------------------------"
            )
            self.log_to_console(resumen_inicial)

            self.continuar_entrenamiento()
        except Exception as e:
            messagebox.showerror("Error al Iniciar", str(e))

    def continuar_entrenamiento(self):
        self.btn_iniciar.config(state="disabled"); self.btn_cancelar.config(state="normal"); self.entrenamiento_cancelado = False
        self.spinner = itertools.cycle([f"Entrenando... Época {self.epoca_inicial_bloque+1}", f"Entrenando... Época {self.epoca_inicial_bloque+1}.", f"Entrenando... Época {self.epoca_inicial_bloque+1}.."])
        self.animar_carga(start=True)
        self.hilo_entrenamiento = threading.Thread(target=self._hilo_entrenamiento_bloque, daemon=True); self.hilo_entrenamiento.start()

    def _hilo_entrenamiento_bloque(self):
        try:
            params = (
                self.clases_info,
                float(self.tasa_aprendizaje_var.get()),
                float(self.error_deseado_var.get()),
                float(self.momentum_var.get()) if self.momentum_activado.get() else 0.0
            )
            # <--- CAMBIO: Ya no se pasa 'log_callback', ahora se recibe 'log_bloque'
            epoca, historial_mse, historial_matrices, log_bloque, completo = self.mlp_actual.entrenar_bloque(
                self.X_train, self.Y_train, *params,
                epoca_inicio=self.epoca_inicial_bloque,
                max_epocas_bloque=self.epocas_bloque_var.get(),
                cancel_event=lambda: self.entrenamiento_cancelado
            )
            resultado = {
                "epoca_final": epoca,
                "historial_mse_bloque": historial_mse,
                "historial_matrices_bloque": historial_matrices,
                "log_del_bloque": log_bloque # <--- CAMBIO: Se añade la lista de logs al resultado
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
        self.entrenamiento_cancelado = True
        self.animar_carga(stop=True)
        
        # Guardar el modelo
        if self.mlp_actual:
            logging.info("Guardando estado actual del modelo...")
            self.mlp_actual.guardar_modelo()
            self.label_animacion.config(text="Entrenamiento detenido. ¡Modelo guardado!")
        else:
            self.label_animacion.config(text="Entrenamiento detenido.")

        self.dibujar_estado_final_graficas()
        
        # <--- CAMBIO: Generar y mostrar el resumen final ---
        if self.historial_mse:
            mse_final = self.historial_mse[-1]
            matriz_final = self.historial_matrices[-1]
            precision_final = np.trace(matriz_final) / len(self.X_train)
            
            resumen_final = (
                f"--- FIN DEL ENTRENAMIENTO ---\n"
                f" Resultados Finales:\n"
                f"   - Épocas Totales Entrenadas: {self.epoca_inicial_bloque}\n"
                f"   - MSE Final:                 {mse_final:.6f}\n"
                f"   - Precisión Final:           {precision_final:.2%}\n"
                f"----------------------------------"
            )
            self.log_to_console(resumen_final)

        self.btn_iniciar.config(state="normal")
        self.btn_cancelar.config(state="disabled")

    # --- CAMBIO: Nueva función helper para escribir en la consola ---
    def log_to_console(self, message):
        """Inserta un mensaje en el widget de la consola de forma segura."""
        self.log_consola.config(state="normal")
        self.log_consola.insert(tk.END, message + "\n")
        self.log_consola.see(tk.END)
        self.log_consola.config(state="disabled")

    def animar_carga(self, start=False, stop=False):
        if stop: self.animacion_activa = False; return
        if start: self.animacion_activa = True
        if self.animacion_activa:
            self.label_animacion.config(text=next(self.spinner)); self.after(200, self.animar_carga)

    def limpiar_graficas(self):
        self.epoca_inicial_bloque = 0; self.historial_mse = []; self.historial_matrices = []
        self.linea_mse = None; self.linea_precision = None; self.textos_matriz = []
        for ax in [self.ax1, self.ax2, self.ax3]: ax.clear(); ax.grid(True)
        self.ax1.set_title("MSE vs. Épocas"); self.ax1.set_xlabel("Época"); self.ax1.set_ylabel("MSE")
        self.ax2.set_title("Precisión vs. Épocas"); self.ax2.set_xlabel("Época"); self.ax2.set_ylabel("Precisión")
        self.ax3.set_title("Matriz de Confusión")
        # <--- CAMBIO: Oculta los ejes cartesianos de la matriz al inicio
        self.ax3.set_xticks([]); self.ax3.set_yticks([])
        self.canvas.draw()
        
    def animar_graficas(self, datos_resultado, on_done_callback):
        epoca_final_anterior = self.epoca_inicial_bloque
        self.epoca_inicial_bloque = datos_resultado["epoca_final"]
        self.historial_mse.extend(datos_resultado["historial_mse_bloque"])
        self.historial_matrices.extend(datos_resultado["historial_matrices_bloque"])
        
        eje_x_nuevo = range(epoca_final_anterior + 1, self.epoca_inicial_bloque + 1)
        mse_nuevo = datos_resultado["historial_mse_bloque"]
        
        self.animacion_activa = True
        self._bucle_animacion(eje_x_nuevo, mse_nuevo, datos_resultado["historial_matrices_bloque"], 0, on_done_callback)

    def _bucle_animacion(self, x_data, mse_data, matrices_data, index, on_done_callback):
        if not self.animacion_activa or index >= len(x_data):
            if self.animacion_activa: self.dibujar_estado_final_graficas(); self.after(100, on_done_callback)
            self.animacion_activa = False; return

        step = 25; delay_ms = 15; MUESTRAS_MATRIZ = 25 # 
        
        # 1. Actualizar las líneas
        if not self.linea_mse: self.linea_mse, = self.ax1.plot([], [], 'r-')
        if not self.linea_precision: self.linea_precision, = self.ax2.plot([], [], 'g-')
        
        next_index = min(index + step, len(x_data))
        self.linea_mse.set_data(np.append(self.linea_mse.get_xdata(), x_data[index:next_index]), np.append(self.linea_mse.get_ydata(), mse_data[index:next_index]))
        precision_data_segmento = [1 - mse for mse in mse_data[index:next_index]]
        self.linea_precision.set_data(np.append(self.linea_precision.get_xdata(), x_data[index:next_index]), np.append(self.linea_precision.get_ydata(), precision_data_segmento))
        
        # 2. Actualizar la matriz de confusión con la data pre-calculada
        # <--- CAMBIO: Se busca la matriz pre-calculada que corresponde a la época actual
        indice_matriz = (x_data[index] - 1) // MUESTRAS_MATRIZ
        if indice_matriz < len(self.historial_matrices):
            self.dibujar_matriz_confusion_estatica(self.historial_matrices[indice_matriz], epoca_actual=x_data[index])

        self.ax1.relim(); self.ax1.autoscale_view(); self.ax2.relim(); self.ax2.autoscale_view()
        self.canvas.draw()
        self.after(delay_ms, self._bucle_animacion, x_data, mse_data, matrices_data, next_index, on_done_callback)

    def dibujar_matriz_confusion_estatica(self, matriz, epoca_actual=None):
        self.ax3.clear()
        if epoca_actual: self.ax3.set_title(f"Matriz de Confusión (Época {epoca_actual})")
        else: self.ax3.set_title("Matriz de Confusión")
        n_clases = len(self.nombres_clases)
        self.ax3.matshow(matriz, cmap=plt.cm.Blues, alpha=0.7); self.ax3.set_xticks(np.arange(n_clases)); self.ax3.set_yticks(np.arange(n_clases))
        self.ax3.set_xticklabels(self.nombres_clases); self.ax3.set_yticklabels(self.nombres_clases)
        for i in range(n_clases):
            for j in range(n_clases):
                self.ax3.text(x=j, y=i, s=int(matriz[i, j]), va='center', ha='center', size='large')
    
    def dibujar_estado_final_graficas(self):
        if not self.historial_mse: return
        eje_x = range(1, len(self.historial_mse) + 1); precision = [1 - mse for mse in self.historial_mse]
        if not self.linea_mse: self.linea_mse, = self.ax1.plot(eje_x, self.historial_mse, 'r-')
        else: self.linea_mse.set_data(eje_x, self.historial_mse)
        if not self.linea_precision: self.linea_precision, = self.ax2.plot(eje_x, precision, 'g-')
        else: self.linea_precision.set_data(eje_x, precision)
        if self.historial_matrices: self.dibujar_matriz_confusion_estatica(self.historial_matrices[-1], epoca_actual=len(self.historial_mse))
        self.ax1.relim(); self.ax1.autoscale_view(); self.ax2.relim(); self.ax2.autoscale_view()
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
        targets_cargados = False

        try:
            with open("targets.txt", 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip(): continue
                    parts = [p.strip() for p in line.strip().split(',')]
                    self.clases_info_uso[parts[0]] = [float(val) for val in parts[1:]]
            targets_cargados = True
            logging.info("Archivo 'targets.txt' cargado exitosamente.")
        except FileNotFoundError:
            messagebox.showerror("Error", "No se encontró el archivo 'targets.txt'. Es necesario para las predicciones.", parent=self.tab_uso)
        except Exception as e:
            messagebox.showerror("Error", f"Error al leer 'targets.txt': {e}", parent=self.tab_uso)

        try:
            self.mlp_uso = MLP.cargar_modelo("modelo_mlp.json")
            if self.mlp_uso:
                modelo_cargado = True
                logging.info("Archivo 'modelo_mlp.json' cargado exitosamente.")
                self.dibujar_red_uso()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar 'modelo_mlp.json': {e}", parent=self.tab_uso)

        if modelo_cargado and targets_cargados:
            self.btn_predecir_imagen.config(state="normal")
        else:
            self.btn_predecir_imagen.config(state="disabled")

    def predecir_imagen(self):
        if not self.mlp_uso or not self.clases_info_uso:
            messagebox.showwarning("Recursos no cargados", "Asegúrese de que el modelo y los targets estén cargados.")
            return
            
        ruta = filedialog.askopenfilename(filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if not ruta: return
        
        try:
            img = Image.open(ruta)
            img_display = img.resize((100, 140), Image.Resampling.NEAREST) 
            photo = ImageTk.PhotoImage(img_display)
            self.label_imagen_predecida.config(image=photo)
            self.label_imagen_predecida.image = photo 
            vector_entrada = convertir_imagen_individual(ruta)
            if len(vector_entrada) != self.mlp_uso.neuronas_entrada:
                messagebox.showerror("Error", f"La imagen no tiene el tamaño correcto. Se esperaba un vector de {self.mlp_uso.neuronas_entrada} píxeles.")
                return
            
            _, salidas_finales = self.mlp_uso._forward_pass(vector_entrada)
            self.label_prediccion_binaria.config(text=f"Salida: {[round(s, 2) for s in salidas_finales]}")

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
        
if __name__ == "__main__":
    app = App()
    app.mainloop()