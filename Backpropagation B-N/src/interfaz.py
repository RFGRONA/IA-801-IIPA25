# interfaz.py (Versión final para Backpropagation)

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

# Importar la nueva lógica
from backpropagation import MLP
from procesador_datos import cargar_y_convertir_dataset, convertir_imagen_individual

import logging 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def resource_path(relative_path):
    try: base_path = sys._MEIPASS
    except Exception: base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class TextRedirector:
    def __init__(self, widget): self.widget = widget
    def write(self, text_string):
        self.widget.insert(tk.END, text_string); self.widget.see(tk.END)
    def flush(self): pass

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MLP con Backpropagation - IIPA 2025"); self.geometry("1400x850")
        self.protocol("WM_DELETE_WINDOW", self.cerrar_aplicacion)
        
        self.crear_cabecera()

        self.notebook = ttk.Notebook(self)
        self.tab_entrenamiento = ttk.Frame(self.notebook)
        self.tab_uso = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_entrenamiento, text="Entrenamiento")
        self.notebook.add(self.tab_uso, text="Uso de la Red")
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        
        self.mlp_uso = None
        self.clases_info = OrderedDict()
        self.cola_gui = queue.Queue()

        self.crear_tab_entrenamiento()
        self.crear_tab_uso()
        self.procesar_cola_gui()

    def cerrar_aplicacion(self): self.destroy(); self.quit()

    def crear_cabecera(self):
        # (código sin cambios)
        frame_cabecera = ttk.Frame(self, height=100); frame_cabecera.pack(fill="x", padx=10, pady=5)
        frame_cabecera.grid_columnconfigure(0, weight=1); frame_cabecera.grid_columnconfigure(1, weight=2); frame_cabecera.grid_columnconfigure(2, weight=1)
        label_escudo = ttk.Label(frame_cabecera)
        try:
            img_escudo = Image.open(resource_path("escudo.png")).resize((60, 80), Image.Resampling.LANCZOS)
            foto_escudo = ImageTk.PhotoImage(img_escudo); label_escudo.image = foto_escudo; label_escudo.config(image=foto_escudo)
        except FileNotFoundError: label_escudo.config(text="[Escudo no encontrado]")
        label_escudo.grid(row=0, column=0, sticky="w")
        texto_info = ("Inteligencia Artificial 801 - IIPA 2025\nIngeniería de Sistemas y Computación\n"
                      "Yohan Leon, Oscar Barbosa, Gabriel Martinez")
        ttk.Label(frame_cabecera, text=texto_info, justify="center", font=("Arial", 12)).grid(row=0, column=1)
        label_logo = ttk.Label(frame_cabecera)
        try:
            img_logo = Image.open(resource_path("logo.png")).resize((160, 80), Image.Resampling.LANCZOS)
            foto_logo = ImageTk.PhotoImage(img_logo); label_logo.image = foto_logo; label_logo.config(image=foto_logo)
        except FileNotFoundError: label_logo.config(text="[Logo no encontrado]")
        label_logo.grid(row=0, column=2, sticky="e")

    # --- PESTAÑA DE ENTRENAMIENTO ---
    def crear_tab_entrenamiento(self):
        frame_izquierdo = ttk.Frame(self.tab_entrenamiento, width=400); frame_izquierdo.pack(side="left", fill="y", padx=10, pady=10); frame_izquierdo.pack_propagate(False)
        frame_graficas = ttk.Frame(self.tab_entrenamiento); frame_graficas.pack(side="left", expand=True, fill="both")
        
        frame_config = ttk.LabelFrame(frame_izquierdo, text="Configuración del Modelo MLP")
        frame_config.pack(fill="x", pady=5)
        
        self.ruta_dataset = tk.StringVar(value="./dataset"); self.ruta_targets = tk.StringVar(value="./targets.txt")
        ttk.Button(frame_config, text="Seleccionar Carpeta Dataset", command=lambda: self.ruta_dataset.set(filedialog.askdirectory(initialdir="."))).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Label(frame_config, textvariable=self.ruta_dataset, wraplength=200).grid(row=0, column=1, columnspan=2, sticky="w")
        ttk.Button(frame_config, text="Seleccionar Archivo de Salidas", command=lambda: self.ruta_targets.set(filedialog.askopenfilename(initialdir="."))).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        ttk.Label(frame_config, textvariable=self.ruta_targets, wraplength=200).grid(row=1, column=1, columnspan=2, sticky="w")
        
        ttk.Label(frame_config, text="Neuronas Capa Oculta:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.neuronas_ocultas_var = tk.IntVar(value=15); ttk.Entry(frame_config, textvariable=self.neuronas_ocultas_var, width=10).grid(row=2, column=1, sticky="w", padx=5)
        ttk.Label(frame_config, text="Tasa de Aprendizaje (\u03B1):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.tasa_aprendizaje_var = tk.StringVar(value="0.1"); ttk.Entry(frame_config, textvariable=self.tasa_aprendizaje_var, width=10).grid(row=3, column=1, sticky="w", padx=5)
        ttk.Label(frame_config, text="MSE Deseado:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.error_deseado_var = tk.StringVar(value="0.01"); ttk.Entry(frame_config, textvariable=self.error_deseado_var, width=10).grid(row=4, column=1, sticky="w", padx=5)
        ttk.Label(frame_config, text="Semilla Aleatoria (0=random):").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        self.semilla_var = tk.IntVar(value=0); ttk.Entry(frame_config, textvariable=self.semilla_var, width=10).grid(row=5, column=1, sticky="w", padx=5)
        
        self.momentum_activado = tk.BooleanVar(value=True)
        self.momentum_var = tk.StringVar(value="0.9")
        ttk.Checkbutton(frame_config, text="Activar Momentum:", variable=self.momentum_activado).grid(row=6, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(frame_config, textvariable=self.momentum_var, width=10).grid(row=6, column=1, sticky="w", padx=5)
        
        frame_pesos = ttk.LabelFrame(frame_izquierdo, text="Pesos Iniciales"); frame_pesos.pack(fill="x", pady=5)
        ttk.Label(frame_pesos, text="Bias:").grid(row=0, column=0); r_bias = ttk.Radiobutton(frame_pesos, text="Automático", value=True); r_bias.grid(row=0, column=1); r_bias.invoke()
        ttk.Label(frame_pesos, text="Pesos:").grid(row=1, column=0); r_pesos = ttk.Radiobutton(frame_pesos, text="Automático", state="disabled", value=True); r_pesos.grid(row=1, column=1); r_pesos.invoke()

        self.btn_iniciar = ttk.Button(frame_izquierdo, text="Iniciar Entrenamiento", command=self.iniciar_entrenamiento); self.btn_iniciar.pack(pady=10, fill="x")
        self.btn_cancelar = ttk.Button(frame_izquierdo, text="Cancelar Entrenamiento", command=lambda: setattr(self, 'entrenamiento_cancelado', True), state="disabled"); self.btn_cancelar.pack(pady=5, fill="x")
        
        self.label_animacion = ttk.Label(frame_izquierdo, text="", font=("Arial", 10, "italic")); self.label_animacion.pack(pady=5)
        self.log_texto = tk.Text(frame_izquierdo, height=10); self.log_texto.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(9, 7), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_graficas); self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.limpiar_graficas()

    def iniciar_entrenamiento(self):
        self.btn_iniciar.config(state="disabled"); self.btn_cancelar.config(state="normal")
        self.log_texto.delete("1.0", tk.END); self.limpiar_graficas()
        self.entrenamiento_cancelado = False
        
        self.animacion_carga_activa = True
        self.spinner = itertools.cycle(["Procesando.", "Procesando..", "Procesando..."])
        self.animar_carga()
        
        hilo = threading.Thread(target=self._hilo_entrenamiento, daemon=True)
        hilo.start()

    def _hilo_entrenamiento(self):
        try:
            sys.stdout = TextRedirector(self.log_texto)
            neuronas_ocultas = self.neuronas_ocultas_var.get()
            tasa = float(self.tasa_aprendizaje_var.get())
            error_deseado = float(self.error_deseado_var.get())
            semilla = self.semilla_var.get()
            momentum = float(self.momentum_var.get()) if self.momentum_activado.get() else 0.0

            X, Y, n_in, n_out, invalidos = cargar_y_convertir_dataset(self.ruta_dataset.get(), self.ruta_targets.get())
            if invalidos:
                self.cola_gui.put(("show_error", ("Archivos con Tamaño Inválido", f"Se descartaron: {invalidos}")))
            
            self.clases_info.clear()
            with open(self.ruta_targets.get(), 'r') as f:
                for line in f:
                    if line.startswith('#') or not line.strip(): continue
                    parts = [p.strip() for p in line.strip().split(',')]
                    self.clases_info[parts[0]] = [float(val) for val in parts[1:]]

            self.mlp_actual = MLP(n_in, neuronas_ocultas, n_out, semilla)
            self.mlp_actual.entrenar(X, Y, tasa, error_deseado, momentum, callback=lambda epoca, mse_hist: self.callback_entrenamiento(epoca, mse_hist, X, Y))
            
            self.mlp_actual.guardar_modelo()
            self.cola_gui.put(("entrenamiento_finalizado", "¡Entrenamiento finalizado y modelo guardado!"))

        except Exception as e: self.cola_gui.put(("show_error", ("Error en el Entrenamiento", str(e))))
        finally:
            sys.stdout = sys.__stdout__
            self.cola_gui.put(("restaurar_gui", None))

    def callback_entrenamiento(self, epoca, historial_mse, X_data, Y_data):
        if epoca % 500 == 0 or self.entrenamiento_cancelado or historial_mse[-1] <= float(self.error_deseado_var.get()):
            
            # Calcular matriz de confusión en el hilo de trabajo
            n_clases = len(self.clases_info)
            matriz = np.zeros((n_clases, n_clases))
            nombres_clases = list(self.clases_info.keys())
            
            for x, y_real_vec in zip(X_data, Y_data):
                prediccion_vec = self.mlp_actual.predecir(x)
                idx_real = Y_data.index(y_real_vec) % n_clases if y_real_vec in Y_data else -1
                idx_pred = np.argmax(prediccion_vec)
                
                # Simple mapeo para las 5 vocales
                map_idx_real_a_clase = int(np.argmax(y_real_vec))

                if idx_real != -1:
                    matriz[map_idx_real_a_clase, idx_pred] += 1

            self.cola_gui.put(("actualizar_graficas", {
                "epoca": epoca,
                "historial_mse": historial_mse,
                "matriz_confusion": matriz,
                "nombres_clases": nombres_clases
            }))
        return not self.entrenamiento_cancelado

    def procesar_cola_gui(self):
        try:
            mensaje, datos = self.cola_gui.get_nowait()
            if mensaje == "show_error": messagebox.showerror(datos[0], datos[1])
            elif mensaje == "restaurar_gui":
                self.animacion_carga_activa = False
                self.label_animacion.config(text="")
                self.btn_iniciar.config(state="normal"); self.btn_cancelar.config(state="disabled")
            elif mensaje == "entrenamiento_finalizado": messagebox.showinfo("Información", datos)
            elif mensaje == "actualizar_graficas":
                self.animacion_carga_activa = False
                self.label_animacion.config(text="")
                self.animar_graficas(datos['historial_mse'], datos['epoca'], datos['matriz_confusion'], datos['nombres_clases'])
        except queue.Empty: pass
        finally: self.after(100, self.procesar_cola_gui)
            
    def animar_carga(self):
        if self.animacion_carga_activa:
            self.label_animacion.config(text=next(self.spinner))
            self.after(200, self.animar_carga)
    
    def limpiar_graficas(self):
        for ax in [self.ax1, self.ax2, self.ax3]: ax.clear(); ax.grid(True)
        self.ax1.set_title("MSE vs. Épocas"); self.ax1.set_xlabel("Época"); self.ax1.set_ylabel("MSE")
        self.ax2.set_title("Precisión vs. Épocas"); self.ax2.set_xlabel("Época"); self.ax2.set_ylabel("Precisión")
        self.ax3.set_title("Matriz de Confusión"); self.ax3.set_xlabel("Predicción"); self.ax3.set_ylabel("Real")
        self.canvas.draw()
        
    def animar_graficas(self, historial_mse, epoca_actual, matriz_confusion, nombres_clases):
        self.btn_cancelar.config(state="disabled") # Desactivar durante animación
        
        # Preparar datos
        epoca_inicio = max(0, epoca_actual - 500)
        eje_x = range(epoca_inicio, epoca_actual)
        mse_slice = historial_mse[epoca_inicio:epoca_actual]
        
        mse_inicial_tramo = mse_slice[0] if mse_slice else 1
        precision_slice = [1.0 if mse_inicial_tramo == 0 else 1 - (mse / mse_inicial_tramo) for mse in mse_slice]


        # Iniciar animación de la primera línea
        self.linea_mse, = self.ax1.plot([], [], 'r-')
        self.animar_linea(self.ax1, self.linea_mse, eje_x, mse_slice, 
                          lambda: self.animar_linea(self.ax2, self.ax2.plot([], [], 'g-')[0], eje_x, precision_slice, 
                                                   lambda: self.animar_matriz_confusion(matriz_confusion, nombres_clases)))

    def animar_linea(self, ax, linea, x_data, y_data, on_done, index=0):
        if index >= len(x_data):
            if on_done: on_done()
            return
        
        linea.set_data(x_data[:index+1], y_data[:index+1])
        ax.relim(); ax.autoscale_view()
        self.canvas.draw()
        self.after(5, self.animar_linea, ax, linea, x_data, y_data, on_done, index + 1)

    def animar_matriz_confusion(self, matriz, nombres_clases):
        self.ax3.clear()
        n_clases = len(nombres_clases)
        self.ax3.matshow(matriz, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(n_clases):
            for j in range(n_clases):
                self.ax3.text(x=j, y=i, s=int(matriz[i, j]), va='center', ha='center', size='xx-large')
        
        self.ax3.set_xticks(np.arange(n_clases)); self.ax3.set_yticks(np.arange(n_clases))
        self.ax3.set_xticklabels(nombres_clases); self.ax3.set_yticklabels(nombres_clases)
        self.ax3.set_xlabel("Predicciones"); self.ax3.set_ylabel("Valores Reales"); self.ax3.set_title("Matriz de Confusión")
        self.canvas.draw()
        
        # Cuando termina la última animación, mostrar el mensaje
        mse_final = self.mlp_actual.historial_mse[-1] if self.mlp_actual and self.mlp_actual.historial_mse else 999
        if mse_final <= float(self.error_deseado_var.get()) or self.entrenamiento_cancelado:
            # Ya no se pregunta, el final se muestra en el log
            pass
        else:
            if messagebox.askyesno("Entrenamiento en Pausa", f"Se han completado {self.mlp_actual.epoca} épocas.\n¿Desea continuar?"):
                self.iniciar_entrenamiento() # Simplificado para continuar
            else:
                self.btn_iniciar.config(state="normal")
        
    # --- PESTAÑA DE USO ---
    def crear_tab_uso(self):
        self.canvas_red = tk.Canvas(self.tab_uso, bg="white"); self.canvas_red.pack(side="left", fill="both", expand=True)
        frame_controles = ttk.Frame(self.tab_uso, width=250); frame_controles.pack(side="left", fill="y", padx=10, pady=10); frame_controles.pack_propagate(False)
        ttk.Button(frame_controles, text="Cargar Modelo (.json)", command=self.cargar_modelo_uso).pack(pady=10, fill="x")
        self.btn_predecir_imagen = ttk.Button(frame_controles, text="Predecir Imagen (.png)", command=self.predecir_imagen, state="disabled")
        self.btn_predecir_imagen.pack(pady=5, fill="x")
        self.label_prediccion_binaria = ttk.Label(frame_controles, text="Salida: [?]", font=("Courier", 12)); self.label_prediccion_binaria.pack(pady=20)
        frame_traduccion = ttk.LabelFrame(frame_controles, text="Traducción"); frame_traduccion.pack(fill="x", pady=10)
        self.label_prediccion_letra = ttk.Label(frame_traduccion, text="?", font=("Arial", 24, "bold")); self.label_prediccion_letra.pack(pady=10)

    def cargar_modelo_uso(self):
        ruta = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not ruta: return
        self.mlp_uso = MLP.cargar_modelo(ruta)
        if self.mlp_uso:
            self.btn_predecir_imagen.config(state="normal")
            self.dibujar_red_uso()
            messagebox.showinfo("Éxito", "Modelo cargado correctamente.")
        else:
            messagebox.showerror("Error", "No se pudo cargar el modelo.")

    def predecir_imagen(self):
        if not self.mlp_uso: return
        ruta = filedialog.askopenfilename(filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
        if not ruta: return
        
        try:
            vector_entrada = convertir_imagen_individual(ruta)
            if len(vector_entrada) != self.mlp_uso.neuronas_entrada:
                messagebox.showerror("Error", f"La imagen no tiene el tamaño correcto. Se esperaba un vector de {self.mlp_uso.neuronas_entrada} píxeles.")
                return
            
            salidas_ocultas, salidas_finales = self.mlp_uso._forward_pass(vector_entrada)
            
            self.label_prediccion_binaria.config(text=f"Salida: {[round(s, 2) for s in salidas_finales]}")
            
            # Traducción
            idx_pred = np.argmax(salidas_finales)
            letra_predicha = list(self.clases_info.keys())[idx_pred]
            self.label_prediccion_letra.config(text=letra_predicha)
            
            self.dibujar_red_uso(salidas_ocultas, salidas_finales)
        except Exception as e:
            messagebox.showerror("Error al Predecir", str(e))

    def dibujar_red_uso(self, salidas_ocultas=None, salidas_finales=None):
        self.canvas_red.delete("all")
        if not self.mlp_uso: return
        w, h = self.canvas_red.winfo_width(), self.canvas_red.winfo_height()

        # Posiciones de las capas
        x_in, x_hidden, x_out = w * 0.1, w * 0.5, w * 0.9
        
        # Capa de Entrada
        self.canvas_red.create_oval(x_in-20, h/2-20, x_in+20, h/2+20, fill="lightgray")
        self.canvas_red.create_text(x_in, h/2, text=f"{self.mlp_uso.neuronas_entrada}\nEntradas")

        # Capa Oculta
        y_step_h = h / (self.mlp_uso.neuronas_ocultas + 1)
        for j in range(self.mlp_uso.neuronas_ocultas):
            y_h = y_step_h * (j + 1)
            self.canvas_red.create_oval(x_hidden-15, y_h-15, x_hidden+15, y_h+15, fill="lightblue")
            # Conexiones y pesos IH
            for i in range(self.mlp_uso.neuronas_entrada):
                self.canvas_red.create_line(x_in, h/2, x_hidden, y_h, fill="gray")
        
        # Capa de Salida
        y_step_o = h / (self.mlp_uso.neuronas_salida + 1)
        for k in range(self.mlp_uso.neuronas_salida):
            y_o = y_step_o * (k + 1)
            self.canvas_red.create_oval(x_out-15, y_o-15, x_out+15, y_o+15, fill="lightgreen")
            # Conexiones y pesos HO
            for j in range(self.mlp_uso.neuronas_ocultas):
                y_h = y_step_h * (j + 1)
                self.canvas_red.create_line(x_hidden, y_h, x_out, y_o, fill="gray")
                peso = self.mlp_uso.pesos_ho[k][j]
                self.canvas_red.create_text((x_hidden+x_out)/2, (y_h+y_o)/2, text=f"{peso:.1f}", font=("Arial", 7))
        
if __name__ == "__main__":
    app = App()
    app.mainloop()