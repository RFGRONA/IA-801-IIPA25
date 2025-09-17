# interfaz.py (Versión final con todas las mejoras)

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import random
import sys
import os

from adaline import Adaline, generar_datos_binarios

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def resource_path(relative_path):
    try: base_path = sys._MEIPASS
    except Exception: base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class TextRedirector:
    def __init__(self, widget): self.widget = widget
    def write(self, text_string):
        self.widget.insert(tk.END, text_string); self.widget.see(tk.END); self.widget.update_idletasks()
    def flush(self): pass

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Adaline Interactivo - IIPA 2025"); self.geometry("1400x800")
        self.protocol("WM_DELETE_WINDOW", self.cerrar_aplicacion)
        self.crear_cabecera()

        self.notebook = ttk.Notebook(self)
        self.tab_entrenamiento = ttk.Frame(self.notebook)
        self.tab_uso = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_entrenamiento, text="Entrenamiento"); self.notebook.add(self.tab_uso, text="Uso de la Red")
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        
        self.entrenamiento_cancelado = False
        self.adaline_actual = None
        self.adaline_uso = None
        self.historial_mse = []; self.historial_precision = []

        self.crear_tab_entrenamiento()
        self.crear_tab_uso()
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)
        self.after(100, self.cargar_modelo_para_uso)

    def cerrar_aplicacion(self): self.destroy(); self.quit()

    def crear_cabecera(self):
        frame_cabecera = ttk.Frame(self, height=100); frame_cabecera.pack(fill="x", padx=10, pady=5)
        frame_cabecera.grid_columnconfigure(0, weight=1); frame_cabecera.grid_columnconfigure(1, weight=2); frame_cabecera.grid_columnconfigure(2, weight=1)
        label_escudo = ttk.Label(frame_cabecera)
        try:
            img_escudo = Image.open(resource_path("escudo.png")).resize((60, 80), Image.Resampling.LANCZOS)
            foto_escudo = ImageTk.PhotoImage(img_escudo)
            label_escudo.image = foto_escudo; label_escudo.config(image=foto_escudo)
        except FileNotFoundError: label_escudo.config(text="[Escudo no encontrado]")
        label_escudo.grid(row=0, column=0, sticky="w")
        texto_info = ("Inteligencia Artificial 801 - IIPA 2025\nIngeniería de Sistemas y Computación\n"
                      "Yohan Leon, Oscar Barbosa, Gabriel Martinez")
        ttk.Label(frame_cabecera, text=texto_info, justify="center", font=("Arial", 12)).grid(row=0, column=1)
        label_logo = ttk.Label(frame_cabecera)
        try:
            img_logo = Image.open(resource_path("logo.png")).resize((160, 80), Image.Resampling.LANCZOS)
            foto_logo = ImageTk.PhotoImage(img_logo)
            label_logo.image = foto_logo; label_logo.config(image=foto_logo)
        except FileNotFoundError: label_logo.config(text="[Logo no encontrado]")
        label_logo.grid(row=0, column=2, sticky="e")

    def crear_tab_entrenamiento(self):
        frame_izquierdo = ttk.Frame(self.tab_entrenamiento, width=350); frame_izquierdo.pack(side="left", fill="y", padx=10, pady=10); frame_izquierdo.pack_propagate(False)
        frame_graficas = ttk.Frame(self.tab_entrenamiento); frame_graficas.pack(side="left", expand=True, fill="both")
        
        frame_config = ttk.LabelFrame(frame_izquierdo, text="Configuración del Modelo Adaline"); frame_config.pack(fill="x", pady=5)
        ttk.Label(frame_config, text="Número de bits de entrada:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.num_bits_var = tk.IntVar(value=2)
        self.num_bits_selector = ttk.Combobox(frame_config, textvariable=self.num_bits_var, values=[2, 3, 4, 5, 6, 7, 8], state="readonly", width=8)
        self.num_bits_selector.grid(row=0, column=1, padx=5, pady=5); self.num_bits_selector.bind("<<ComboboxSelected>>", self.actualizar_panel_pesos)
        ttk.Label(frame_config, text="Tasa de Aprendizaje (\u03B1):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.tasa_aprendizaje_var = tk.StringVar(value="0.01"); ttk.Entry(frame_config, textvariable=self.tasa_aprendizaje_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(frame_config, text="MSE Deseado (ej: 0.01):").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.error_deseado_var = tk.StringVar(value="0.01"); ttk.Entry(frame_config, textvariable=self.error_deseado_var, width=10).grid(row=2, column=1, padx=5, pady=5)

        self.frame_pesos = ttk.LabelFrame(frame_izquierdo, text="Pesos Iniciales"); self.frame_pesos.pack(fill="x", pady=10)
        self.entradas_pesos_manuales = []; self.actualizar_panel_pesos()

        self.btn_iniciar = ttk.Button(frame_izquierdo, text="Iniciar Entrenamiento", command=self.iniciar_entrenamiento); self.btn_iniciar.pack(pady=10, fill="x")
        self.btn_cancelar = ttk.Button(frame_izquierdo, text="Cancelar Entrenamiento", command=lambda: setattr(self, 'entrenamiento_cancelado', True), state="disabled"); self.btn_cancelar.pack(pady=5, fill="x")
        
        frame_log = ttk.LabelFrame(frame_izquierdo, text="Salida del Entrenamiento"); frame_log.pack(fill="both", expand=True)
        self.log_texto = tk.Text(frame_log, height=8); self.log_texto.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(9, 7), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_graficas); self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.limpiar_graficas()

    def actualizar_panel_pesos(self, event=None):
        for widget in self.frame_pesos.winfo_children(): widget.destroy()
        self.entradas_pesos_manuales.clear(); num_bits = self.num_bits_var.get()
        self.pesos_mode_var = tk.StringVar(value="auto")
        ttk.Label(self.frame_pesos, text="Modo:").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Radiobutton(self.frame_pesos, text="Automático", variable=self.pesos_mode_var, value="auto", command=self.toggle_pesos_entries).grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(self.frame_pesos, text="Manual", variable=self.pesos_mode_var, value="manual", command=self.toggle_pesos_entries).grid(row=0, column=2, sticky="w")
        ttk.Label(self.frame_pesos, text="Bias (w0):").grid(row=1, column=0, sticky="w", pady=2)
        entry_bias = ttk.Entry(self.frame_pesos, state="disabled", width=8); entry_bias.grid(row=1, column=1, columnspan=2, sticky="w")
        self.entradas_pesos_manuales.append(entry_bias)
        for i in range(num_bits):
            ttk.Label(self.frame_pesos, text=f"Peso w{i+1}:").grid(row=i+2, column=0, sticky="w", pady=2)
            entry = ttk.Entry(self.frame_pesos, state="disabled", width=8); entry.grid(row=i+2, column=1, columnspan=2, sticky="w")
            self.entradas_pesos_manuales.append(entry)

    def toggle_pesos_entries(self):
        state = "normal" if self.pesos_mode_var.get() == "manual" else "disabled"
        for entry in self.entradas_pesos_manuales: entry.config(state=state)

    def limpiar_graficas(self):
        self.ax1.clear(); self.ax1.grid(True); self.ax1.set_title("Error Cuadrático Medio (MSE) vs. Épocas"); self.ax1.set_xlabel("Época"); self.ax1.set_ylabel("MSE")
        self.linea_mse, = self.ax1.plot([], [], 'r-', label='MSE')
        self.ax2.clear(); self.ax2.grid(True); self.ax2.set_title("Precisión vs. Épocas"); self.ax2.set_xlabel("Época"); self.ax2.set_ylabel("Precisión (0 a 1)")
        self.linea_precision, = self.ax2.plot([], [], 'g-', label='Precisión')
        self.ax3.clear(); self.ax3.grid(True); self.ax3.set_title("Predicción vs. Realidad"); self.ax3.set_xlabel("Valor Real"); self.ax3.set_ylabel("Valor Predicho")
        self.canvas.draw()
        
    def iniciar_entrenamiento(self):
        self.btn_iniciar.config(state="disabled"); self.btn_cancelar.config(state="normal"); self.entrenamiento_cancelado = False
        original_stdout = sys.stdout; sys.stdout = TextRedirector(self.log_texto)
        try:
            num_bits = self.num_bits_var.get(); tasa = float(self.tasa_aprendizaje_var.get()); error_deseado = float(self.error_deseado_var.get())
            X_data, Y_data = generar_datos_binarios(num_bits)
            self.adaline_actual = Adaline(num_entradas=num_bits, tasa_aprendizaje=tasa)
            self.log_texto.delete("1.0", tk.END); self.limpiar_graficas(); self.historial_mse = []; self.historial_precision = []
            
            # --- CORRECCIÓN 1: Mostrar resumen de datos iniciales en la consola de la GUI ---
            print("--- Parámetros de Entrenamiento ---")
            print(f"Problema: {num_bits} bits a decimal")
            print(f"Tasa de Aprendizaje (α): {tasa}")
            print(f"MSE Deseado: {error_deseado}")
            print(f"Pesos Iniciales (Wi): {[round(w, 4) for w in self.adaline_actual.pesos]}")
            print("-------------------------------------\n")
            
            self.ax3.plot([min(Y_data), max(Y_data)], [min(Y_data), max(Y_data)], 'k--', label='Predicción Perfecta')
            self.puntos_prediccion, = self.ax3.plot(Y_data, [0]*len(Y_data), 'bo', label='Predicciones')
            self.ax3.legend()
            
            pesos_finales = self.adaline_actual.entrenar(X_data, Y_data, error_deseado, callback=self.actualizar_en_tiempo_real)
            
            if self.historial_mse:
                mse_final = self.historial_mse[-1]; precision_final = self.historial_precision[-1]
                print(f"\n--- Resumen Final ---"); print(f"Pesos: {[round(w, 4) for w in pesos_finales]}")
                print(f"MSE Final: {mse_final:.6f}"); print(f"Precisión Final: {precision_final:.2%}")

            self.adaline_actual.guardar_pesos()
            if not self.entrenamiento_cancelado: messagebox.showinfo("Éxito", "Entrenamiento finalizado y pesos guardados.")
        except ValueError: messagebox.showerror("Error de Entrada", "Por favor, revise los valores numéricos.")
        except Exception as e: messagebox.showerror("Error Inesperado", f"Ha ocurrido un error: {e}")
        finally:
            sys.stdout = original_stdout
            self.btn_iniciar.config(state="normal"); self.btn_cancelar.config(state="disabled")

    def actualizar_en_tiempo_real(self, epoca, pesos, mse, mse_inicial):
        precision = 1 - (mse / mse_inicial) if mse_inicial > 0 else 1.0
        self.historial_mse.append(mse); self.historial_precision.append(precision)
        eje_x = range(1, epoca + 1)

        self.ax1.set_title(f"MSE vs. Épocas (Época {epoca})"); self.linea_mse.set_data(eje_x, self.historial_mse)
        self.ax1.relim(); self.ax1.autoscale_view()
        self.linea_precision.set_data(eje_x, self.historial_precision); self.ax2.relim(); self.ax2.autoscale_view()
        
        num_bits = self.num_bits_var.get(); X_data, Y_data = generar_datos_binarios(num_bits)
        predicciones = [self.adaline_actual.predecir(x) for x in X_data]
        self.puntos_prediccion.set_ydata(predicciones)
        self.ax3.relim(); self.ax3.autoscale_view()
        
        self.canvas.draw(); self.update()
        
        if epoca > 0 and epoca % 500 == 0:
            if not messagebox.askyesno("Entrenamiento en Pausa", f"Se han completado {epoca} épocas.\n¿Desea continuar?"):
                self.entrenamiento_cancelado = True
        return not self.entrenamiento_cancelado

    def crear_tab_uso(self):
        self.canvas_red = tk.Canvas(self.tab_uso, bg="white"); self.canvas_red.pack(side="left", fill="both", expand=True)
        self.frame_uso_controles = ttk.Frame(self.tab_uso, width=250)
        self.frame_uso_controles.pack(side="left", fill="y", padx=10, pady=10); self.frame_uso_controles.pack_propagate(False)
        self.botones_entrada_uso = []
        
    def cargar_modelo_para_uso(self):
        pesos_cargados = Adaline.cargar_pesos()
        num_bits_guardado = len(pesos_cargados) - 1 if pesos_cargados else 0
        if pesos_cargados and num_bits_guardado > 0:
            self.adaline_uso = Adaline(num_entradas=num_bits_guardado)
            self.adaline_uso.pesos = pesos_cargados
            self.reconstruir_panel_uso()
        else:
            self.canvas_red.delete("all")
            for widget in self.frame_uso_controles.winfo_children(): widget.destroy()
            ttk.Label(self.frame_uso_controles, text="No hay un modelo entrenado.").pack(pady=20)
            self.canvas_red.create_text(300, 200, text="Entrene un modelo para usar esta pestaña.", font=("Arial", 14))

    def toggle_input_uso(self, var, btn, index):
        new_value = 1 - var.get(); var.set(new_value)
        num_bits = self.adaline_uso.num_entradas
        btn.config(text=f"Entrada X{num_bits-index}: {new_value}"); self.actualizar_salida_uso()

    def actualizar_salida_uso(self):
        if not self.adaline_uso: return
        self.dibujar_red_uso()

    def reconstruir_panel_uso(self):
        for widget in self.frame_uso_controles.winfo_children(): widget.destroy()
        self.botones_entrada_uso.clear()
        
        num_bits = self.adaline_uso.num_entradas
        ttk.Label(self.frame_uso_controles, text=f"Modelo de {num_bits} bits").pack(pady=10)
        
        for i in range(num_bits):
            var = tk.IntVar(value=0)
            
            # --- CORRECCIÓN 1: Se cambia la etiqueta del botón ---
            # Ahora se usa i+1 para que el orden sea X1, X2, X3...
            btn_text = f"Entrada X{i+1}: 0"
            
            btn = ttk.Button(self.frame_uso_controles, text=btn_text, 
                            command=lambda v=var, b=None, i=i: self.toggle_input_uso(v, b, i))
            btn.var_ref = var
            btn.config(command=lambda v=var, b=btn, i=i: self.toggle_input_uso(v, b, i))
            btn.pack(pady=5, fill="x")
            self.botones_entrada_uso.append(btn)
            
        self.actualizar_salida_uso()

    def dibujar_red_uso(self):
        self.canvas_red.delete("all"); self.update_idletasks()
        w, h = self.canvas_red.winfo_width(), self.canvas_red.winfo_height()
        if w < 10 or h < 10 or not self.adaline_uso: return

        num_entradas = self.adaline_uso.num_entradas
        x_in, x_out, y_out = w * 0.2, w * 0.8, h * 0.5
        
        self.canvas_red.create_oval(x_out-30, y_out-30, x_out+30, y_out+30, fill="lightblue", width=2)
        
        # --- CORRECCIÓN 2: Se invierte la lista de entradas para el modelo ---
        # Los botones están en orden [X1, X2, X3], pero el modelo espera [MSB, ..., LSB] ([X3, X2, X1])
        entradas_gui = [btn.var_ref.get() for btn in self.botones_entrada_uso]
        entradas_modelo = list(reversed(entradas_gui))
        prediccion = self.adaline_uso.predecir(entradas_modelo)
        
        self.canvas_red.create_text(x_out, y_out, text=f"{prediccion:.2f}", font=("Arial", 14, "bold"))
        
        y_step = h / (num_entradas + 1)
        for i in range(num_entradas):
            y_in = y_step * (i + 1)
            
            # Se usa el valor de la GUI sin invertir para la visualización
            valor_entrada_actual = entradas_gui[i]
            
            # Se corrige la etiqueta de la neurona y el peso
            nombre_bit = f"X{i+1}"
            # El peso w1 corresponde a la entrada X1 (LSB), que es el último en la lista de pesos del modelo
            nombre_peso = f"w{i+1}"
            indice_peso_modelo = num_entradas - i

            self.canvas_red.create_oval(x_in-15, y_in-15, x_in+15, y_in+15, fill="lightgreen")
            self.canvas_red.create_text(x_in, y_in, text=f"{nombre_bit}={valor_entrada_actual}", font=("Arial", 10))
            self.canvas_red.create_line(x_in, y_in, x_out, y_out, fill="gray")
            self.canvas_red.create_text((x_in + x_out)/2, (y_in + y_out)/2 + 10, 
                                        text=f"{nombre_peso}={self.adaline_uso.pesos[indice_peso_modelo]:.2f}", 
                                        fill="darkblue")
        
        self.canvas_red.create_text(x_out, y_out - (h * 0.4), text=f"Bias w0 = {self.adaline_uso.pesos[0]:.2f}", font=("Arial",10))

        
    def on_tab_change(self, event):
        if self.notebook.tab(self.notebook.select(), "text") == "Uso de la Red":
            self.cargar_modelo_para_uso()

if __name__ == "__main__":
    app = App()
    app.mainloop()