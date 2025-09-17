# interfaz.py (Versión final con indentación corregida)

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import random
import sys
import os

# Se importa la clase Perceptron y el diccionario de datos desde el archivo de lógica.
from perceptron import Perceptron, COMPUERTAS_LOGICAS

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

def resource_path(relative_path):
    """ Obtiene la ruta absoluta al recurso, funciona para desarrollo y para PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class TextRedirector:
    """Clase auxiliar para redirigir la salida de la consola (stdout) a un widget Text de tkinter."""
    def __init__(self, widget): self.widget = widget
    def write(self, text_string):
        self.widget.insert(tk.END, text_string); self.widget.see(tk.END); self.widget.update_idletasks()
    def flush(self): pass

class App(tk.Tk):
    """Clase principal que define la aplicación y su comportamiento."""
    def __init__(self):
        super().__init__()
        
        self.title("Perceptrón Interactivo - IIPA 2025"); self.geometry("1400x800")
        self.protocol("WM_DELETE_WINDOW", self.cerrar_aplicacion)
        
        # Todas las funciones de la clase se llaman con 'self.'
        self.crear_cabecera()

        self.notebook = ttk.Notebook(self)
        self.tab_entrenamiento = ttk.Frame(self.notebook); self.tab_uso = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_entrenamiento, text="Entrenamiento"); self.notebook.add(self.tab_uso, text="Uso de la Red")
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        
        self.entrenamiento_cancelado = False
        self.historial_pesos = []
        self.historial_errores_patron = []
        self.perceptron_uso = Perceptron(pesos_iniciales=[0, 0, 0])

        self.crear_tab_entrenamiento()
        self.crear_tab_uso()
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)
        self.after(100, self.cargar_pesos_para_uso)

    # Todas las siguientes funciones deben estar indentadas a este nivel
    
    def cerrar_aplicacion(self):
        """Maneja el cierre de la ventana para terminar el script de forma limpia."""
        self.destroy(); self.quit()

    def crear_cabecera(self):
        frame_cabecera = ttk.Frame(self, height=100)
        frame_cabecera.pack(fill="x", padx=10, pady=5)
        frame_cabecera.grid_columnconfigure(0, weight=1)
        frame_cabecera.grid_columnconfigure(1, weight=2)
        frame_cabecera.grid_columnconfigure(2, weight=1)

        # --- Columna 1: Escudo (con corrección de anclaje de imagen) ---
        label_escudo = ttk.Label(frame_cabecera)
        try:
            img_escudo = Image.open(resource_path("escudo.png")).resize((60, 80), Image.Resampling.LANCZOS)
            foto_escudo = ImageTk.PhotoImage(img_escudo)
            label_escudo.image = foto_escudo
            label_escudo.config(image=foto_escudo)
        except FileNotFoundError:
            label_escudo.config(text="[Escudo no encontrado]")
        label_escudo.grid(row=0, column=0, sticky="w")

        # --- Columna 2: Texto ---
        texto_info = ("Inteligencia Artificial 801 - IIPA 2025\nIngeniería de Sistemas y Computación\n"
                      "Yohan Leon, Oscar Barbosa, Gabriel Martinez")
        ttk.Label(frame_cabecera, text=texto_info, justify="center", font=("Arial", 12)).grid(row=0, column=1)

        # --- Columna 3: Logotipo (con corrección de anclaje de imagen) ---
        label_logo = ttk.Label(frame_cabecera)
        try:
            img_logo = Image.open(resource_path("logo.png")).resize((160, 80), Image.Resampling.LANCZOS)
            foto_logo = ImageTk.PhotoImage(img_logo)
            label_logo.image = foto_logo
            label_logo.config(image=foto_logo)
        except FileNotFoundError:
            label_logo.config(text="[Logo no encontrado]")
        label_logo.grid(row=0, column=2, sticky="e")
        
    def crear_tab_entrenamiento(self):
        frame_izquierdo = ttk.Frame(self.tab_entrenamiento, width=350); frame_izquierdo.pack(side="left", fill="y", padx=10, pady=10); frame_izquierdo.pack_propagate(False)
        frame_graficas = ttk.Frame(self.tab_entrenamiento); frame_graficas.pack(side="left", expand=True, fill="both")
        frame_config = ttk.LabelFrame(frame_izquierdo, text="Configuración"); frame_config.pack(fill="x", pady=5)
        ttk.Label(frame_config, text="Compuerta Lógica:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.compuerta_var = tk.StringVar(value="OR"); compuertas = list(COMPUERTAS_LOGICAS.keys())
        ttk.Combobox(frame_config, textvariable=self.compuerta_var, values=compuertas, state="readonly").grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(frame_config, text="Tasa de Aprendizaje:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.tasa_aprendizaje_var = tk.StringVar(value="0.1"); ttk.Entry(frame_config, textvariable=self.tasa_aprendizaje_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        frame_pesos = ttk.LabelFrame(frame_izquierdo, text="Pesos Iniciales"); frame_pesos.pack(fill="x", pady=10)
        self.bias_mode_var = tk.StringVar(value="auto"); ttk.Label(frame_pesos, text="Bias (w0):").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(frame_pesos, text="Auto", variable=self.bias_mode_var, value="auto", command=self.toggle_bias_entry).grid(row=0, column=1)
        ttk.Radiobutton(frame_pesos, text="Manual:", variable=self.bias_mode_var, value="manual", command=self.toggle_bias_entry).grid(row=0, column=2)
        self.bias_entry_var = tk.StringVar(value="0.5"); self.bias_entry = ttk.Entry(frame_pesos, textvariable=self.bias_entry_var, state="disabled", width=8); self.bias_entry.grid(row=0, column=3, padx=5)
        self.pesos_mode_var = tk.StringVar(value="auto"); ttk.Label(frame_pesos, text="Pesos (w1,w2):").grid(row=1, column=0, sticky="w")
        ttk.Radiobutton(frame_pesos, text="Auto", variable=self.pesos_mode_var, value="auto", command=self.toggle_pesos_entries).grid(row=1, column=1)
        ttk.Radiobutton(frame_pesos, text="Manual:", variable=self.pesos_mode_var, value="manual", command=self.toggle_pesos_entries).grid(row=1, column=2)
        self.w1_entry_var = tk.StringVar(value="0.3"); self.w2_entry_var = tk.StringVar(value="-0.8")
        self.w1_entry = ttk.Entry(frame_pesos, textvariable=self.w1_entry_var, state="disabled", width=8); self.w2_entry = ttk.Entry(frame_pesos, textvariable=self.w2_entry_var, state="disabled", width=8)
        self.w1_entry.grid(row=1, column=3, padx=2); self.w2_entry.grid(row=2, column=3, padx=2); ttk.Label(frame_pesos, text="w1").grid(row=1, column=4); ttk.Label(frame_pesos, text="w2").grid(row=2, column=4)
        self.btn_iniciar = ttk.Button(frame_izquierdo, text="Iniciar Entrenamiento", command=self.iniciar_entrenamiento)
        self.btn_iniciar.pack(pady=10, fill="x")
        self.btn_cancelar = ttk.Button(frame_izquierdo, text="Cancelar Entrenamiento", command=self.cancelar_entrenamiento, state="disabled")
        self.btn_cancelar.pack(pady=5, fill="x")
        self.log_texto = tk.Text(frame_izquierdo, height=10); self.log_texto.pack(fill="both", expand=True)
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(9, 7), tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_graficas); self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.limpiar_graficas()

    def limpiar_graficas(self):
        for ax in [self.ax1, self.ax2, self.ax3]: ax.clear(); ax.grid(True)
        self.ax1.set_title("Error por Patrón"); self.ax1.set_xlabel("Época"); self.ax1.set_ylabel("Error (-1, 0, 1)")
        self.ax2.set_title("Evolución de Pesos"); self.ax2.set_xlabel("Época"); self.ax2.set_ylabel("Valor")
        self.linea_w0, = self.ax2.plot([], [], label='w0 (Bias)'); self.linea_w1, = self.ax2.plot([], [], label='w1'); self.linea_w2, = self.ax2.plot([], [], label='w2'); self.ax2.legend()
        self.ax3.set_title("Plano de Decisión"); self.ax3.set_xlabel("Entrada X1"); self.ax3.set_ylabel("Entrada X2")
        self.ax3.set_xlim(-0.5, 1.5); self.ax3.set_ylim(-0.5, 1.5); self.canvas.draw()
        
    def cancelar_entrenamiento(self):
        self.entrenamiento_cancelado = True

    def iniciar_entrenamiento(self):
        self.btn_iniciar.config(state="disabled"); self.btn_cancelar.config(state="normal"); self.entrenamiento_cancelado = False
        original_stdout = sys.stdout; sys.stdout = TextRedirector(self.log_texto)
        try:
            tasa = float(self.tasa_aprendizaje_var.get()); compuerta = self.compuerta_var.get()
            pesos_iniciales = None
            if self.bias_mode_var.get() == "manual" or self.pesos_mode_var.get() == "manual":
                w0 = float(self.bias_entry_var.get()) if self.bias_mode_var.get() == "manual" else random.uniform(-1, 1)
                w1 = float(self.w1_entry_var.get()) if self.pesos_mode_var.get() == "manual" else random.uniform(-1, 1)
                w2 = float(self.w2_entry_var.get()) if self.pesos_mode_var.get() == "manual" else random.uniform(-1, 1)
                pesos_iniciales = [w0, w1, w2]
            self.log_texto.delete("1.0", tk.END); self.limpiar_graficas(); self.historial_pesos = []; self.historial_errores_patron = []
            self.perceptron_actual = Perceptron(tasa_aprendizaje=tasa, pesos_iniciales=pesos_iniciales)
            self.preparar_grafica_puntos(compuerta)
            pesos_finales = self.perceptron_actual.entrenar(compuerta=compuerta, callback=self.actualizar_en_tiempo_real)
            print(f"\nEntrenamiento Finalizado.\nPesos: {[round(w, 4) for w in pesos_finales]}")
            self.perceptron_actual.guardar_pesos()
            if not self.entrenamiento_cancelado: messagebox.showinfo("Éxito", "Entrenamiento finalizado y pesos guardados.")
        except ValueError: messagebox.showerror("Error de Entrada", "Por favor, revise los valores numéricos.")
        except Exception as e: messagebox.showerror("Error Inesperado", f"Ha ocurrido un error: {e}")
        finally:
            sys.stdout = original_stdout
            self.btn_iniciar.config(state="normal"); self.btn_cancelar.config(state="disabled")

    def actualizar_en_tiempo_real(self, epoca, pesos, errores_patron):
        self.historial_pesos.append(pesos)
        self.historial_errores_patron.append(errores_patron)
        eje_x_epocas = range(len(self.historial_pesos))
        self.ax1.clear(); self.ax1.grid(True); self.ax1.set_title(f"Error por Patrón (Época {epoca})")
        self.ax1.set_xlabel("Época"); self.ax1.set_ylabel("Error (-1, 0, 1)")
        errores_t = list(zip(*self.historial_errores_patron))
        patrones = COMPUERTAS_LOGICAS[self.compuerta_var.get()]["X"]
        for i, hist_patron in enumerate(errores_t): self.ax1.plot(hist_patron, label=f'Patrón {i+1}: {patrones[i]}')
        self.ax1.legend()
        pesos_t = list(zip(*self.historial_pesos))
        self.linea_w0.set_data(eje_x_epocas, pesos_t[0])
        self.linea_w1.set_data(eje_x_epocas, pesos_t[1])
        self.linea_w2.set_data(eje_x_epocas, pesos_t[2])
        self.ax2.relim(); self.ax2.autoscale_view()
        puntos_recta = self.perceptron_actual.obtener_puntos_recta()
        if puntos_recta: (x1, y1), (x2, y2) = puntos_recta; self.linea_separacion.set_data([x1, x2], [y1, y2])
        self.canvas.draw(); self.update()
        if epoca > 0 and epoca % 500 == 0:
            if not messagebox.askyesno("Entrenamiento en Pausa", f"Se han completado {epoca} épocas.\n¿Desea continuar entrenando?"):
                self.entrenamiento_cancelado = True
        return not self.entrenamiento_cancelado
        
    def preparar_grafica_puntos(self, compuerta):
        datos = COMPUERTAS_LOGICAS[compuerta]; X, Y = datos["X"], datos["Y"]
        for i in range(len(X)):
            color = 'blue' if Y[i] == 1 else 'red'
            self.ax3.scatter(X[i][0], X[i][1], c=color, s=100, edgecolors='k'); self.ax3.text(X[i][0] + 0.05, X[i][1], f'({X[i][0]},{X[i][1]}) -> {Y[i]}', fontsize=9)
        self.linea_separacion, = self.ax3.plot([], [], 'g-', linewidth=2, label='Frontera de decisión')
        self.ax3.legend(loc='upper left'); self.canvas.draw()
        
    def crear_tab_uso(self):
        self.canvas_red = tk.Canvas(self.tab_uso, bg="white"); self.canvas_red.pack(expand=True, fill="both")
        self.entrada_x1 = tk.IntVar(value=0); self.entrada_x2 = tk.IntVar(value=0)
        self.btn_x1 = ttk.Button(self, text="Entrada X1: 0", command=lambda: self.toggle_input(self.entrada_x1, self.btn_x1, "X1"))
        self.btn_x2 = ttk.Button(self, text="Entrada X2: 0", command=lambda: self.toggle_input(self.entrada_x2, self.btn_x2, "X2"))
        
    def cargar_pesos_para_uso(self):
        pesos_cargados = Perceptron.cargar_pesos(ruta_archivo="pesos.txt")
        if pesos_cargados: self.perceptron_uso.pesos = pesos_cargados
        self.dibujar_red(); self.actualizar_salida()
        
    def on_tab_change(self, event):
        if self.notebook.tab(self.notebook.select(), "text") == "Uso de la Red": self.cargar_pesos_para_uso()
        
    def dibujar_red(self):
        self.canvas_red.delete("all"); self.update_idletasks()
        w, h = self.canvas_red.winfo_width(), self.canvas_red.winfo_height()
        if w < 10 or h < 10: return
        x_in, y_in1, y_in2 = w * 0.2, h * 0.25, h * 0.75; x_out, y_out = w * 0.8, h * 0.5
        self.canvas_red.create_window(x_in, y_in1, window=self.btn_x1); self.canvas_red.create_window(x_in, y_in2, window=self.btn_x2)
        if self.perceptron_uso.pesos:
            w0, w1, w2 = [round(p, 2) for p in self.perceptron_uso.pesos]
            self.canvas_red.create_line(x_in, y_in1, x_out, y_out, width=2, fill="gray"); self.canvas_red.create_text((x_in + x_out)/2, (y_in1 + y_out)/2 - 10, text=f"w1 = {w1}")
            self.canvas_red.create_line(x_in, y_in2, x_out, y_out, width=2, fill="gray"); self.canvas_red.create_text((x_in + x_out)/2, (y_in2 + y_out)/2 + 10, text=f"w2 = {w2}")
            self.canvas_red.create_text(x_out, y_out - h*0.3, text=f"Bias (w0) = {w0}", font=("Arial",10))
            self.canvas_red.create_oval(x_out-30, y_out-30, x_out+30, y_out+30, fill="lightblue", width=2)
            self.output_text_id = self.canvas_red.create_text(x_out, y_out, text="?", font=("Arial", 16, "bold"))
        else: self.canvas_red.create_text(w/2, h/2, text="Entrene la red para ver la visualización.", font=("Arial", 12))
        
    def toggle_input(self, input_var, btn, name):
        new_value = 1 - input_var.get(); input_var.set(new_value)
        btn.config(text=f"Entrada {name}: {new_value}"); self.actualizar_salida()
        
    def actualizar_salida(self):
        if self.perceptron_uso.pesos and hasattr(self, 'output_text_id'):
            prediccion = self.perceptron_uso.predecir([self.entrada_x1.get(), self.entrada_x2.get()])
            self.canvas_red.itemconfig(self.output_text_id, text=str(prediccion))

    def toggle_bias_entry(self):
        self.bias_entry.config(state="normal" if self.bias_mode_var.get() == "manual" else "disabled")
        
    def toggle_pesos_entries(self):
        state = "normal" if self.pesos_mode_var.get() == "manual" else "disabled"
        self.w1_entry.config(state=state); self.w2_entry.config(state=state)

if __name__ == "__main__":
    app = App()
    app.mainloop()
