# interfaz.py (Versión 5.3 - Corrección de Cancelar y Gráfica de Pesos)
# Este archivo crea y gestiona la interfaz gráfica (GUI) para el Perceptrón.
# Se comunica con la lógica definida en 'perceptron.py' a través de una instancia
# de la clase Perceptron y un sistema de callbacks para las actualizaciones en tiempo real.

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import random
import sys

# Se importa la clase Perceptron y el diccionario de datos desde el archivo de lógica.
from perceptron import Perceptron, COMPUERTAS_LOGICAS

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class TextRedirector:
    """
    Clase auxiliar para redirigir la salida de la consola (stdout) a un widget Text de tkinter.
    Esto permite que todos los 'print()' del backend aparezcan en la interfaz.
    """
    def __init__(self, widget): self.widget = widget
    def write(self, text_string):
        # Inserta el texto en el widget y se asegura de que la vista se desplace hasta el final.
        self.widget.insert(tk.END, text_string); self.widget.see(tk.END); self.widget.update_idletasks()
    def flush(self): pass

class App(tk.Tk):
    """Clase principal que define la aplicación y su comportamiento."""
    def __init__(self):
        super().__init__()
        self.title("Perceptrón Interactivo - IIPA 2025"); self.geometry("1400x800")

        # Intercepta el evento de cierre de la ventana para asegurar que el programa termine correctamente.
        self.protocol("WM_DELETE_WINDOW", self.cerrar_aplicacion)
        
        self.crear_cabecera()
        self.notebook = ttk.Notebook(self)
        self.tab_entrenamiento = ttk.Frame(self.notebook); self.tab_uso = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_entrenamiento, text="Entrenamiento"); self.notebook.add(self.tab_uso, text="Uso de la Red")
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)
        
        # --- Variables de estado clave para el funcionamiento del Perceptrón ---
        self.entrenamiento_cancelado = False      # Bandera para detener el entrenamiento.
        self.historial_pesos = []                 # Almacena la evolución de los pesos para graficar.
        self.historial_errores_patron = []    # Almacena la evolución de errores por patrón.
        self.perceptron_uso = Perceptron(pesos_iniciales=[0, 0, 0]) # Instancia persistente para la pestaña "Uso".

        self.crear_tab_entrenamiento()
        self.crear_tab_uso()
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)
        self.after(100, self.cargar_pesos_para_uso)

    def cerrar_aplicacion(self):
        """Maneja el cierre de la ventana para terminar el script de forma limpia."""
        self.destroy(); self.quit()

    def actualizar_en_tiempo_real(self, epoca, pesos, errores_patron):
        """
        Esta es la función CALLBACK. Es el corazón de la interactividad en tiempo real.
        El método 'entrenar' del Perceptrón llama a esta función en cada época.

        Args:
            epoca (int): El número de la época actual.
            pesos (list): Una COPIA de los pesos actuales de la red.
            errores_patron (list): Una lista con el error calculado para cada patrón.

        Returns:
            bool: True para continuar el entrenamiento, False para detenerlo.
        """
        # Se guardan los datos recibidos del Perceptrón en los historiales para graficarlos.
        self.historial_pesos.append(pesos)
        self.historial_errores_patron.append(errores_patron)
        eje_x_epocas = range(len(self.historial_pesos))
        
        # Actualiza la Gráfica 1: Error por Patrón, añadiendo una leyenda para claridad.
        self.ax1.clear(); self.ax1.grid(True); self.ax1.set_title(f"Error por Patrón (Época {epoca})")
        self.ax1.set_xlabel("Época") 
        self.ax1.set_ylabel("Error (-1, 0, 1)") 
        errores_t = list(zip(*self.historial_errores_patron))
        patrones = COMPUERTAS_LOGICAS[self.compuerta_var.get()]["X"]
        for i, hist_patron in enumerate(errores_t): self.ax1.plot(hist_patron, label=f'Patrón {i+1}: {patrones[i]}')
        self.ax1.legend()
        
        # Actualiza la Gráfica 2: Evolución de los pesos de forma eficiente usando set_data.
        pesos_t = list(zip(*self.historial_pesos))
        self.linea_w0.set_data(eje_x_epocas, pesos_t[0])
        self.linea_w1.set_data(eje_x_epocas, pesos_t[1])
        self.linea_w2.set_data(eje_x_epocas, pesos_t[2])
        self.ax2.relim(); self.ax2.autoscale_view()
        
        # Actualiza la Gráfica 3: La línea de separación.
        puntos_recta = self.perceptron_actual.obtener_puntos_recta()
        if puntos_recta: (x1, y1), (x2, y2) = puntos_recta; self.linea_separacion.set_data([x1, x2], [y1, y2])
        
        # Dibuja todos los cambios en las gráficas.
        self.canvas.draw()
        
        # Procesa TODOS los eventos pendientes de la GUI (clics, etc.).
        # ESTA LÍNEA ES CRÍTICA para que el botón "Cancelar" sea responsivo.
        self.update()

        # Lógica para la pausa gráfica cada 500 épocas.
        if epoca > 0 and epoca % 500 == 0:
            if not messagebox.askyesno("Entrenamiento en Pausa", f"Se han completado {epoca} épocas.\n¿Desea continuar entrenando?"):
                self.entrenamiento_cancelado = True
        
        # Devuelve la señal al Perceptrón para que sepa si debe continuar o no.
        return not self.entrenamiento_cancelado
        
    def crear_cabecera(self):
        frame_cabecera = ttk.Frame(self, height=100); frame_cabecera.pack(fill="x", padx=10, pady=5)
        frame_cabecera.grid_columnconfigure(0, weight=1); frame_cabecera.grid_columnconfigure(1, weight=2); frame_cabecera.grid_columnconfigure(2, weight=1)
        try:
            img_escudo = Image.open("escudo.png").resize((60, 80), Image.Resampling.LANCZOS); self.foto_escudo = ImageTk.PhotoImage(img_escudo)
            ttk.Label(frame_cabecera, image=self.foto_escudo).grid(row=0, column=0, sticky="w")
        except FileNotFoundError: ttk.Label(frame_cabecera, text="[Escudo no encontrado]").grid(row=0, column=0, sticky="w")
        texto_info = ("Inteligencia Artificial 801 - IIPA 2025\nIngeniería de Sistemas y Computación\n"
                      "Yojahn Leon, Oscar Barbosa, Gabriel Martinez")
        ttk.Label(frame_cabecera, text=texto_info, justify="center", font=("Arial", 12)).grid(row=0, column=1)
        try:
            img_logo = Image.open("logo.png").resize((160, 80), Image.Resampling.LANCZOS); self.foto_logo = ImageTk.PhotoImage(img_logo)
            ttk.Label(frame_cabecera, image=self.foto_logo).grid(row=0, column=2, sticky="e")
        except FileNotFoundError: ttk.Label(frame_cabecera, text="[Logo no encontrado]").grid(row=0, column=2, sticky="e")
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
        """Activa la bandera para detener el entrenamiento en la siguiente época."""
        self.entrenamiento_cancelado = True
    def iniciar_entrenamiento(self):
        """
        Orquesta todo el proceso de entrenamiento. Se llama al presionar el botón 'Iniciar'.
        """
        # Gestiona el estado de los botones para evitar múltiples ejecuciones.
        self.btn_iniciar.config(state="disabled"); self.btn_cancelar.config(state="normal"); self.entrenamiento_cancelado = False
        
        # Redirige la salida de la consola al widget de texto.
        original_stdout = sys.stdout; sys.stdout = TextRedirector(self.log_texto)
        
        # El bloque try...finally asegura que la consola se restaure incluso si hay errores.
        try:
            # Recolecta todos los parámetros de configuración de la GUI.
            tasa = float(self.tasa_aprendizaje_var.get()); compuerta = self.compuerta_var.get()
            pesos_iniciales = None
            if self.bias_mode_var.get() == "manual" or self.pesos_mode_var.get() == "manual":
                w0 = float(self.bias_entry_var.get()) if self.bias_mode_var.get() == "manual" else random.uniform(-1, 1)
                w1 = float(self.w1_entry_var.get()) if self.pesos_mode_var.get() == "manual" else random.uniform(-1, 1)
                w2 = float(self.w2_entry_var.get()) if self.pesos_mode_var.get() == "manual" else random.uniform(-1, 1)
                pesos_iniciales = [w0, w1, w2]
            
            # Reinicia el estado de la interfaz para un nuevo entrenamiento.
            self.log_texto.delete("1.0", tk.END); self.limpiar_graficas(); self.historial_pesos = []; self.historial_errores_patron = []
            
            # Crea la instancia del Perceptrón con la configuración seleccionada.
            self.perceptron_actual = Perceptron(tasa_aprendizaje=tasa, pesos_iniciales=pesos_iniciales)
            
            # Dibuja el estado inicial de la gráfica del plano de decisión.
            self.preparar_grafica_puntos(compuerta)
            
            # Llama al método de entrenamiento, pasándole la función de callback.
            pesos_finales = self.perceptron_actual.entrenar(compuerta=compuerta, callback=self.actualizar_en_tiempo_real)
            
            print(f"\nEntrenamiento Finalizado.\nPesos: {[round(w, 4) for w in pesos_finales]}")
            self.perceptron_actual.guardar_pesos()
            if not self.entrenamiento_cancelado: messagebox.showinfo("Éxito", "Entrenamiento finalizado y pesos guardados.")
        except ValueError: messagebox.showerror("Error de Entrada", "Por favor, revise los valores numéricos.")
        except Exception as e: messagebox.showerror("Error Inesperado", f"Ha ocurrido un error: {e}")
        finally:
            # Restaura la consola y el estado de los botones.
            sys.stdout = original_stdout
            self.btn_iniciar.config(state="normal"); self.btn_cancelar.config(state="disabled")
    def preparar_grafica_puntos(self, compuerta):
        """Dibuja los puntos de datos (0s y 1s) en el plano de decisión antes de entrenar."""
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
        """Carga los pesos del archivo y los asigna a la instancia de 'Uso'."""
        pesos_cargados = Perceptron.cargar_pesos(ruta_archivo="pesos.txt")
        if pesos_cargados:
            # Asigna los pesos cargados a la instancia persistente, asegurando que se usen los correctos.
            self.perceptron_uso.pesos = pesos_cargados
        self.dibujar_red(); self.actualizar_salida()
    def on_tab_change(self, event):
        """Se activa al cambiar de pestaña para recargar los pesos en la de 'Uso'."""
        if self.notebook.tab(self.notebook.select(), "text") == "Uso de la Red": self.cargar_pesos_para_uso()
    def dibujar_red(self):
        self.canvas_red.delete("all"); self.update_idletasks()
        w, h = self.canvas_red.winfo_width(), self.canvas_red.winfo_height()
        if w < 10 or h < 10: return
        x_in, y_in1, y_in2 = w * 0.2, h * 0.25, h * 0.75; x_out, y_out = w * 0.8, h * 0.5
        # Integra los botones de entrada como parte del diagrama de la red.
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
        """Calcula y muestra la salida de la red usando la instancia 'perceptron_uso'."""
        if self.perceptron_uso.pesos and hasattr(self, 'output_text_id'):
            # Usa la instancia persistente para asegurar que los pesos son los cargados del archivo.
            prediccion = self.perceptron_uso.predecir([self.entrada_x1.get(), self.entrada_x2.get()])
            self.canvas_red.itemconfig(self.output_text_id, text=str(prediccion))
    def toggle_bias_entry(self): self.bias_entry.config(state="normal" if self.bias_mode_var.get() == "manual" else "disabled")
    def toggle_pesos_entries(self):
        state = "normal" if self.pesos_mode_var.get() == "manual" else "disabled"
        self.w1_entry.config(state=state); self.w2_entry.config(state=state)

if __name__ == "__main__":
    # Inicia la aplicación.
    app = App()
    app.mainloop()