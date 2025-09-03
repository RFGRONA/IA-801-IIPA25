# interfaz.py (Corregido y Mejorado)

import tkinter as tk
from tkinter import ttk, messagebox
import random

from perceptron import Perceptron
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Perceptrón para Compuerta OR")
        self.geometry("1200x700")

        self.notebook = ttk.Notebook(self)
        
        self.tab_entrenamiento = ttk.Frame(self.notebook)
        self.tab_uso = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_entrenamiento, text="Entrenamiento")
        self.notebook.add(self.tab_uso, text="Uso de la Red")
        
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        self.crear_tab_entrenamiento()
        self.crear_tab_uso()

        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)
        self.after(100, self.cargar_pesos_para_uso)

    def crear_tab_entrenamiento(self):
        frame_izquierdo = ttk.Frame(self.tab_entrenamiento)
        frame_izquierdo.pack(side="left", fill="y", padx=10, pady=10)

        frame_config = ttk.LabelFrame(frame_izquierdo, text="Configuración")
        frame_config.pack(fill="x", pady=5)
        
        ttk.Label(frame_config, text="Tasa de Aprendizaje:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.tasa_aprendizaje_var = tk.StringVar(value="0.1")
        ttk.Entry(frame_config, textvariable=self.tasa_aprendizaje_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame_config, text="Máximo de Épocas:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.max_epocas_var = tk.StringVar(value="100")
        ttk.Entry(frame_config, textvariable=self.max_epocas_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        frame_pesos = ttk.LabelFrame(frame_izquierdo, text="Pesos Iniciales")
        frame_pesos.pack(fill="x", pady=10)
        
        self.bias_mode_var = tk.StringVar(value="auto")
        ttk.Label(frame_pesos, text="Bias (w0):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Radiobutton(frame_pesos, text="Automático", variable=self.bias_mode_var, value="auto", command=self.toggle_bias_entry).grid(row=0, column=1)
        ttk.Radiobutton(frame_pesos, text="Manual:", variable=self.bias_mode_var, value="manual", command=self.toggle_bias_entry).grid(row=0, column=2)
        self.bias_entry_var = tk.StringVar(value="0.5")
        self.bias_entry = ttk.Entry(frame_pesos, textvariable=self.bias_entry_var, state="disabled", width=10)
        self.bias_entry.grid(row=0, column=3, padx=5)

        self.pesos_mode_var = tk.StringVar(value="auto")
        ttk.Label(frame_pesos, text="Pesos (w1, w2):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Radiobutton(frame_pesos, text="Automáticos", variable=self.pesos_mode_var, value="auto", command=self.toggle_pesos_entries).grid(row=1, column=1)
        ttk.Radiobutton(frame_pesos, text="Manuales:", variable=self.pesos_mode_var, value="manual", command=self.toggle_pesos_entries).grid(row=1, column=2)
        self.w1_entry_var = tk.StringVar(value="0.3")
        self.w2_entry_var = tk.StringVar(value="-0.8")
        self.w1_entry = ttk.Entry(frame_pesos, textvariable=self.w1_entry_var, state="disabled", width=10)
        self.w2_entry = ttk.Entry(frame_pesos, textvariable=self.w2_entry_var, state="disabled", width=10)
        self.w1_entry.grid(row=1, column=3, padx=5)
        self.w2_entry.grid(row=2, column=3, padx=5)
        ttk.Label(frame_pesos, text="w1").grid(row=1, column=4)
        ttk.Label(frame_pesos, text="w2").grid(row=2, column=4)

        ttk.Button(frame_izquierdo, text="Iniciar Entrenamiento", command=self.iniciar_entrenamiento).pack(pady=20, fill="x")

        self.log_texto = tk.Text(frame_izquierdo, height=10)
        self.log_texto.pack(fill="both", expand=True)

        frame_graficas = ttk.Frame(self.tab_entrenamiento)
        frame_graficas.pack(side="left", expand=True, fill="both")
        
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, figsize=(8, 6), tight_layout=True) 
        self.canvas = FigureCanvasTkAgg(self.fig, master=frame_graficas)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.limpiar_graficas()

    def limpiar_graficas(self):
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.clear()
            ax.grid(True)
        self.ax1.set_title("Evolución del Error")
        self.ax1.set_xlabel("Época")
        self.ax1.set_ylabel("N° de Errores")
        
        self.ax2.set_title("Evolución de los Pesos")
        self.ax2.set_xlabel("Época")
        self.ax2.set_ylabel("Valor del Peso")

        self.ax3.set_title("Salida por Patrón")
        self.ax3.set_xlabel("Época")
        self.ax3.set_ylabel("Salida (0 o 1)")
        self.canvas.draw()
        
    def toggle_bias_entry(self):
        self.bias_entry.config(state="normal" if self.bias_mode_var.get() == "manual" else "disabled")

    def toggle_pesos_entries(self):
        state = "normal" if self.pesos_mode_var.get() == "manual" else "disabled"
        self.w1_entry.config(state=state)
        self.w2_entry.config(state=state)
        
    def iniciar_entrenamiento(self):
        try:
            tasa = float(self.tasa_aprendizaje_var.get())
            max_epocas = int(self.max_epocas_var.get())

            pesos_iniciales = None
            if self.bias_mode_var.get() == "manual" or self.pesos_mode_var.get() == "manual":
                w0 = float(self.bias_entry_var.get()) if self.bias_mode_var.get() == "manual" else random.uniform(-1, 1)
                w1 = float(self.w1_entry_var.get()) if self.pesos_mode_var.get() == "manual" else random.uniform(-1, 1)
                w2 = float(self.w2_entry_var.get()) if self.pesos_mode_var.get() == "manual" else random.uniform(-1, 1)
                pesos_iniciales = [w0, w1, w2]
            
            self.log_texto.delete("1.0", tk.END)
            self.limpiar_graficas()
            
            p = Perceptron(tasa_aprendizaje=tasa, pesos_iniciales=pesos_iniciales)
            
            self.log_texto.insert(tk.END, f"Pesos Iniciales: {[round(w, 4) for w in p.pesos]}\n---\n")
            self.update_idletasks()
            
            X_or = [[0, 0], [0, 1], [1, 0], [1, 1]]
            Y_or = [0, 1, 1, 1]
            
            resultados = p.entrenar(X_or, Y_or, max_epocas)
            pesos_finales, err_hist, pesos_hist, out_hist = resultados
            
            resumen = f"Entrenamiento Finalizado.\nÉpocas: {len(err_hist)}\nPesos: {[round(w, 4) for w in pesos_finales]}"
            self.log_texto.insert(tk.END, resumen)

            self.actualizar_graficas(err_hist, pesos_hist, out_hist)
            p.guardar_pesos()
            messagebox.showinfo("Éxito", "Entrenamiento finalizado y pesos guardados.")

        except ValueError:
            messagebox.showerror("Error de Entrada", "Por favor, revise los siguientes campos:\n\n- Tasa de Aprendizaje\n- Máximo de Épocas\n- Valores manuales de pesos/bias (si están activados)\n\nDeben contener valores numéricos válidos.")
        except Exception as e:
            messagebox.showerror("Error Inesperado", f"Ha ocurrido un error: {e}")

    def actualizar_graficas(self, err_hist, pesos_hist, out_hist):
        self.limpiar_graficas()
        
        eje_x = range(len(err_hist))
        
        # Gráfica 1: Error
        self.ax1.plot(eje_x, err_hist, marker='.', linestyle='-')
        
        # Gráfica 2: Pesos
        if pesos_hist:
            pesos_t = list(zip(*pesos_hist))
            self.ax2.plot(eje_x, pesos_t[0], label='w0 (Bias)')
            self.ax2.plot(eje_x, pesos_t[1], label='w1')
            self.ax2.plot(eje_x, pesos_t[2], label='w2')
            self.ax2.legend()

        # Gráfica 3: Salidas
        if out_hist:
            salidas_t = list(zip(*out_hist))
            patrones = ['[0,0]', '[0,1]', '[1,0]', '[1,1]']
            for i, s_patron in enumerate(salidas_t):
                self.ax3.plot(eje_x, s_patron, marker='.', linestyle='--', label=f'Salida para {patrones[i]}')
            self.ax3.legend()
            self.ax3.set_yticks([0, 1])

        self.canvas.draw()
        
    def crear_tab_uso(self):
        # Layout con panel izquierdo para controles y derecho para visualización
        frame_controles = ttk.Frame(self.tab_uso, width=200)
        frame_controles.pack(side="left", fill="y", padx=10, pady=10)
        self.canvas_red = tk.Canvas(self.tab_uso, bg="white")
        self.canvas_red.pack(side="left", expand=True, fill="both", padx=10, pady=10)

        ttk.Label(frame_controles, text="Entradas", font=("Arial", 14)).pack(pady=10)
        self.entrada_x1 = tk.IntVar(value=0)
        self.entrada_x2 = tk.IntVar(value=0)
        
        self.btn_x1 = ttk.Button(frame_controles, text="Entrada X1: 0", command=lambda: self.toggle_input(self.entrada_x1, self.btn_x1, "X1"))
        self.btn_x1.pack(pady=10, fill="x")
        self.btn_x2 = ttk.Button(frame_controles, text="Entrada X2: 0", command=lambda: self.toggle_input(self.entrada_x2, self.btn_x2, "X2"))
        self.btn_x2.pack(pady=10, fill="x")

    def cargar_pesos_para_uso(self):
        self.pesos_uso = Perceptron.cargar_pesos()
        self.dibujar_red() # Siempre dibujar la estructura
        self.actualizar_salida()

    def on_tab_change(self, event):
        if self.notebook.tab(self.notebook.select(), "text") == "Uso de la Red":
            self.cargar_pesos_para_uso()

    def dibujar_red(self):
        self.canvas_red.delete("all")
        self.update_idletasks() # Asegurar que el canvas tenga dimensiones
        
        w, h = self.canvas_red.winfo_width(), self.canvas_red.winfo_height()
        if w < 10 or h < 10: return # Evitar dibujar si el canvas es muy pequeño
        
        # Coordenadas relativas
        x_in, y_in1, y_in2 = w * 0.2, h * 0.25, h * 0.75
        x_out, y_out = w * 0.8, h * 0.5
        
        if self.pesos_uso:
            w0, w1, w2 = [round(p, 2) for p in self.pesos_uso]
            
            # Conexiones y pesos
            self.canvas_red.create_line(x_in, y_in1, x_out, y_out, width=2, fill="gray")
            self.canvas_red.create_text((x_in + x_out)/2, (y_in1 + y_out)/2 - 10, text=f"w1 = {w1}")
            self.canvas_red.create_line(x_in, y_in2, x_out, y_out, width=2, fill="gray")
            self.canvas_red.create_text((x_in + x_out)/2, (y_in2 + y_out)/2 + 10, text=f"w2 = {w2}")
            self.canvas_red.create_text(x_out, y_out - h*0.25, text=f"Bias (w0) = {w0}")

            # Neurona de salida
            self.canvas_red.create_oval(x_out-30, y_out-30, x_out+30, y_out+30, fill="lightblue", width=2)
            self.output_text_id = self.canvas_red.create_text(x_out, y_out, text="?", font=("Arial", 16, "bold"))
        else:
            self.canvas_red.create_text(w/2, h/2, text="Entrene la red para ver la visualización.", font=("Arial", 12))

    def toggle_input(self, input_var, btn, name):
        new_value = 1 - input_var.get()
        input_var.set(new_value)
        btn.config(text=f"Entrada {name}: {new_value}")
        self.actualizar_salida()

    def actualizar_salida(self):
        if self.pesos_uso and hasattr(self, 'output_text_id'):
            p = Perceptron(pesos_iniciales=self.pesos_uso)
            prediccion = p.predecir([self.entrada_x1.get(), self.entrada_x2.get()])
            self.canvas_red.itemconfig(self.output_text_id, text=str(prediccion))

if __name__ == "__main__":
    app = App()
    app.mainloop()