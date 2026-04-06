import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import customtkinter as ctk
import pandas as pd
import numpy as np
import time
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function

# Nuevas librerías para métricas y Holt-Winters
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- Funciones matemáticas personalizadas ---
def _logexp(x):
    with np.errstate(over='ignore', invalid='ignore'):
        return np.where(x > 100, x, np.log(1 + np.exp(x)))

logexp = make_function(function=_logexp, name='logexp', arity=1)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class TimeSeriesApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Backoffice Evolutivo Multivariable - Series de Tiempo")
        self.geometry("1300x950")
        
        # Estado
        self.df = None
        self.model = None
        self.target_col = None
        self.extra_cols = []
        self.X_full = None
        self.y_full = None
        self.l_val = 12
        self.p_val = 3
        self.std_error = 0 
        self.feature_names = []

        self._setup_ui()

    def _setup_ui(self):
        self.nav_frame = ctk.CTkFrame(self, height=60)
        self.nav_frame.pack(side="top", fill="x", padx=10, pady=5)

        ctk.CTkButton(self.nav_frame, text="1. Cargar CSV", command=self.load_file).pack(side="left", padx=10)
        ctk.CTkButton(self.nav_frame, text="2. Configurar Evolución", command=self.open_gp_settings).pack(side="left", padx=10)
        ctk.CTkButton(self.nav_frame, text="3. Resultados y Proyección", command=self.show_results).pack(side="left", padx=10)
        
        # --- NUEVO BOTÓN: COMPARACIÓN ---
        ctk.CTkButton(self.nav_frame, text="4. Comparativa Holt-Winters", fg_color="#2c3e50", hover_color="#34495e", command=self.open_comparison_window).pack(side="left", padx=10)

        self.main_container = ctk.CTkFrame(self)
        self.main_container.pack(expand=True, fill="both", padx=10, pady=10)

        self.fig = Figure(figsize=(10, 5), dpi=100, facecolor='#2b2b2b')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#2b2b2b')
        self.ax.tick_params(colors='white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_container)
        self.canvas.get_tk_widget().pack(expand=True, fill="both")

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            try:
                self.df = pd.read_csv(path).dropna()
                if not np.issubdtype(self.df.iloc[:, 0].dtype, np.number):
                    self.df.iloc[:, 0] = pd.to_datetime(self.df.iloc[:, 0])
                    self.df.set_index(self.df.columns[0], inplace=True)
                
                self.ax.clear()
                for col in self.df.columns:
                    self.ax.plot(self.df.index, self.df[col], label=col)
                self.ax.legend(fontsize='x-small')
                self.ax.set_title("Series de Tiempo Cargadas", color="white")
                self.canvas.draw()
                messagebox.showinfo("Éxito", f"Datos cargados: {len(self.df.columns)} columnas detectadas.")
            except Exception as e: messagebox.showerror("Error", str(e))

    def open_gp_settings(self):
        if self.df is None: return messagebox.showwarning("Atención", "Cargue datos primero.")
        
        self.gp_win = ctk.CTkToplevel(self)
        self.gp_win.title("Parámetros Evolutivos Multivariables"); self.gp_win.geometry("500x750")
        self.gp_win.attributes("-topmost", True)

        ctk.CTkLabel(self.gp_win, text="Atributo Objetivo (Y):").pack(pady=5)
        self.target_selector = ctk.CTkComboBox(self.gp_win, values=list(self.df.columns))
        self.target_selector.pack()

        fields = [
            ("Periodos Test (Validación)", "3"), 
            ("Letargos Máximos (Lags Y)", "12"), 
            ("Población", "1000"), 
            ("Generaciones", "20"),
            ("Semilla (0 = Aleatoria)", "0")
        ]
        self.entries = {}
        for text, default in fields:
            ctk.CTkLabel(self.gp_win, text=text).pack()
            e = ctk.CTkEntry(self.gp_win); e.insert(0, default); e.pack(); self.entries[text] = e

        self.txt_eq = ctk.CTkTextbox(self.gp_win, height=180); self.txt_eq.pack(pady=10, fill="x", padx=10)
        ctk.CTkButton(self.gp_win, text="Iniciar Proceso Evolutivo", 
                     command=lambda: self.run_gp()).pack(pady=10)

    def run_gp(self):
        try:
            target = self.target_selector.get()
            self.p_val = int(self.entries["Periodos Test (Validación)"].get())
            self.l_val = int(self.entries["Letargos Máximos (Lags Y)"].get())
            seed_input = int(self.entries["Semilla (0 = Aleatoria)"].get())
            current_seed = int(time.time() % 1000000) if seed_input == 0 else seed_input

            self.target_col = target
            self.extra_cols = [c for c in self.df.columns if c != target]
            
            data_y = self.df[target].values
            data_ext = self.df[self.extra_cols].values
            
            X, y = [], []
            for i in range(self.l_val, len(data_y)):
                lags = data_y[i-self.l_val : i][::-1]
                ext_vals = data_ext[i]
                X.append(np.concatenate([lags, ext_vals]))
                y.append(data_y[i])
            
            self.X_full, self.y_full = np.array(X), np.array(y)
            self.feature_names = [f"Lag{j+1}_{target}" for j in range(self.l_val)] + self.extra_cols
            
            X_train, y_train = self.X_full[:-self.p_val], self.y_full[:-self.p_val]

            self.model = SymbolicRegressor(
                population_size=int(self.entries["Población"].get()), 
                generations=int(self.entries["Generaciones"].get()),
                function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'sin', 'cos', logexp],
                metric='rmse', random_state=current_seed, verbose=1,
                feature_names=self.feature_names
            )
            self.model.fit(X_train, y_train)
            
            self.std_error = np.std(y_train - self.model.predict(X_train))
            
            self.txt_eq.delete("1.0", tk.END)
            self.txt_eq.insert("1.0", f"SEMILLA: {current_seed}\nVARIABLES USADAS: {self.feature_names}\n\nECUACIÓN:\n{self.model._program}")
            messagebox.showinfo("Éxito", "Modelo entrenado con atributos adicionales.")
        except Exception as e: messagebox.showerror("Error", str(e))

    def show_results(self):
        if self.model is None: return
        res_win = ctk.CTkToplevel(self); res_win.title("Análisis y Proyección Multivariable"); res_win.geometry("1200x900")
        res_win.attributes("-topmost", True)

        ctrl = ctk.CTkFrame(res_win); ctrl.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(ctrl, text="Proyectar a futuro:").pack(side="left", padx=5)
        ent_fut = ctk.CTkEntry(ctrl, width=60); ent_fut.insert(0, "5"); ent_fut.pack(side="left")
        ctk.CTkLabel(ctrl, text="Confianza Z:").pack(side="left", padx=5)
        ent_z = ctk.CTkEntry(ctrl, width=60); ent_z.insert(0, "1.96"); ent_z.pack(side="left")
        
        ctk.CTkButton(ctrl, text="Calcular", 
                     command=lambda: self.update_results_ui(res_win, int(ent_fut.get()), float(ent_z.get()))).pack(side="left", padx=20)

    def update_results_ui(self, window, future_steps, z_score):
        try:
            train_pred = self.model.predict(self.X_full[:-self.p_val])
            test_pred = self.model.predict(self.X_full[-self.p_val:])
            
            last_x = self.X_full[-1].copy()
            future_preds = []
            
            for _ in range(future_steps):
                p = self.model.predict(np.nan_to_num(last_x.reshape(1, -1)))[0]
                val = p if np.isfinite(p) else last_x[0]
                future_preds.append(val)
                new_lags = np.roll(last_x[:self.l_val], 1)
                new_lags[0] = val
                last_x[:self.l_val] = new_lags
            
            if hasattr(self, 'tree_frame'): self.tree_frame.destroy()
            self.tree_frame = ctk.CTkFrame(window)
            self.tree_frame.pack(fill="x", padx=10)
            
            cols = ("N°", "Fase", "Real", "Modelo", "Residuo")
            tree = ttk.Treeview(self.tree_frame, columns=cols, show='headings', height=7)
            for c in cols: tree.heading(c, text=c); tree.column(c, width=110)
            
            y_train_real = self.y_full[:-self.p_val]
            y_test_real = self.y_full[-self.p_val:]
            
            for i in range(len(train_pred)):
                tree.insert("", "end", values=(i+1, "Entreno", f"{y_train_real[i]:.2f}", f"{train_pred[i]:.2f}", f"{y_train_real[i]-train_pred[i]:.2f}"))
            for i in range(len(test_pred)):
                idx = len(train_pred)+i+1
                tree.insert("", "end", values=(idx, "Testeo", f"{y_test_real[i]:.2f}", f"{test_pred[i]:.2f}", f"{y_test_real[i]-test_pred[i]:.2f}"))
            for i in range(len(future_preds)):
                idx = len(train_pred)+len(test_pred)+i+1
                tree.insert("", "end", values=(idx, "Proyectado", "---", f"{future_preds[i]:.2f}", "---"))
            
            tree.pack(side="left", fill="x", expand=True)
            sb = ttk.Scrollbar(self.tree_frame, orient="vertical", command=tree.yview); tree.configure(yscroll=sb.set); sb.pack(side="right", fill="y")

            if hasattr(self, 'res_canvas_widget'): self.res_canvas_widget.destroy()
            fig = Figure(figsize=(10, 8), dpi=100)
            ax1 = fig.add_subplot(211); ax2 = fig.add_subplot(212)
            fig.tight_layout(pad=5.0)

            x_train = np.arange(self.l_val, self.l_val + len(train_pred))
            x_test = np.arange(x_train[-1] + 1, x_train[-1] + 1 + len(test_pred))
            x_fut = np.arange(x_test[-1] + 1, x_test[-1] + 1 + future_steps)

            ax1.plot(range(len(self.df)), self.df[self.target_col], color="gray", alpha=0.3, label="Serie Real")
            ax1.plot(x_train, train_pred, label="Ajuste Entrenamiento", color="blue")
            ax1.plot(x_test, test_pred, label="Ajuste Testeo", color="orange", linewidth=2)
            ax1.plot(x_fut, future_preds, label="Proyección Futura", color="red", linestyle="--", marker='o')
            
            ci = z_score * self.std_error
            ax1.fill_between(x_fut, np.array(future_preds)-ci, np.array(future_preds)+ci, color='red', alpha=0.1, label=f"IC ({z_score}σ)")
            ax1.set_title("Comparativa de Serie Multivariable", fontsize=12)
            ax1.legend(loc='upper left', fontsize='x-small')

            ax2.bar(x_train, y_train_real - train_pred, color="blue", alpha=0.5, label="Res. Entrenamiento")
            ax2.bar(x_test, y_test_real - test_pred, color="orange", alpha=0.8, label="Res. Testeo")
            ax2.axhline(0, color="black", lw=1)
            ax2.set_title("Análisis de Residuos", fontsize=12)
            ax2.legend(loc='upper left', fontsize='x-small')

            self.res_canvas = FigureCanvasTkAgg(fig, master=window)
            self.res_canvas_widget = self.res_canvas.get_tk_widget()
            self.res_canvas_widget.pack(fill="both", expand=True, padx=10, pady=10)
            self.res_canvas.draw()
        except Exception as e: messagebox.showerror("Error", str(e))

    # --- NUEVAS FUNCIONES DE COMPARACIÓN HOLT-WINTERS ---

    def open_comparison_window(self):
        if self.model is None or self.df is None: 
            return messagebox.showwarning("Atención", "Cargue datos y entrene el modelo evolutivo primero.")
        
        comp_win = ctk.CTkToplevel(self)
        comp_win.title("Comparativa de Modelos: Evolutivo vs Holt-Winters")
        comp_win.geometry("1100x850")
        comp_win.attributes("-topmost", True)

        ctrl = ctk.CTkFrame(comp_win)
        ctrl.pack(fill="x", padx=20, pady=20)

        ctk.CTkLabel(ctrl, text="Periodos Testeo:").pack(side="left", padx=5)
        ent_test = ctk.CTkEntry(ctrl, width=60); ent_test.insert(0, str(self.p_val)); ent_test.pack(side="left")

        ctk.CTkLabel(ctrl, text="Periodos Proyección:").pack(side="left", padx=5)
        ent_proj = ctk.CTkEntry(ctrl, width=60); ent_proj.insert(0, "10"); ent_proj.pack(side="left")

        ctk.CTkButton(ctrl, text="Ejecutar Comparativa", 
                     command=lambda: self.run_hw_comparison(comp_win, int(ent_test.get()), int(ent_proj.get()))).pack(side="left", padx=20)

        self.comp_results_frame = ctk.CTkFrame(comp_win)
        self.comp_results_frame.pack(fill="both", expand=True, padx=20, pady=10)

    def run_hw_comparison(self, window, n_test, n_proj):
        try:
            for widget in self.comp_results_frame.winfo_children(): widget.destroy()

            y_real = self.df[self.target_col].values
            train_data = y_real[:-n_test]
            test_real = y_real[-n_test:]

            # 1. HOLT-WINTERS
            hw_model = ExponentialSmoothing(train_data, seasonal_periods=self.l_val, trend='add', seasonal='add').fit()
            hw_test_pred = hw_model.forecast(n_test)
            hw_proj_pred = hw_model.forecast(n_test + n_proj)[n_test:]

            # 2. MODELO EVOLUTIVO
            evol_test_pred = self.model.predict(self.X_full[-n_test:])
            
            last_x = self.X_full[-1].copy()
            evol_proj_pred = []
            for _ in range(n_proj):
                p = self.model.predict(last_x.reshape(1, -1))[0]
                evol_proj_pred.append(p)
                new_lags = np.roll(last_x[:self.l_val], 1)
                new_lags[0] = p
                last_x[:self.l_val] = new_lags

            # MÉTRICAS
            r2_evol = r2_score(test_real, evol_test_pred)
            r2_hw = r2_score(test_real, hw_test_pred)
            rmse_evol = np.sqrt(mean_squared_error(test_real, evol_test_pred))
            rmse_hw = np.sqrt(mean_squared_error(test_real, hw_test_pred))

            # UI: TABLA E INDICADORES
            metrics_frame = ctk.CTkFrame(self.comp_results_frame)
            metrics_frame.pack(fill="x", pady=10)

            title_lbl = ctk.CTkLabel(metrics_frame, text="INDICADORES DE DESEMPEÑO (FASE TESTEO)", font=("Arial", 14, "bold"))
            title_lbl.grid(row=0, column=0, columnspan=3, pady=10)

            # Headers
            ctk.CTkLabel(metrics_frame, text="Modelo", font=("Arial", 12, "bold")).grid(row=1, column=0, padx=20)
            ctk.CTkLabel(metrics_frame, text="R² (Precisión)", font=("Arial", 12, "bold")).grid(row=1, column=1, padx=20)
            ctk.CTkLabel(metrics_frame, text="Ruido (RMSE)", font=("Arial", 12, "bold")).grid(row=1, column=2, padx=20)

            # Data
            ctk.CTkLabel(metrics_frame, text="Evolutivo").grid(row=2, column=0)
            ctk.CTkLabel(metrics_frame, text=f"{r2_evol:.4f}").grid(row=2, column=1)
            ctk.CTkLabel(metrics_frame, text=f"{rmse_evol:.4f}").grid(row=2, column=2)

            ctk.CTkLabel(metrics_frame, text="Holt-Winters").grid(row=3, column=0)
            ctk.CTkLabel(metrics_frame, text=f"{r2_hw:.4f}").grid(row=3, column=1)
            ctk.CTkLabel(metrics_frame, text=f"{rmse_hw:.4f}").grid(row=3, column=2)

            # GRÁFICO
            fig = Figure(figsize=(10, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            idx_real = np.arange(len(y_real))
            idx_test = np.arange(len(y_real) - n_test, len(y_real))
            idx_proj = np.arange(len(y_real), len(y_real) + n_proj)

            ax.plot(idx_real, y_real, label="Real Histórico", color="gray", alpha=0.4)
            ax.plot(idx_test, test_real, color="black", linewidth=1.5, label="Real Test")
            
            ax.plot(idx_test, evol_test_pred, label="Evolutivo (Test)", color="#3498db", linewidth=2)
            ax.plot(idx_proj, evol_proj_pred, "--", label="Evolutivo (Proy)", color="#3498db")
            
            ax.plot(idx_test, hw_test_pred, label="Holt-Winters (Test)", color="#2ecc71", linewidth=2)
            ax.plot(idx_proj, hw_proj_pred, "--", label="Holt-Winters (Proy)", color="#2ecc71")

            ax.set_title(f"Comparativa: {n_test} periodos test / {n_proj} proyección")
            ax.legend(loc="upper left", fontsize="small")
            
            canvas = FigureCanvasTkAgg(fig, master=self.comp_results_frame)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            canvas.draw()

        except Exception as e:
            messagebox.showerror("Error", f"Fallo en comparativa: {str(e)}")

if __name__ == "__main__":
    app = TimeSeriesApp()
    app.mainloop()