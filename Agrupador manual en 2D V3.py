"""
Explorador Dimensional 2D — Backoffice
PCA / t-SNE  +  dibujo de regiones poligonales
Pizarra: coordenadas 0..100 (1 decimal), margen de 5 unidades en cada borde
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv, math, os, random

try:
    import numpy as np
except ImportError:
    raise SystemExit("Falta numpy.   Ejecuta:  pip install numpy")
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise SystemExit("Falta scikit-learn.   Ejecuta:  pip install scikit-learn")

# ══════════════════════════════════════════════════════════════════════════════
#  Constantes de layout
# ══════════════════════════════════════════════════════════════════════════════
CANVAS_SIZE  = 620
MARGIN_PX    = 45
USABLE_PX    = CANVAS_SIZE - 2 * MARGIN_PX

COORD_MIN    = 0.0
COORD_MAX    = 100.0
DRAW_MARGIN  = 5.0
PLOT_LO      = COORD_MIN + DRAW_MARGIN
PLOT_HI      = COORD_MAX - DRAW_MARGIN

PT_R         = 4
SNAP_PX      = 10

BG_ROOT   = "#0d0d14"
BG_PANEL  = "#12121e"
BG_CANVAS = "#07070f"
GRID_COL  = "#1a1a30"
AXIS_COL  = "#2a2a55"
BORDER_COL= "#1e3a5f"
TEXT_LT   = "#dde3f0"
TEXT_DIM  = "#4a5068"
ACCENT    = "#7c3aed"
ACCENT2   = "#06b6d4"
OK_COL    = "#10b981"
ERR_COL   = "#ef4444"
ACCENT3   = "#f59e0b"
PT_NONE   = "#2d3a52"

FIG_COLORS = [
    "#7c3aed","#06b6d4","#f59e0b","#10b981",
    "#ef4444","#ec4899","#84cc16","#f97316",
    "#6366f1","#14b8a6","#eab308","#8b5cf6",
]

# ══════════════════════════════════════════════════════════════════════════════
#  Conversiones
# ══════════════════════════════════════════════════════════════════════════════
def logi_to_px(val):
    return MARGIN_PX + (val - COORD_MIN) / (COORD_MAX - COORD_MIN) * USABLE_PX

def px_to_logi(px):
    raw = COORD_MIN + (px - MARGIN_PX) / USABLE_PX * (COORD_MAX - COORD_MIN)
    return round(max(COORD_MIN, min(COORD_MAX, raw)), 1)

def data_logi(v, lo, hi):
    if hi == lo:
        return (PLOT_LO + PLOT_HI) / 2
    return PLOT_LO + (v - lo) / (hi - lo) * (PLOT_HI - PLOT_LO)

def pt_to_cv(lx, ly):
    cx = logi_to_px(lx)
    cy = CANVAS_SIZE - logi_to_px(ly)
    return cx, cy

def cv_to_pt(cx, cy):
    lx = px_to_logi(cx)
    ly = px_to_logi(CANVAS_SIZE - cy)
    return lx, ly

def point_in_polygon(px, py, poly):
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

# ══════════════════════════════════════════════════════════════════════════════
#  App
# ══════════════════════════════════════════════════════════════════════════════
class ExplorerApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Explorador Dimensional 2D")
        self.resizable(False, False)
        self.configure(bg=BG_ROOT)

        self.raw_df = []
        self.headers = []
        self.num_headers = []
        self.source_file = None

        self.coords2d = []
        self.explained_var = None
        self.method = tk.StringVar(value="PCA")
        self.tsne_perp = tk.IntVar(value=30)

        self.figures = []
        self.fig_counter = 0
        self.drawing = False
        self.current_poly = []
        self.point_fig = []

        self._build_ui()
        self._draw_empty_board()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        hdr = tk.Frame(self, bg=BG_ROOT)
        hdr.pack(fill="x", padx=14, pady=(12,4))
        tk.Label(hdr, text="◈  EXPLORADOR DIMENSIONAL  2D",
                 font=("Courier New", 13, "bold"),
                 bg=BG_ROOT, fg=ACCENT).pack(side="left")
        self.status_lbl = tk.Label(hdr, text="Sin datos",
                                   font=("Courier New", 8),
                                   bg=BG_ROOT, fg=TEXT_DIM)
        self.status_lbl.pack(side="right")

        body = tk.Frame(self, bg=BG_ROOT)
        body.pack(padx=14, pady=4)

        # canvas column
        cv_col = tk.Frame(body, bg=BG_ROOT)
        cv_col.pack(side="left", anchor="n", padx=(0,12))

        wrap = tk.Frame(cv_col, bg=ACCENT2, bd=1)
        wrap.pack()
        self.canvas = tk.Canvas(wrap,
                                width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg=BG_CANVAS, highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack()
        self.canvas.bind("<Button-1>",        self._on_click)
        self.canvas.bind("<Motion>",          self._on_motion)
        self.canvas.bind("<Double-Button-1>", self._on_dbl)

        self.coord_lbl = tk.Label(cv_col, text="",
                                  font=("Courier New", 8),
                                  bg=BG_ROOT, fg=TEXT_DIM, anchor="w")
        self.coord_lbl.pack(fill="x", pady=(3,0))

        # sidebar
        self.sb = tk.Frame(body, bg=BG_ROOT, width=285)
        self.sb.pack(side="left", fill="y")
        self.sb.pack_propagate(False)
        self._build_sidebar()

    def _build_sidebar(self):
        # Archivo
        self._sec("ARCHIVO")
        r = tk.Frame(self.sb, bg=BG_ROOT)
        r.pack(fill="x", pady=4)
        self._btn(r, "⇪ Cargar CSV", self._load_csv, ACCENT2).pack(
            side="left", fill="x", expand=True, padx=(0,3))
        self._btn(r, "💾 Exportar",  self._export_csv, OK_COL).pack(
            side="left", fill="x", expand=True)

        # Proyección
        self._sec("PROYECCIÓN")
        mr = tk.Frame(self.sb, bg=BG_ROOT)
        mr.pack(fill="x", pady=4)
        self.btn_pca  = self._mode_btn(mr, "PCA",   lambda: self._set_method("PCA"),  True)
        self.btn_tsne = self._mode_btn(mr, "t-SNE", lambda: self._set_method("TSNE"), False)
        self.btn_pca.pack(side="left",  fill="x", expand=True, padx=(0,3))
        self.btn_tsne.pack(side="left", fill="x", expand=True)

        pr = tk.Frame(self.sb, bg=BG_ROOT)
        pr.pack(fill="x", pady=2)
        tk.Label(pr, text="Perplejidad t-SNE:", font=("Courier New", 8),
                 bg=BG_ROOT, fg=TEXT_DIM, anchor="w").pack(side="left")
        tk.Spinbox(pr, textvariable=self.tsne_perp,
                   from_=5, to=100, increment=5, width=5,
                   font=("Courier New", 9), bg=BG_PANEL, fg=ACCENT2,
                   relief="flat", bd=0, buttonbackground=BG_PANEL,
                   insertbackground=ACCENT2).pack(side="left", padx=6)

        self._btn(self.sb, "▶  Calcular proyección",
                  self._compute_projection, ACCENT).pack(fill="x", pady=6)

        # Varianza
        self._sec("VARIANZA EXPLICADA  (PCA)")
        vf = tk.Frame(self.sb, bg=BG_PANEL)
        vf.pack(fill="x", pady=2)
        self.var_lbl = tk.Label(vf, text="—",
                                font=("Courier New", 9),
                                bg=BG_PANEL, fg=TEXT_DIM,
                                padx=8, pady=6, justify="left")
        self.var_lbl.pack(fill="x")

        # Figuras
        self._sec("FIGURAS / REGIONES")
        fr = tk.Frame(self.sb, bg=BG_ROOT)
        fr.pack(fill="x", pady=4)
        self.btn_draw   = self._mode_btn(fr, "✏  Dibujar figura", self._start_draw,  False)
        self.btn_cancel = self._mode_btn(fr, "✕  Cancelar",       self._cancel_draw, False)
        self.btn_draw.pack(side="left",   fill="x", expand=True, padx=(0,3))
        self.btn_cancel.pack(side="left", fill="x", expand=True)

        tk.Label(self.sb,
                 text="• Clic: añadir vértice\n"
                      "• Clic sobre ⊙ (1er vértice): cerrar figura\n"
                      "• Doble-clic: cerrar automáticamente",
                 font=("Courier New", 7), bg=BG_ROOT, fg=TEXT_DIM,
                 justify="left").pack(fill="x", pady=(0,6))

        lf = tk.Frame(self.sb, bg=BG_PANEL)
        lf.pack(fill="both", expand=True, pady=4)
        sb2 = ttk.Scrollbar(lf)
        sb2.pack(side="right", fill="y")
        self.fig_lb = tk.Listbox(lf, yscrollcommand=sb2.set,
                                 bg=BG_PANEL, fg=TEXT_LT,
                                 selectbackground=ACCENT,
                                 font=("Courier New", 9),
                                 bd=0, highlightthickness=0, activestyle="none")
        self.fig_lb.pack(fill="both", expand=True)
        sb2.config(command=self.fig_lb.yview)

        self._btn(self.sb, "🗑  Eliminar figura seleccionada",
                  self._delete_fig, ERR_COL).pack(fill="x", pady=(2,0))

        self._btn(self.sb, "📊  Boxplot por figura",
                  self._show_boxplot, ACCENT3).pack(fill="x", pady=(4,0))

        self.fig_info = tk.Label(self.sb, text="",
                                 font=("Courier New", 8),
                                 bg=BG_ROOT, fg=TEXT_DIM, justify="left")
        self.fig_info.pack(fill="x", pady=(4,0))

    def _sec(self, title):
        f = tk.Frame(self.sb, bg=BG_ROOT)
        f.pack(fill="x", pady=(10,2))
        tk.Label(f, text=title, font=("Courier New", 8, "bold"),
                 bg=BG_ROOT, fg=ACCENT2).pack(side="left")
        tk.Frame(f, height=1, bg=ACCENT2).pack(side="left", fill="x",
                                                expand=True, padx=(6,0), pady=4)

    def _btn(self, parent, text, cmd, color=ACCENT2):
        b = tk.Button(parent, text=text, command=cmd,
                      font=("Courier New", 9, "bold"),
                      bg=BG_PANEL, fg=color,
                      activebackground=color, activeforeground=BG_ROOT,
                      relief="flat", bd=0, pady=6, cursor="hand2")
        b.bind("<Enter>", lambda e: b.configure(bg=color, fg=BG_ROOT))
        b.bind("<Leave>", lambda e: b.configure(bg=BG_PANEL, fg=color))
        return b

    def _mode_btn(self, parent, text, cmd, active=False):
        return tk.Button(parent, text=text, command=cmd,
                         font=("Courier New", 8, "bold"),
                         bg=ACCENT2 if active else BG_PANEL,
                         fg=BG_ROOT  if active else TEXT_DIM,
                         relief="flat", bd=0, pady=5, cursor="hand2")

    def _set_method(self, m):
        self.method.set(m)
        self.btn_pca.configure( bg=ACCENT2 if m=="PCA"  else BG_PANEL,
                                fg=BG_ROOT  if m=="PCA"  else TEXT_DIM)
        self.btn_tsne.configure(bg=ACCENT2 if m=="TSNE" else BG_PANEL,
                                fg=BG_ROOT  if m=="TSNE" else TEXT_DIM)

    # ── pizarra vacía ─────────────────────────────────────────────────────────
    def _draw_empty_board(self):
        self.canvas.delete("all")
        self._draw_grid()
        self.canvas.create_text(CANVAS_SIZE//2, CANVAS_SIZE//2 - 20,
                                text="Carga un CSV y calcula\nla proyección para ver los puntos",
                                font=("Courier New", 11), fill=TEXT_DIM,
                                justify="center")

    # ── grilla ────────────────────────────────────────────────────────────────
    def _draw_grid(self):
        lo_px  = logi_to_px(COORD_MIN)
        hi_px  = logi_to_px(COORD_MAX)
        plo_px = logi_to_px(PLOT_LO)
        phi_px = logi_to_px(PLOT_HI)

        # borde exterior
        self.canvas.create_rectangle(
            lo_px, CANVAS_SIZE - hi_px,
            hi_px, CANVAS_SIZE - lo_px,
            outline=BORDER_COL, fill="", width=1)

        # zona activa (5..95)
        self.canvas.create_rectangle(
            plo_px, CANVAS_SIZE - phi_px,
            phi_px, CANVAS_SIZE - plo_px,
            outline=AXIS_COL, fill="#09091a", width=1)

        # líneas de grilla cada 10
        for v in range(0, 101, 10):
            cpx = logi_to_px(v)
            cpy = CANVAS_SIZE - logi_to_px(v)
            self.canvas.create_line(cpx, CANVAS_SIZE - hi_px,
                                    cpx, CANVAS_SIZE - lo_px,
                                    fill=GRID_COL, width=1)
            self.canvas.create_line(lo_px, cpy, hi_px, cpy,
                                    fill=GRID_COL, width=1)
            self.canvas.create_text(cpx, CANVAS_SIZE - lo_px + 12,
                                    text=str(v), fill=TEXT_DIM,
                                    font=("Courier New", 7))
            self.canvas.create_text(lo_px - 16, cpy,
                                    text=str(v), fill=TEXT_DIM,
                                    font=("Courier New", 7))

        # ejes
        o  = logi_to_px(0)
        e  = logi_to_px(100)
        oy = CANVAS_SIZE - o
        ey = CANVAS_SIZE - e
        self.canvas.create_line(o, oy, e, oy, fill=AXIS_COL, width=2)
        self.canvas.create_line(o, oy, o, ey, fill=AXIS_COL, width=2)
        self.canvas.create_text(e + 10, oy, text="X",
                                fill=ACCENT2, font=("Courier New", 8, "bold"))
        self.canvas.create_text(o, ey - 10, text="Y",
                                fill=ACCENT2, font=("Courier New", 8, "bold"))

        # líneas de margen punteadas (5 y 95)
        for v in (PLOT_LO, PLOT_HI):
            cpx = logi_to_px(v)
            cpy = CANVAS_SIZE - logi_to_px(v)
            self.canvas.create_line(cpx, CANVAS_SIZE - hi_px,
                                    cpx, CANVAS_SIZE - lo_px,
                                    fill="#1e3a5f", width=1, dash=(4,4))
            self.canvas.create_line(lo_px, cpy, hi_px, cpy,
                                    fill="#1e3a5f", width=1, dash=(4,4))

    # ── redibujado completo ───────────────────────────────────────────────────
    def _redraw(self):
        self.canvas.delete("all")
        self._draw_grid()
        self._draw_figures()
        self._draw_points()
        self._draw_current_poly()

    def _draw_points(self):
        if not self.coords2d:
            return
        if len(self.point_fig) != len(self.coords2d):
            self.point_fig = [-1] * len(self.coords2d)
        for i, (lx, ly) in enumerate(self.coords2d):
            cx, cy = pt_to_cv(lx, ly)
            fi = self.point_fig[i]
            if fi >= 0:
                color   = FIG_COLORS[fi % len(FIG_COLORS)]
                outline = "white"
                r       = PT_R + 1
            else:
                color   = PT_NONE
                outline = "#3d4f6a"
                r       = PT_R
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                    fill=color, outline=outline, width=0.8)

    def _draw_figures(self):
        for fi, fig in enumerate(self.figures):
            poly = fig["poly"]
            if len(poly) < 2:
                continue
            color   = FIG_COLORS[fi % len(FIG_COLORS)]
            pts_px  = [pt_to_cv(lx, ly) for lx, ly in poly]

            if fig["closed"] and len(pts_px) >= 3:
                flat = [c for p in pts_px for c in p]
                self.canvas.create_polygon(*flat,
                                           fill=color, stipple="gray12",
                                           outline="")
            n = len(pts_px)
            for k in range(n):
                nxt = (k+1) % n
                if not fig["closed"] and nxt == 0:
                    break
                self.canvas.create_line(*pts_px[k], *pts_px[nxt],
                                        fill=color, width=2)
            for cx, cy in pts_px:
                self.canvas.create_oval(cx-4, cy-4, cx+4, cy+4,
                                        fill=color, outline="white", width=1)
            if pts_px:
                mx = sum(p[0] for p in pts_px) / len(pts_px)
                my = sum(p[1] for p in pts_px) / len(pts_px)
                self.canvas.create_text(mx, my, text=f"F{fig['id']}",
                                        font=("Courier New", 10, "bold"),
                                        fill="white")

    def _draw_current_poly(self):
        if not self.current_poly:
            return
        color  = FIG_COLORS[self.fig_counter % len(FIG_COLORS)]
        pts_px = [pt_to_cv(lx, ly) for lx, ly in self.current_poly]
        for k in range(len(pts_px)-1):
            self.canvas.create_line(*pts_px[k], *pts_px[k+1],
                                    fill=color, width=2, dash=(6,3))
        for k, (cx, cy) in enumerate(pts_px):
            r = 7 if k == 0 else 4
            self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                                    fill=color, outline="white", width=1.5)
        if pts_px:
            cx, cy = pts_px[0]
            self.canvas.create_text(cx, cy-14, text="⊙ inicio",
                                    fill=color, font=("Courier New", 7))

    # ── eventos canvas ────────────────────────────────────────────────────────
    def _on_click(self, event):
        lx, ly = cv_to_pt(event.x, event.y)
        self.coord_lbl.config(text=f"  clic  x={lx:.1f}   y={ly:.1f}")
        if not self.drawing:
            return
        # cerrar si cerca del primer vértice
        if len(self.current_poly) >= 3:
            fx, fy = pt_to_cv(*self.current_poly[0])
            if math.hypot(event.x - fx, event.y - fy) <= SNAP_PX:
                self._close_figure()
                return
        self.current_poly.append((lx, ly))
        self._redraw()

    def _on_dbl(self, event):
        if self.drawing and len(self.current_poly) >= 3:
            self._close_figure()

    def _on_motion(self, event):
        lx, ly = cv_to_pt(event.x, event.y)
        self.coord_lbl.config(text=f"  cursor  x={lx:.1f}   y={ly:.1f}")
        if not self.drawing or not self.current_poly:
            return
        self._redraw()
        color = FIG_COLORS[self.fig_counter % len(FIG_COLORS)]
        last  = pt_to_cv(*self.current_poly[-1])
        self.canvas.create_line(*last, event.x, event.y,
                                fill=color, width=1.5, dash=(4,4))
        if len(self.current_poly) >= 3:
            fx, fy = pt_to_cv(*self.current_poly[0])
            if math.hypot(event.x - fx, event.y - fy) <= SNAP_PX:
                self.canvas.create_oval(fx-SNAP_PX, fy-SNAP_PX,
                                        fx+SNAP_PX, fy+SNAP_PX,
                                        outline="white", width=2)

    # ── figuras ───────────────────────────────────────────────────────────────
    def _start_draw(self):
        if not self.coords2d:
            messagebox.showwarning("Sin datos", "Primero carga y proyecta un CSV.")
            return
        self.drawing = True
        self.current_poly = []
        self.canvas.config(cursor="crosshair")
        self.btn_draw.configure(bg=ACCENT, fg="white")
        self.btn_cancel.configure(bg=ERR_COL, fg="white")
        self.status_lbl.config(text="✏  Modo dibujo activo — haz clic en la pizarra")

    def _cancel_draw(self):
        self.drawing = False
        self.current_poly = []
        self.canvas.config(cursor="crosshair")
        self.btn_draw.configure(bg=BG_PANEL, fg=TEXT_DIM)
        self.btn_cancel.configure(bg=BG_PANEL, fg=TEXT_DIM)
        self.status_lbl.config(text="Dibujo cancelado")
        self._redraw()

    def _close_figure(self):
        if len(self.current_poly) < 3:
            messagebox.showinfo("Figura inválida",
                                "Se necesitan al menos 3 vértices.")
            return
        fig = {"id":     self.fig_counter + 1,
               "poly":   list(self.current_poly),
               "closed": True}
        self.figures.append(fig)
        self.fig_counter += 1
        self.current_poly = []
        self.drawing = False
        self.canvas.config(cursor="crosshair")
        self.btn_draw.configure(bg=BG_PANEL, fg=TEXT_DIM)
        self.btn_cancel.configure(bg=BG_PANEL, fg=TEXT_DIM)
        self._assign_points()
        self._refresh_fig_list()
        self._redraw()
        self.status_lbl.config(text=f"Figura F{fig['id']} creada — {sum(1 for f in self.point_fig if f==len(self.figures)-1)} puntos dentro")

    def _assign_points(self):
        self.point_fig = [-1] * len(self.coords2d)
        for i, (lx, ly) in enumerate(self.coords2d):
            for fi, fig in enumerate(self.figures):
                if fig["closed"] and point_in_polygon(lx, ly, fig["poly"]):
                    self.point_fig[i] = fi
                    break

    def _refresh_fig_list(self):
        self.fig_lb.delete(0, tk.END)
        counts = {}
        for fi in self.point_fig:
            counts[fi] = counts.get(fi, 0) + 1
        for fi, fig in enumerate(self.figures):
            n = counts.get(fi, 0)
            self.fig_lb.insert(tk.END,
                               f"  F{fig['id']}  ▎  {n} punto{'s' if n!=1 else ''}")
        none_n = counts.get(-1, len(self.coords2d) if self.coords2d else 0)
        self.fig_info.config(
            text=f"  {len(self.figures)} figura(s)  |  {none_n} pts sin región")

    def _delete_fig(self):
        sel = self.fig_lb.curselection()
        if not sel:
            messagebox.showinfo("Info", "Selecciona una figura de la lista.")
            return
        idx = sel[0]
        if 0 <= idx < len(self.figures):
            fid = self.figures[idx]["id"]
            self.figures.pop(idx)
            self._assign_points()
            self._refresh_fig_list()
            self._redraw()
            self.status_lbl.config(text=f"Figura F{fid} eliminada")

    # ── carga CSV ─────────────────────────────────────────────────────────────
    def _load_csv(self):
        path = filedialog.askopenfilename(
            title="Abrir CSV",
            filetypes=[("CSV","*.csv"),("Todos","*.*")])
        if not path:
            return
        try:
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                raise ValueError("El archivo está vacío.")
            self.headers = list(rows[0].keys())
            self.raw_df  = rows
            self.source_file = path

            self.num_headers = []
            for col in self.headers:
                try:
                    [float(r[col]) for r in rows]
                    self.num_headers.append(col)
                except (ValueError, KeyError):
                    pass

            n_num = len(self.num_headers)
            if n_num < 2:
                messagebox.showerror("Error",
                    f"Se necesitan >= 2 columnas numéricas. Encontradas: {n_num}.")
                return

            self.coords2d = []
            self.figures  = []
            self.fig_counter = 0
            self.current_poly = []
            self.drawing  = False
            self.point_fig = []
            self.explained_var = None
            self._refresh_fig_list()
            self._update_var_label()

            fname = os.path.basename(path)

            if n_num == 2:
                xs = [float(r[self.num_headers[0]]) for r in rows]
                ys = [float(r[self.num_headers[1]]) for r in rows]
                lo_x, hi_x = min(xs), max(xs)
                lo_y, hi_y = min(ys), max(ys)
                self.coords2d = [
                    (round(data_logi(xs[i], lo_x, hi_x), 1),
                     round(data_logi(ys[i], lo_y, hi_y), 1))
                    for i in range(len(rows))]
                self.point_fig = [-1] * len(self.coords2d)
                self._redraw()
                self.status_lbl.config(
                    text=f"{fname}  —  {len(rows)} filas  (2D directo)")
            else:
                self._draw_empty_board()
                self.status_lbl.config(
                    text=f"{fname}  —  {len(rows)} filas  |  {n_num} atrib. num.  →  pulsa ▶ Calcular")
        except Exception as ex:
            messagebox.showerror("Error al cargar", str(ex))

    # ── proyección ────────────────────────────────────────────────────────────
    def _compute_projection(self):
        if not self.raw_df:
            messagebox.showwarning("Sin datos", "Primero carga un CSV.")
            return
        if len(self.num_headers) < 2:
            messagebox.showerror("Error", "Se necesitan >= 2 columnas numéricas.")
            return

        self.status_lbl.config(text="Calculando…")
        self.update_idletasks()

        try:
            X  = np.array([[float(r[c]) for c in self.num_headers]
                           for r in self.raw_df])
            Xs = StandardScaler().fit_transform(X)

            method = self.method.get()
            if method == "PCA":
                model = PCA(n_components=2, random_state=42)
                Xr    = model.fit_transform(Xs)
                ev    = model.explained_variance_ratio_
                self.explained_var = (float(ev[0]), float(ev[1]))
            else:
                perp  = min(self.tsne_perp.get(), max(5, len(self.raw_df)-1))
                # n_iter fue renombrado a max_iter en scikit-learn >= 1.2
                import sklearn
                _skv = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
                _iter_kw = "max_iter" if _skv >= (1, 2) else "n_iter"
                model = TSNE(n_components=2, perplexity=perp,
                             random_state=42, **{_iter_kw: 1000})
                Xr    = model.fit_transform(Xs)
                self.explained_var = None

            raw_x = Xr[:, 0].tolist()
            raw_y = Xr[:, 1].tolist()
            lo_x, hi_x = min(raw_x), max(raw_x)
            lo_y, hi_y = min(raw_y), max(raw_y)

            self.coords2d = [
                (round(data_logi(raw_x[i], lo_x, hi_x), 1),
                 round(data_logi(raw_y[i], lo_y, hi_y), 1))
                for i in range(len(self.raw_df))]

            self.figures      = []
            self.fig_counter  = 0
            self.current_poly = []
            self.drawing      = False
            self.point_fig    = [-1] * len(self.coords2d)
            self._refresh_fig_list()
            self._update_var_label()
            self._redraw()

            fname = os.path.basename(self.source_file) if self.source_file else ""
            self.status_lbl.config(
                text=f"{fname}  —  {method}  |  {len(self.coords2d)} puntos proyectados")

        except Exception as ex:
            messagebox.showerror("Error en proyección", str(ex))
            self.status_lbl.config(text="Error en la proyección")

    def _update_var_label(self):
        if self.explained_var is None:
            m = self.method.get()
            txt = ("t-SNE no provee varianza explicada\n(preservación estructura local)"
                   if m == "TSNE" else "—  Calcula la proyección primero")
            self.var_lbl.config(text=txt, fg=TEXT_DIM)
        else:
            v1, v2 = self.explained_var
            tot = v1 + v2
            b1  = "█" * max(1, int(v1 * 25))
            b2  = "█" * max(1, int(v2 * 25))
            self.var_lbl.config(
                text=f"  PC1 : {v1*100:5.1f}%  {b1}\n"
                     f"  PC2 : {v2*100:5.1f}%  {b2}\n"
                     f"  Total: {tot*100:.1f}%",
                fg=OK_COL)

    # ── exportar ──────────────────────────────────────────────────────────────
    def _export_csv(self):
        if not self.raw_df:
            messagebox.showwarning("Sin datos", "Carga un CSV primero.")
            return
        if not self.coords2d:
            messagebox.showwarning("Sin proyección",
                                   "Calcula la proyección antes de exportar.")
            return

        path = filedialog.asksaveasfilename(
            title="Guardar CSV enriquecido",
            defaultextension=".csv",
            filetypes=[("CSV","*.csv"),("Todos","*.*")],
            initialfile="datos_con_figuras.csv")
        if not path:
            return

        try:
            method = self.method.get()
            c1, c2 = f"{method}_1", f"{method}_2"
            new_cols   = [c1, c2, "figura"]
            fieldnames = self.headers + [c for c in new_cols if c not in self.headers]

            pf = (self.point_fig if len(self.point_fig)==len(self.raw_df)
                  else [-1]*len(self.raw_df))

            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for i, row in enumerate(self.raw_df):
                    nr = dict(row)
                    nr[c1] = f"{self.coords2d[i][0]:.1f}"
                    nr[c2] = f"{self.coords2d[i][1]:.1f}"
                    fi = pf[i]
                    nr["figura"] = str(self.figures[fi]["id"]) if fi >= 0 else ""
                    w.writerow(nr)

            n_asig = sum(1 for f in pf if f >= 0)
            messagebox.showinfo("Exportado",
                                f"Guardado: {os.path.basename(path)}\n"
                                f"{len(self.raw_df)} filas  |  "
                                f"{n_asig} puntos con figura asignada")
            self.status_lbl.config(text=f"✔ Exportado: {os.path.basename(path)}")
        except Exception as ex:
            messagebox.showerror("Error al exportar", str(ex))

    # ── boxplot ───────────────────────────────────────────────────────────────
    def _show_boxplot(self):
        if not self.figures:
            messagebox.showwarning("Sin figuras", "Dibuja al menos una figura primero.")
            return
        if not self.raw_df:
            messagebox.showwarning("Sin datos", "Carga un CSV primero.")
            return
        if len(self.point_fig) != len(self.raw_df):
            messagebox.showwarning("Sin proyección", "Calcula la proyección primero.")
            return

        assigned = [i for i, fi in enumerate(self.point_fig) if fi >= 0]
        if not assigned:
            messagebox.showwarning("Sin puntos asignados",
                                   "Ningún punto está dentro de alguna figura.")
            return

        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            from matplotlib.figure import Figure as MplFigure
        except ImportError:
            messagebox.showerror(
                "Falta matplotlib",
                "Instala matplotlib:\n\npip install matplotlib")
            return
        method = self.method.get()
        excluir = {f"{method}_1", f"{method}_2", "figura"}
        cols = [c for c in self.num_headers if c not in excluir]
        if not cols:
            messagebox.showwarning("Sin atributos", "No hay atributos numéricos para graficar.")
            return

        # Construir datos por figura
        fig_data = {}
        for i, fi in enumerate(self.point_fig):
            if fi < 0:
                continue
            if fi not in fig_data:
                fig_data[fi] = {c: [] for c in cols}
            row = self.raw_df[i]
            for c in cols:
                try:
                    fig_data[fi][c].append(float(row[c]))
                except (ValueError, KeyError):
                    pass

        figs_con_datos = sorted(fig_data.keys())
        if not figs_con_datos:
            messagebox.showwarning("Sin datos", "No hay puntos asignados a figuras.")
            return

        n_cols = len(cols)

        # Ventana Toplevel
        win = tk.Toplevel(self)
        win.title("Boxplot por figura")
        win.configure(bg=BG_ROOT)
        win.resizable(True, True)

        # Barra de selección de figura
        top_bar = tk.Frame(win, bg=BG_ROOT)
        top_bar.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(top_bar, text="Figura: ",
                 font=("Courier New", 9, "bold"),
                 bg=BG_ROOT, fg=ACCENT2).pack(side="left")

        btn_frame = tk.Frame(top_bar, bg=BG_ROOT)
        btn_frame.pack(side="left", padx=4)

        info_lbl = tk.Label(top_bar, text="",
                            font=("Courier New", 8),
                            bg=BG_ROOT, fg=TEXT_DIM)
        info_lbl.pack(side="right", padx=8)

        # Canvas matplotlib
        fig_w = max(8, min(n_cols * 1.6, 22))
        mpl_fig = MplFigure(figsize=(fig_w, 5.5), facecolor="#0d0d14")
        ax = mpl_fig.add_subplot(111)
        mpl_fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.22)

        canvas_frame = tk.Frame(win, bg=BG_ROOT)
        canvas_frame.pack(fill="both", expand=True, padx=8, pady=(0, 0))
        mpl_canvas = FigureCanvasTkAgg(mpl_fig, master=canvas_frame)
        mpl_canvas.get_tk_widget().pack(fill="both", expand=True)

        toolbar_frame = tk.Frame(win, bg="#1a1a2e")
        toolbar_frame.pack(fill="x")
        toolbar = NavigationToolbar2Tk(mpl_canvas, toolbar_frame)
        toolbar.config(bg="#1a1a2e")
        toolbar.update()

        btn_refs = {}

        def _render(fi_idx):
            ax.clear()
            ax.set_facecolor("#09091a")
            mpl_fig.patch.set_facecolor("#0d0d14")

            data_cols = fig_data[fi_idx]
            color_hex = FIG_COLORS[fi_idx % len(FIG_COLORS)]
            fid       = self.figures[fi_idx]["id"]
            n_pts     = len(next(iter(data_cols.values()), []))

            def hex_rgba(h, a=1.0):
                return (int(h[1:3],16)/255, int(h[3:5],16)/255,
                        int(h[5:7],16)/255, a)

            plot_data = [data_cols[c] for c in cols]
            positions = list(range(1, n_cols + 1))

            bp = ax.boxplot(plot_data, positions=positions,
                            patch_artist=True, widths=0.55,
                            showfliers=True)

            box_rgba = hex_rgba(color_hex, 0.25)
            lin_rgba = hex_rgba(color_hex, 1.0)

            for patch in bp["boxes"]:
                patch.set_facecolor(box_rgba)
                patch.set_edgecolor(lin_rgba)
                patch.set_linewidth(1.8)
            for w in bp["whiskers"]:
                w.set_color(lin_rgba); w.set_linewidth(1.3); w.set_linestyle("--")
            for cap in bp["caps"]:
                cap.set_color(lin_rgba); cap.set_linewidth(1.5)
            for med in bp["medians"]:
                med.set_color("#ffffff"); med.set_linewidth(2.2)
            for flier in bp["fliers"]:
                flier.set(marker="o", markerfacecolor=color_hex,
                          markeredgecolor="none", markersize=4, alpha=0.6)

            ax.set_xticks(positions)
            ax.set_xticklabels(cols, rotation=38, ha="right",
                               fontsize=8.5, color="#aab0c8",
                               fontfamily="monospace")
            ax.tick_params(axis="y", colors="#aab0c8", labelsize=8)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["bottom", "left"]:
                ax.spines[spine].set_color("#2a2a50")
            ax.yaxis.grid(True, color="#1a1a30", linewidth=0.8, linestyle="--")
            ax.set_axisbelow(True)
            ax.set_xlim(0.3, n_cols + 0.7)
            ax.set_title(f"Figura  F{fid}  —  {n_pts} punto{'s' if n_pts!=1 else ''}",
                         color=color_hex, fontsize=12, fontweight="bold",
                         fontfamily="monospace", pad=10)
            ax.set_ylabel("Valor", color="#aab0c8", fontsize=9,
                          fontfamily="monospace")

            info_lbl.config(text=f"{n_pts} puntos  |  {n_cols} atributos")

            # Actualizar estado de botones
            for k, b in btn_refs.items():
                c = FIG_COLORS[k % len(FIG_COLORS)]
                if k == fi_idx:
                    b.configure(bg=c, fg=BG_ROOT)
                else:
                    b.configure(bg=BG_PANEL, fg=TEXT_DIM)

            mpl_canvas.draw()

        # Crear botones de figura
        for fi in figs_con_datos:
            color = FIG_COLORS[fi % len(FIG_COLORS)]
            fid   = self.figures[fi]["id"]
            b = tk.Button(btn_frame,
                          text=f"  F{fid}  ",
                          font=("Courier New", 9, "bold"),
                          bg=BG_PANEL, fg=TEXT_DIM,
                          relief="flat", bd=0, padx=6, pady=4,
                          cursor="hand2",
                          command=lambda f=fi: _render(f))
            b.pack(side="left", padx=2)
            btn_refs[fi] = b

        _render(figs_con_datos[0])

        def _on_close():
            mpl_fig.clf()
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_close)



if __name__ == "__main__":
    app = ExplorerApp()
    app.mainloop()
# (boxplot appended below - placeholder removed by patch)
