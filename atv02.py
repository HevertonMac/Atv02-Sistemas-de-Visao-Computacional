"""
Atividade Prática 02 - Processamento de Imagem: Pirâmides (Blending de Imagens)
Disciplina: Sistemas de Visão Computacional
Autor: Heverton Macedo <heverton.macedo@ufpi.edu.br>
VipLab UFPI

Técnica de Burt & Adelson (1983): blending multiresolução usando pirâmides
Gaussianas e Laplacianas.  As baixas frequências são misturadas gradualmente
pela pirâmide Gaussiana da máscara; as altas frequências (texturas, detalhes)
são preservadas localmente em cada nível Laplaciano, eliminando cortes bruscos.

Implementação manual — sem funções de pirâmide de bibliotecas de CV.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageDraw, ImageFont, ImageTk
import numpy as np
import os

# ─────────────────────────────────────────────
# Constantes de estilo (mesmo padrão do atv00)
# ─────────────────────────────────────────────
BG_DARK   = "#1e1e2e"
BG_PANEL  = "#2a2a3e"
BG_BTN    = "#3d3d5c"
BG_BTN_HL = "#5c5c8a"
FG_TEXT   = "#cdd6f4"
FG_DIM    = "#7c7fa8"
ACCENT    = "#89b4fa"
DANGER    = "#f38ba8"
SUCCESS   = "#a6e3a1"
WARN      = "#f9e2af"
FONT_MAIN = ("Segoe UI", 10)
FONT_H1   = ("Segoe UI", 13, "bold")
FONT_H2   = ("Segoe UI", 10, "bold")
FONT_MONO = ("Consolas", 9)


# ═══════════════════════════════════════════════════════════════════
# Implementação manual das pirâmides (Burt & Adelson 1983)
# ═══════════════════════════════════════════════════════════════════
#
# Kernel Gaussiano separável 1-D: w = [1, 4, 6, 4, 1] / 16
# A convolução 2-D é feita como dois passes 1-D (horizontal + vertical).
# ═══════════════════════════════════════════════════════════════════

_W = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625], dtype=np.float64)


def _gauss_blur(img: np.ndarray) -> np.ndarray:
    """Filtragem Gaussiana 5×5 separável — implementação manual sem scipy/cv2."""
    f = img.astype(np.float64)
    is_2d = f.ndim == 2

    # ── passo horizontal ──────────────────────────────────
    if is_2d:
        ph = np.pad(f, ((0, 0), (2, 2)), mode="reflect")
        hout = (_W[0]*ph[:, 0:ph.shape[1]-4] + _W[1]*ph[:, 1:ph.shape[1]-3] +
                _W[2]*ph[:, 2:ph.shape[1]-2] + _W[3]*ph[:, 3:ph.shape[1]-1] +
                _W[4]*ph[:, 4:ph.shape[1]  ])
    else:
        ph = np.pad(f, ((0, 0), (2, 2), (0, 0)), mode="reflect")
        hout = (_W[0]*ph[:, 0:ph.shape[1]-4, :] + _W[1]*ph[:, 1:ph.shape[1]-3, :] +
                _W[2]*ph[:, 2:ph.shape[1]-2, :] + _W[3]*ph[:, 3:ph.shape[1]-1, :] +
                _W[4]*ph[:, 4:ph.shape[1],   :])

    # ── passo vertical ────────────────────────────────────
    if is_2d:
        pv = np.pad(hout, ((2, 2), (0, 0)), mode="reflect")
        vout = (_W[0]*pv[0:pv.shape[0]-4, :] + _W[1]*pv[1:pv.shape[0]-3, :] +
                _W[2]*pv[2:pv.shape[0]-2, :] + _W[3]*pv[3:pv.shape[0]-1, :] +
                _W[4]*pv[4:pv.shape[0],   :])
    else:
        pv = np.pad(hout, ((2, 2), (0, 0), (0, 0)), mode="reflect")
        vout = (_W[0]*pv[0:pv.shape[0]-4, :, :] + _W[1]*pv[1:pv.shape[0]-3, :, :] +
                _W[2]*pv[2:pv.shape[0]-2, :, :] + _W[3]*pv[3:pv.shape[0]-1, :, :] +
                _W[4]*pv[4:pv.shape[0],   :, :])

    return vout


def _reduce(img: np.ndarray) -> np.ndarray:
    """REDUCE: blur Gaussiano + subamostragem ×2 (pega linhas/colunas pares)."""
    return _gauss_blur(img)[::2, ::2]


def _expand(img: np.ndarray, target_shape: tuple) -> np.ndarray:
    """
    EXPAND: insere zeros nas posições ímpares e aplica blur ×4.
    target_shape = (H, W) da imagem de destino (para corrigir off-by-one).
    """
    H, W = img.shape[:2]
    if img.ndim == 2:
        out = np.zeros((H * 2, W * 2), dtype=np.float64)
        out[::2, ::2] = img.astype(np.float64)
    else:
        out = np.zeros((H * 2, W * 2, img.shape[2]), dtype=np.float64)
        out[::2, ::2] = img.astype(np.float64)

    result = _gauss_blur(out) * 4.0          # ×4 compensa os zeros inseridos
    th, tw = target_shape
    return result[:th, :tw]                  # ajusta dimensões (tamanhos ímpares)


# ──────────────────────────────────────────────────────────────────

def build_gaussian_pyramid(img: np.ndarray, levels: int) -> list:
    """Pirâmide Gaussiana com `levels` níveis (G[0] = original)."""
    pyr = [img.astype(np.float64)]
    for _ in range(levels - 1):
        pyr.append(_reduce(pyr[-1]))
    return pyr


def build_laplacian_pyramid(gpyr: list) -> list:
    """
    Pirâmide Laplaciana a partir da Gaussiana.
    L[i] = G[i] − EXPAND(G[i+1])   para i < N-1
    L[N-1] = G[N-1]                 (topo: imagem de baixa frequência)
    """
    N = len(gpyr)
    lpyr = []
    for i in range(N - 1):
        expanded = _expand(gpyr[i + 1], target_shape=gpyr[i].shape[:2])
        lpyr.append(gpyr[i] - expanded)
    lpyr.append(gpyr[-1].copy())
    return lpyr


def blend_pyramids(la: list, lb: list, gm: list) -> list:
    """
    Combinação nível a nível:
      LS[k] = GM[k] · LA[k]  +  (1 − GM[k]) · LB[k]
    GM[k] ∈ [0,1]: 1 = usa totalmente A, 0 = usa totalmente B.
    """
    return [gm[k] * la[k] + (1.0 - gm[k]) * lb[k] for k in range(len(la))]


def collapse_pyramid(lpyr: list) -> np.ndarray:
    """Reconstrói a imagem combinando a pirâmide Laplaciana de baixo para cima."""
    result = lpyr[-1].copy()
    for i in range(len(lpyr) - 2, -1, -1):
        result = _expand(result, target_shape=lpyr[i].shape[:2]) + lpyr[i]
    return np.clip(result, 0, 255)


def pyramid_blend(img_a: np.ndarray, img_b: np.ndarray,
                  mask: np.ndarray, levels: int) -> np.ndarray:
    """
    Pipeline completo de blending (Burt & Adelson 1983).

    Parâmetros
    ----------
    img_a, img_b : arrays float64, shape H×W×3, valores [0,255]
    mask         : array float64, shape H×W×3, valores [0,1]
                   1 = pixel vem de A; 0 = pixel vem de B
    levels       : número de níveis das pirâmides
    """
    ga = build_gaussian_pyramid(img_a,  levels)
    gb = build_gaussian_pyramid(img_b,  levels)
    gm = build_gaussian_pyramid(mask,   levels)

    la = build_laplacian_pyramid(ga)
    lb = build_laplacian_pyramid(gb)

    ls = blend_pyramids(la, lb, gm)
    return collapse_pyramid(ls).astype(np.uint8)


def direct_cut(img_a: np.ndarray, img_b: np.ndarray,
               mask: np.ndarray) -> np.ndarray:
    """Justaposição direta (corte simples pela máscara binarizada em 0.5)."""
    m = (mask > 0.5).astype(np.float64)
    return np.clip(m * img_a + (1.0 - m) * img_b, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════
# Widget auxiliar: frame rolável (igual ao atv01)
# ═══════════════════════════════════════════════════════════════════

class _ScrollFrame(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self._cv = tk.Canvas(self, bg=BG_PANEL, highlightthickness=0)
        self._sb = ttk.Scrollbar(self, orient="vertical", command=self._cv.yview)
        self.inner = tk.Frame(self._cv, bg=BG_PANEL)
        self._win = self._cv.create_window((0, 0), window=self.inner, anchor="nw")
        self._cv.configure(yscrollcommand=self._sb.set)
        self._cv.pack(side="left", fill="both", expand=True)
        self._sb.pack(side="right", fill="y")
        self.inner.bind("<Configure>",
                        lambda e: self._cv.configure(scrollregion=self._cv.bbox("all")))
        self._cv.bind("<Configure>",
                      lambda e: self._cv.itemconfig(self._win, width=e.width))
        self._cv.bind("<Enter>",
                      lambda e: self._cv.bind_all("<MouseWheel>", self._scroll))
        self._cv.bind("<Leave>",
                      lambda e: self._cv.unbind_all("<MouseWheel>"))

    def _scroll(self, event):
        self._cv.yview_scroll(int(-1 * (event.delta / 120)), "units")


# ═══════════════════════════════════════════════════════════════════
# Aplicação principal
# ═══════════════════════════════════════════════════════════════════

class Atv02App(tk.Tk):
    """Janela principal — Blending com Pirâmides Gaussiana/Laplaciana."""

    _DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        super().__init__()
        self.title("atv02 — Pirâmides & Blending  |  VipLab UFPI")
        self.geometry("1300x740")
        self.minsize(960, 580)
        self.configure(bg=BG_DARK)

        # ── Estado ──────────────────────────────────────────
        self._img_a:   np.ndarray | None = None   # H×W×3 float64 [0,255]
        self._img_b:   np.ndarray | None = None
        self._mask:    np.ndarray | None = None   # H×W×3 float64 [0,1]
        self._blend_result: np.ndarray | None = None  # resultado blending
        self._direct_result: np.ndarray | None = None # justaposição direta
        self._photo:   ImageTk.PhotoImage | None = None

        self._levels_var = tk.IntVar(value=6)
        self._view_var   = tk.StringVar(value="blend")
        self._name_a     = tk.StringVar(value="—")
        self._name_b     = tk.StringVar(value="—")
        self._name_m     = tk.StringVar(value="—")

        self._build_menu()
        self._build_layout()
        self._update_status("Pronto. Carregando imagens padrão…")
        self.after(100, self._try_load_defaults)

    # ───────────────────── Construção da UI ────────────────

    def _build_menu(self):
        mb = tk.Menu(self, bg=BG_PANEL, fg=FG_TEXT,
                     activebackground=BG_BTN_HL, activeforeground=FG_TEXT,
                     relief="flat", bd=0)
        mf = tk.Menu(mb, tearoff=0, bg=BG_PANEL, fg=FG_TEXT,
                     activebackground=BG_BTN_HL, activeforeground=FG_TEXT)
        mf.add_command(label="Salvar resultado…  Ctrl+S", command=self.cmd_save)
        mf.add_separator()
        mf.add_command(label="Sair  Alt+F4", command=self.destroy)
        mb.add_cascade(label="Arquivo", menu=mf)

        mh = tk.Menu(mb, tearoff=0, bg=BG_PANEL, fg=FG_TEXT,
                     activebackground=BG_BTN_HL, activeforeground=FG_TEXT)
        mh.add_command(label="Sobre", command=self._show_about)
        mb.add_cascade(label="Ajuda", menu=mh)

        self.config(menu=mb)
        self.bind("<Control-s>", lambda e: self.cmd_save())

    def _build_layout(self):
        paned = tk.PanedWindow(self, orient="horizontal", bg=BG_DARK,
                               sashwidth=5, sashrelief="flat")
        paned.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        # ── Painel esquerdo (scrollável, 250px) ─────────────
        left_outer = tk.Frame(paned, bg=BG_PANEL, width=250)
        left_outer.pack_propagate(False)
        paned.add(left_outer, minsize=230)

        sf = _ScrollFrame(left_outer, bg=BG_PANEL)
        sf.pack(fill="both", expand=True)
        p = sf.inner

        tk.Label(p, text="🔺  Blending  Pirâmides", font=FONT_H1,
                 bg=BG_PANEL, fg=ACCENT).pack(pady=(14, 6), padx=10)
        ttk.Separator(p, orient="horizontal").pack(fill="x", padx=10)

        btn = dict(font=FONT_MAIN, bg=BG_BTN, fg=FG_TEXT,
                   activebackground=BG_BTN_HL, activeforeground=FG_TEXT,
                   relief="flat", bd=0, cursor="hand2", width=28, pady=7)

        # — Imagens ——————————————————————————————————————————
        self._sec(p, "Imagens de Entrada")

        tk.Button(p, text="📂  Carregar Imagem A",
                  command=self.cmd_load_a, **btn).pack(pady=(8, 1), padx=10)
        tk.Label(p, textvariable=self._name_a, font=FONT_MONO,
                 bg=BG_PANEL, fg=FG_DIM).pack(padx=16, anchor="w")

        tk.Button(p, text="📂  Carregar Imagem B",
                  command=self.cmd_load_b, **btn).pack(pady=(6, 1), padx=10)
        tk.Label(p, textvariable=self._name_b, font=FONT_MONO,
                 bg=BG_PANEL, fg=FG_DIM).pack(padx=16, anchor="w")

        tk.Button(p, text="🎭  Carregar Máscara",
                  command=self.cmd_load_mask, **btn).pack(pady=(6, 1), padx=10)
        tk.Label(p, textvariable=self._name_m, font=FONT_MONO,
                 bg=BG_PANEL, fg=FG_DIM).pack(padx=16, anchor="w")

        ttk.Separator(p, orient="horizontal").pack(fill="x", padx=10, pady=8)

        # — Parâmetros ——————————————————————————————————————
        self._sec(p, "Parâmetros")

        tk.Label(p, text="Níveis da pirâmide:", font=FONT_MAIN,
                 bg=BG_PANEL, fg=FG_TEXT).pack(padx=14, anchor="w", pady=(6, 0))

        row_lvl = tk.Frame(p, bg=BG_PANEL)
        row_lvl.pack(fill="x", padx=12, pady=2)
        self._lbl_levels = tk.Label(row_lvl, text="6", font=FONT_H2,
                                    bg=BG_PANEL, fg=ACCENT, width=3)
        self._lbl_levels.pack(side="right")
        tk.Scale(row_lvl, from_=2, to=8, orient="horizontal",
                 variable=self._levels_var, showvalue=False,
                 command=lambda v: self._lbl_levels.config(text=str(int(float(v)))),
                 bg=BG_PANEL, fg=FG_TEXT, troughcolor=BG_BTN,
                 highlightthickness=0, sliderrelief="flat",
                 activebackground=ACCENT).pack(side="left", fill="x", expand=True)

        self._lbl_size_hint = tk.Label(p, text="", font=FONT_MONO,
                                       bg=BG_PANEL, fg=FG_DIM)
        self._lbl_size_hint.pack(padx=14, anchor="w")

        ttk.Separator(p, orient="horizontal").pack(fill="x", padx=10, pady=8)

        # — Executar ————————————————————————————————————————
        self._sec(p, "Processar")

        tk.Button(p, text="▶  Executar Blending",
                  command=self.cmd_run,
                  **{**btn, "bg": "#253a5e", "activebackground": ACCENT}
                  ).pack(pady=(8, 4), padx=10)

        ttk.Separator(p, orient="horizontal").pack(fill="x", padx=10, pady=8)

        # — Visualização ————————————————————————————————————
        self._sec(p, "Visualização")

        for label, val in [
            ("Resultado — Pirâmide",  "blend"),
            ("Justaposição direta",   "direct"),
            ("Comparação (lado a lado)", "compare"),
        ]:
            tk.Radiobutton(p, text=label, variable=self._view_var, value=val,
                           command=self._refresh_canvas,
                           font=FONT_MAIN, bg=BG_PANEL, fg=FG_TEXT,
                           selectcolor=BG_BTN, activebackground=BG_PANEL,
                           activeforeground=ACCENT).pack(anchor="w", padx=18, pady=2)

        ttk.Separator(p, orient="horizontal").pack(fill="x", padx=10, pady=8)

        tk.Button(p, text="💾  Salvar Resultado",
                  command=self.cmd_save, **btn).pack(pady=(2, 12), padx=10)

        # ── Canvas direito ──────────────────────────────────
        right = tk.Frame(paned, bg=BG_DARK)
        paned.add(right, minsize=600)

        self._canvas = tk.Canvas(right, bg="#111118", highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Configure>", lambda e: self._refresh_canvas())

        self._placeholder = self._canvas.create_text(
            500, 300,
            text="Carregue as imagens e clique em ▶ Executar Blending\n"
                 "(as imagens padrão da atividade são carregadas automaticamente)",
            fill=FG_DIM, font=("Segoe UI", 13), anchor="center", justify="center"
        )

        # ── Barra de status ─────────────────────────────────
        self._status_var = tk.StringVar()
        tk.Label(self, textvariable=self._status_var, font=FONT_MONO,
                 bg="#141420", fg=FG_DIM, anchor="w", padx=10, pady=4
                 ).pack(side="bottom", fill="x")

    def _sec(self, parent, title):
        tk.Label(parent, text=title, font=("Segoe UI", 9, "italic"),
                 bg=BG_PANEL, fg=FG_DIM).pack(padx=14, anchor="w", pady=(4, 0))

    # ───────────────────── Carregamento ────────────────────

    def _try_load_defaults(self):
        d = self._DIR
        paths = {
            "a": os.path.join(d, "512_burt_apple.png"),
            "b": os.path.join(d, "512_burt_orange.png"),
            "m": os.path.join(d, "512_mask.jpg"),
        }
        loaded = 0
        if os.path.exists(paths["a"]):
            self._load_a(paths["a"]); loaded += 1
        if os.path.exists(paths["b"]):
            self._load_b(paths["b"]); loaded += 1
        if os.path.exists(paths["m"]):
            self._load_m(paths["m"]); loaded += 1

        if loaded == 3:
            self._update_status(
                "Imagens padrão carregadas (maçã, laranja, máscara). "
                "Clique em ▶ Executar Blending.")
        else:
            self._update_status("Carregue a Imagem A, Imagem B e a Máscara.")

    def cmd_load_a(self):
        p = filedialog.askopenfilename(
            title="Imagem A",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("Todos", "*.*")])
        if p:
            self._load_a(p)

    def cmd_load_b(self):
        p = filedialog.askopenfilename(
            title="Imagem B",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("Todos", "*.*")])
        if p:
            self._load_b(p)

    def cmd_load_mask(self):
        p = filedialog.askopenfilename(
            title="Máscara (grayscale ou RGB — será convertida para L)",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp"), ("Todos", "*.*")])
        if p:
            self._load_m(p)

    def _load_a(self, path):
        self._img_a = np.array(Image.open(path).convert("RGB"), dtype=np.float64)
        self._name_a.set(os.path.basename(path))
        self._update_hint()

    def _load_b(self, path):
        self._img_b = np.array(Image.open(path).convert("RGB"), dtype=np.float64)
        self._name_b.set(os.path.basename(path))
        self._update_hint()

    def _load_m(self, path):
        arr = np.array(Image.open(path).convert("L"), dtype=np.float64) / 255.0
        self._mask = arr[:, :, np.newaxis]   # H×W×1  (broadcast para H×W×3)
        self._name_m.set(os.path.basename(path))

    def _update_hint(self):
        """Mostra as dimensões das imagens e o menor nível máximo possível."""
        a = self._img_a
        b = self._img_b
        if a is None and b is None:
            return
        parts = []
        if a is not None:
            parts.append(f"A: {int(a.shape[1])}×{int(a.shape[0])}")
        if b is not None:
            parts.append(f"B: {int(b.shape[1])}×{int(b.shape[0])}")
        self._lbl_size_hint.config(text="  ".join(parts))

    # ───────────────────── Processamento ───────────────────

    def cmd_run(self):
        if self._img_a is None or self._img_b is None or self._mask is None:
            messagebox.showwarning(
                "Atenção", "Carregue a Imagem A, a Imagem B e a Máscara antes de executar.")
            return

        # Garantir dimensões compatíveis ────────────────────
        ha, wa = self._img_a.shape[:2]
        hb, wb = self._img_b.shape[:2]
        hm, wm = self._mask.shape[:2]

        if (hb, wb) != (ha, wa):
            pil_b = Image.fromarray(self._img_b.astype(np.uint8)).resize(
                (wa, ha), Image.LANCZOS)
            self._img_b = np.array(pil_b, dtype=np.float64)
            self._update_status("Imagem B redimensionada para coincidir com A.")

        if (hm, wm) != (ha, wa):
            pil_m = Image.fromarray(
                (self._mask[:, :, 0] * 255).astype(np.uint8)
            ).resize((wa, ha), Image.LANCZOS)
            self._mask = np.array(pil_m, dtype=np.float64)[:, :, np.newaxis] / 255.0

        levels = self._levels_var.get()
        self._update_status(
            f"Calculando pirâmides com {levels} níveis… (pode demorar alguns segundos)")
        self.update_idletasks()

        try:
            mask_3ch = np.repeat(self._mask, 3, axis=2)   # H×W×3

            # Blending por pirâmides (Burt & Adelson 1983)
            self._blend_result = pyramid_blend(
                self._img_a, self._img_b, mask_3ch, levels)

            # Justaposição direta (comparação)
            self._direct_result = direct_cut(
                self._img_a, self._img_b, mask_3ch)

            self._refresh_canvas()
            self._update_status(
                f"Pronto! {levels} níveis — use as opções de visualização para comparar.")
        except Exception as exc:
            messagebox.showerror("Erro durante o blending", str(exc))
            self._update_status("Erro no processamento.")

    def cmd_save(self):
        img = self._get_canvas_image()
        if img is None:
            messagebox.showwarning("Aviso", "Execute o blending antes de salvar.")
            return
        path = filedialog.asksaveasfilename(
            title="Salvar resultado",
            initialfile="blending_resultado.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"),
                       ("Todos os arquivos", "*.*")])
        if not path:
            return
        try:
            Image.fromarray(img).save(path)
            self._update_status(f"Resultado salvo em: {path}")
        except Exception as exc:
            messagebox.showerror("Erro ao salvar", str(exc))

    # ───────────────────── Renderização ────────────────────

    def _get_canvas_image(self) -> np.ndarray | None:
        """Retorna o array uint8 correspondente à visualização selecionada."""
        view = self._view_var.get()
        if view == "blend":
            return self._blend_result
        if view == "direct":
            return self._direct_result
        if view == "compare":
            if self._blend_result is None or self._direct_result is None:
                return None
            h, w = self._blend_result.shape[:2]
            sep = 6
            combined = np.full((h, w * 2 + sep, 3), 20, dtype=np.uint8)
            combined[:, :w]         = self._blend_result
            combined[:, w+sep:]     = self._direct_result
            return combined
        return None

    def _refresh_canvas(self):
        arr = self._get_canvas_image()
        if arr is None:
            return

        self._canvas.delete(self._placeholder)
        cw = self._canvas.winfo_width()  or 900
        ch = self._canvas.winfo_height() or 600

        pil = Image.fromarray(arr)

        # Rótulos na visualização de comparação
        if self._view_var.get() == "compare":
            pil = self._add_compare_labels(pil, arr.shape[1] // 2)

        pil.thumbnail((cw, ch), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil)
        self._canvas.delete("img")
        self._canvas.create_image(cw // 2, ch // 2, anchor="center",
                                   image=self._photo, tags="img")

    def _add_compare_labels(self, pil: Image.Image, half_w: int) -> Image.Image:
        """Adiciona rótulos 'Pirâmide' e 'Direto' na imagem de comparação."""
        h = pil.height
        overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        bar_h = 28
        draw.rectangle([0, h - bar_h, half_w - 3, h], fill=(0, 0, 0, 170))
        draw.rectangle([half_w + 3, h - bar_h, pil.width, h], fill=(0, 0, 0, 170))
        result = Image.alpha_composite(pil.convert("RGBA"), overlay).convert("RGB")
        d = ImageDraw.Draw(result)
        d.text((10, h - bar_h + 6),        "◀ Blending por Pirâmide",  fill=(170, 215, 255))
        d.text((half_w + 12, h - bar_h + 6), "Justaposição Direta ▶", fill=(255, 175, 175))
        return result

    # ───────────────────── Utilitários ─────────────────────

    def _update_status(self, msg: str):
        self._status_var.set(f"  {msg}")

    def _show_about(self):
        messagebox.showinfo(
            "Sobre",
            "Atividade Prática 02 — Blending com Pirâmides\n"
            "Técnica de Burt & Adelson (1983)\n\n"
            "Disciplina: Sistemas de Visão Computacional\n"
            "VipLab UFPI\n"
            "Autor: Heverton Macedo"
        )


# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = Atv02App()
    app.mainloop()
