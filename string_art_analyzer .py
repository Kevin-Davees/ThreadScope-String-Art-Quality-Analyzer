"""
String Art Quality Analyzer
============================
Simulates string/thread art from nail-sequence files, compares each simulation
against a reference photograph, and visualises the quality vs thread-count curve.

Requirements:  pip install pillow scikit-image matplotlib numpy
"""

import os, math, threading as _threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageTk
from skimage.draw import line as sk_line
from skimage.metrics import structural_similarity as ssim

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ─────────────────────────────────────────────────────────────────────────────
#  Core image / art helpers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(path: str, size: int) -> np.ndarray:
    """Load → grayscale → resize to square → mask to circle.  Returns uint8 array."""
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.LANCZOS)

    # circular crop
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, size - 1, size - 1], fill=255)
    result = Image.new("L", (size, size), 0)
    result.paste(img, mask=mask)
    return np.array(result, dtype=np.uint8)


def nail_positions(num_nails: int, size: int):
    """Uniformly distributed nail (x, y) coords on circle, starting at top."""
    cx = cy = size // 2
    r = size // 2 - 3
    return [
        (
            int(cx + r * math.cos(2 * math.pi * i / num_nails)),
            int(cy + r * math.sin(2 * math.pi * i / num_nails)),
        )
        for i in range(num_nails)
    ]


def simulate_thread_art(
    sequence: list[int],
    num_nails: int,
    size: int,
    opacity: float = 0.12,
) -> np.ndarray:
    """
    Render thread art onto a black canvas using additive (light-accumulation)
    blending, which approximates how overlapping threads brighten a region
    under studio / ambient lighting.

    Returns a uint8 grayscale array.
    """
    nails = nail_positions(num_nails, size)
    acc = np.zeros((size, size), dtype=np.float32)

    for i in range(len(sequence) - 1):
        n0, n1 = sequence[i], sequence[i + 1]
        if not (0 <= n0 < num_nails and 0 <= n1 < num_nails):
            continue
        x0, y0 = nails[n0]
        x1, y1 = nails[n1]

        rr, cc = sk_line(y0, x0, y1, x1)          # row = y, col = x
        valid = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
        acc[rr[valid], cc[valid]] += opacity

    # ── Realistic lighting ────────────────────────────────────────────────
    # 1. Gamma compression (≈ monitor gamma) to brighten mid-tones
    acc = np.clip(acc, 0.0, 1.0)
    acc = np.power(acc, 0.65)

    # 2. Soft blur → simulates thread diameter & slight focus fall-off
    pil_acc = Image.fromarray((acc * 255).astype(np.uint8))
    pil_acc = pil_acc.filter(ImageFilter.GaussianBlur(radius=0.6))
    arr = np.array(pil_acc, dtype=np.float32) / 255.0

    # 3. Re-clip after blur
    arr = np.clip(arr, 0.0, 1.0)

    # 4. Invert: black threads on white background
    arr = 1.0 - arr

    # 5. Circular mask — outside the frame is white (background)
    cx = cy = size // 2
    r = size // 2 - 3
    Y, X = np.ogrid[:size, :size]
    outside = (X - cx) ** 2 + (Y - cy) ** 2 > r ** 2
    arr[outside] = 1.0

    return (arr * 255).astype(np.uint8)


def compute_similarity(ref: np.ndarray, art: np.ndarray) -> tuple[float, float, float]:
    """
    Compare reference image and thread-art using SSIM.

    Because some string-art algorithms target the *inverted* intensity space
    (dark original → dense threads → bright art region), we evaluate both
    orientations and return the higher score so the graph is always meaningful.

    Returns: (best_pct, ssim_normal, ssim_inverted)
    """
    r = ref.astype(np.float32) / 255.0
    a = art.astype(np.float32) / 255.0

    s_normal   = ssim(r, a,        data_range=1.0)
    s_inverted = ssim(1.0 - r, a,  data_range=1.0)

    best = max(s_normal, s_inverted)
    # Map SSIM [-1, 1] → [0, 100 %]
    pct = max(0.0, best * 100.0)
    return pct, s_normal, s_inverted


def parse_sequence_file(path: str) -> list[int]:
    """Read comma/newline/space separated nail indices from a text file."""
    with open(path, "r") as fh:
        raw = fh.read()
    raw = raw.replace("\n", ",").replace("\r", ",").replace(" ", ",")
    seq = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            try:
                seq.append(int(tok))
            except ValueError:
                pass
    return seq


def thread_count_from_name(path: str):
    """Extract thread count from filename (e.g. '1400.txt' → 1400)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem)
    except ValueError:
        return stem          # fallback: use the raw string


# ─────────────────────────────────────────────────────────────────────────────
#  Output builders
# ─────────────────────────────────────────────────────────────────────────────

def _best_font(sizes=(14, 12)):
    """Return the largest available TrueType font, or the built-in default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                from PIL import ImageFont as _IF
                return [_IF.truetype(path, s) for s in sizes]
            except Exception:
                pass
    default = ImageFont.load_default()
    return [default, default]


def build_stitched_image(
    source_arr: np.ndarray,
    arts: list[tuple],          # [(count, arr, similarity_pct), …]
    cell_size: int = 600,
) -> Image.Image:
    """
    Lay out the original image + all thread-art images in a single horizontal
    strip.  Every cell is exactly cell_size × cell_size pixels (matching the
    working resolution), with a label band underneath showing thread count
    and similarity percentage.
    """
    PAD      = 14          # gap between cells
    LABEL_H  = 52          # height of the text band below each image
    BG       = (20, 20, 22)
    BORDER   = (60, 60, 65)

    all_cells = [("Original", Image.fromarray(source_arr), None)] + [
        (str(cnt), Image.fromarray(arr), sim) for cnt, arr, sim in arts
    ]

    n  = len(all_cells)
    W  = n * cell_size + (n + 1) * PAD
    H  = cell_size + LABEL_H + 2 * PAD

    canvas = Image.new("RGB", (W, H), BG)
    draw   = ImageDraw.Draw(canvas)
    font_lg, font_sm = _best_font((16, 13))

    for idx, (label, img, sim) in enumerate(all_cells):
        x = PAD + idx * (cell_size + PAD)
        y = PAD

        # full-resolution cell image
        cell_img = img.resize((cell_size, cell_size), Image.LANCZOS).convert("RGB")
        canvas.paste(cell_img, (x, y))

        # thin border around each cell
        draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1],
                       outline=BORDER, width=1)

        # label band
        ty = y + cell_size + 6
        draw.text(
            (x + cell_size // 2, ty),
            "Original" if label == "Original" else f"Threads: {label}",
            fill=(230, 230, 230),
            font=font_lg,
            anchor="mt",
        )
        if sim is not None:
            sim_color = (
                (120, 255, 140) if sim >= 60 else
                (255, 200,  80) if sim >= 35 else
                (255, 100, 100)
            )
            draw.text(
                (x + cell_size // 2, ty + 24),
                f"Similarity: {sim:.1f}%",
                fill=sim_color,
                font=font_sm,
                anchor="mt",
            )

    return canvas


# ─────────────────────────────────────────────────────────────────────────────
#  GUI
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("String Art Quality Analyzer")
        self.geometry("960x720")
        self.resizable(True, True)
        self.configure(bg="#1a1a1f")

        self._image_path: str | None = None
        self._txt_files: list[str] = []
        self._stitch_photo = None          # keep reference so GC won't collect

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame",        background="#1a1a1f")
        style.configure("TLabel",        background="#1a1a1f", foreground="#d4d4d4")
        style.configure("TLabelframe",   background="#1a1a1f", foreground="#7cb9e8")
        style.configure("TLabelframe.Label", background="#1a1a1f", foreground="#7cb9e8", font=("Arial", 10, "bold"))
        style.configure("TButton",       background="#2d5a8e", foreground="white", font=("Arial", 10, "bold"))
        style.configure("TNotebook",     background="#1a1a1f")
        style.configure("TNotebook.Tab", background="#252530", foreground="#a0a0b0", padding=[10, 4])
        style.map("TNotebook.Tab", background=[("selected", "#2d5a8e")], foreground=[("selected", "white")])

        outer = ttk.Frame(self, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        # ── Title ─────────────────────────────────────────────────────────────
        tk.Label(
            outer, text="🧵  String Art Quality Analyzer",
            font=("Arial", 17, "bold"), bg="#1a1a1f", fg="#7cb9e8"
        ).pack(pady=(0, 8))

        # ── Inputs ────────────────────────────────────────────────────────────
        inp = ttk.LabelFrame(outer, text="  Inputs  ", padding=10)
        inp.pack(fill=tk.X, pady=4)

        self._img_var  = tk.StringVar(value="No file selected")
        self._txt_var  = tk.StringVar(value="No files selected")
        self._make_file_row(inp, "Source Image :", self._img_var,
                            self._browse_image,  row=0)
        self._make_file_row(inp, "Thread Files :", self._txt_var,
                            self._browse_txt,    row=1)

        # ── Settings ──────────────────────────────────────────────────────────
        cfg = ttk.LabelFrame(outer, text="  Settings  ", padding=10)
        cfg.pack(fill=tk.X, pady=4)

        self._nails_var   = tk.IntVar(value=320)
        self._size_var    = tk.IntVar(value=735)
        self._opacity_var = tk.DoubleVar(value=0.05)

        row = ttk.Frame(cfg)
        row.pack(fill=tk.X)
        params = [
            ("Nails :",          self._nails_var,   50,   500, 1,    6),
            ("Image size px :",  self._size_var,    200, 1200, 50,   6),
            ("Thread opacity :", self._opacity_var, 0.01, 1.0, 0.01, 6),
        ]
        for label, var, lo, hi, inc, w in params:
            ttk.Label(row, text=label).pack(side=tk.LEFT, padx=(8, 2))
            ttk.Spinbox(row, from_=lo, to=hi, increment=inc,
                        textvariable=var, width=w).pack(side=tk.LEFT)

        # ── Run / progress ────────────────────────────────────────────────────
        ctrl = ttk.Frame(outer)
        ctrl.pack(fill=tk.X, pady=6)

        self._run_btn = ttk.Button(ctrl, text="▶  Run Analysis", command=self._start)
        self._run_btn.pack(side=tk.LEFT)

        self._progress = ttk.Progressbar(ctrl, maximum=100, length=400)
        self._progress.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)

        self._status_var = tk.StringVar(value="Ready")
        ttk.Label(ctrl, textvariable=self._status_var, width=22, anchor="e").pack(side=tk.RIGHT)

        # ── Notebook ──────────────────────────────────────────────────────────
        nb = ttk.Notebook(outer)
        nb.pack(fill=tk.BOTH, expand=True, pady=4)
        self._nb = nb

        self._graph_tab  = ttk.Frame(nb)
        self._stitch_tab = ttk.Frame(nb)
        self._log_tab    = ttk.Frame(nb)
        nb.add(self._graph_tab,  text="  📈 Similarity Graph  ")
        nb.add(self._stitch_tab, text="  🖼  Comparison Strip  ")
        nb.add(self._log_tab,    text="  📋 Log  ")

        self._log_box = tk.Text(
            self._log_tab, wrap=tk.WORD, bg="#0d0d12", fg="#c0c0c0",
            font=("Courier", 10), insertbackground="white"
        )
        sc = ttk.Scrollbar(self._log_tab, command=self._log_box.yview)
        self._log_box.configure(yscrollcommand=sc.set)
        sc.pack(side=tk.RIGHT, fill=tk.Y)
        self._log_box.pack(fill=tk.BOTH, expand=True)

    def _make_file_row(self, parent, label, var, cmd, row):
        f = ttk.Frame(parent)
        f.pack(fill=tk.X, pady=3)
        ttk.Label(f, text=label, width=16, anchor="e").pack(side=tk.LEFT)
        ttk.Label(f, textvariable=var, foreground="#888", width=50,
                  anchor="w").pack(side=tk.LEFT, padx=6)
        ttk.Button(f, text="Browse…", command=cmd).pack(side=tk.RIGHT)

    # ── File browsing ──────────────────────────────────────────────────────────

    def _browse_image(self):
        p = filedialog.askopenfilename(
            title="Select Source Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"), ("All", "*.*")],
        )
        if p:
            self._image_path = p
            self._img_var.set(os.path.basename(p))

    def _browse_txt(self):
        ps = filedialog.askopenfilenames(
            title="Select Thread-Pattern Files",
            filetypes=[("Text", "*.txt"), ("All", "*.*")],
        )
        if ps:
            self._txt_files = list(ps)
            self._txt_var.set(f"{len(ps)} file(s) selected")

    # ── Analysis ──────────────────────────────────────────────────────────────

    def _start(self):
        if not self._image_path:
            messagebox.showerror("Missing input", "Please select a source image.")
            return
        if not self._txt_files:
            messagebox.showerror("Missing input", "Please select at least one thread-pattern file.")
            return
        self._run_btn.configure(state=tk.DISABLED)
        self._log_box.delete("1.0", tk.END)
        t = _threading.Thread(target=self._run, daemon=True)
        t.start()

    def _log(self, msg: str):
        self.after(0, lambda m=msg: (
            self._log_box.insert(tk.END, m + "\n"),
            self._log_box.see(tk.END),
        ))

    def _set_status(self, msg: str, prog: float = None):
        self.after(0, lambda: self._status_var.set(msg))
        if prog is not None:
            self.after(0, lambda: self._progress.configure(value=prog))

    def _run(self):
        try:
            size    = self._size_var.get()
            nails   = self._nails_var.get()
            opacity = self._opacity_var.get()

            self._log("═" * 52)
            self._log("  String Art Quality Analyzer")
            self._log("═" * 52)
            self._log(f"  Image size : {size}×{size} px")
            self._log(f"  Nails      : {nails}")
            self._log(f"  Opacity    : {opacity}")

            # Step 1-3: pre-process reference image
            self._set_status("Loading image…", 2)
            self._log("\n[1/3] Loading & pre-processing image…")
            ref_arr = preprocess_image(self._image_path, size)
            self._log(f"      ✓ Loaded → grayscale → circular mask ({size}px)")

            # Sort files by thread count
            file_data = sorted(
                [(thread_count_from_name(p), p) for p in self._txt_files],
                key=lambda x: x[0] if isinstance(x[0], int) else 0,
            )

            self._log(f"\n[2/3] Simulating {len(file_data)} thread-art file(s)…")
            results  = []   # [(count, pct)]
            art_data = []   # [(count, arr, pct)]

            total = len(file_data)
            for i, (cnt, fpath) in enumerate(file_data):
                prog = 5 + int(80 * i / total)
                self._set_status(f"Simulating {cnt} threads… ({i+1}/{total})", prog)
                self._log(f"\n  ── {os.path.basename(fpath)}  ({cnt} threads) ──")

                seq = parse_sequence_file(fpath)
                self._log(f"     Nail indices in sequence : {len(seq)}")
                self._log(f"     Thread segments          : {len(seq) - 1}")

                art = simulate_thread_art(seq, nails, size, opacity)
                pct, s_n, s_i = compute_similarity(ref_arr, art)

                self._log(f"     SSIM (normal)   : {s_n:+.4f}")
                self._log(f"     SSIM (inverted) : {s_i:+.4f}")
                self._log(f"     ► Similarity    : {pct:.2f}%")

                results.append((cnt, pct))
                art_data.append((cnt, art, pct))

            # Step 4: outputs
            self._set_status("Building outputs…", 87)
            self._log("\n[3/3] Building outputs…")

            # Line graph
            self.after(0, self._draw_graph, results)

            # Stitched image
            stitch = build_stitched_image(ref_arr, art_data, cell_size=size)
            save_dir = os.path.dirname(self._image_path)
            save_path = os.path.join(save_dir, "string_art_comparison.png")
            stitch.save(save_path)
            self._log(f"     Saved stitched image → {save_path}")

            self.after(0, self._show_stitch, stitch, save_path)

            self._set_status("Done ✓", 100)
            self._log("\n" + "═" * 52)
            self._log("  Analysis complete.")
            self._log("═" * 52)

        except Exception as exc:
            import traceback
            self._log(f"\n❌ ERROR: {exc}")
            self._log(traceback.format_exc())
            self._set_status(f"Error: {exc}", 0)
        finally:
            self.after(0, lambda: self._run_btn.configure(state=tk.NORMAL))

    # ── Graph ─────────────────────────────────────────────────────────────────

    def _draw_graph(self, results):
        from matplotlib.figure import Figure

        for w in self._graph_tab.winfo_children():
            w.destroy()

        # close any stale pyplot figures
        plt.close("all")

        counts = [r[0] for r in results]
        sims   = [r[1] for r in results]

        # Use Figure directly — the pyplot interface breaks inside tkinter callbacks
        fig = Figure(figsize=(8, 4.5), facecolor="#0d0d12")
        ax  = fig.add_subplot(111)
        ax.set_facecolor("#13131a")

        ax.plot(counts, sims, color="#4fc3f7", linewidth=2.2, zorder=3)
        ax.scatter(counts, sims, color="#ff6e40", s=60, zorder=4)

        for x, y in zip(counts, sims):
            ax.annotate(
                f"{y:.1f}%", (x, y),
                textcoords="offset points", xytext=(0, 11),
                ha="center", fontsize=8, color="#cccccc",
            )

        ax.set_xlabel("Number of Threads", color="#aaaaaa", fontsize=11)
        ax.set_ylabel("Similarity (%)",    color="#aaaaaa", fontsize=11)
        ax.set_title("String Art Quality vs Thread Count",
                     color="#7cb9e8", fontsize=13, fontweight="bold", pad=10)
        ax.tick_params(axis="both", color="#888888", labelcolor="#888888")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")
        ax.grid(True, color="#222233", linewidth=0.7, alpha=0.8)
        ax.set_ylim(0, 100)
        fig.tight_layout()

        # embed in tkinter and keep a strong reference on self
        self._graph_canvas = FigureCanvasTkAgg(fig, master=self._graph_tab)
        self._graph_canvas.draw()
        self._graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # show graph tab — it is the primary result
        self._nb.select(self._graph_tab)

    # ── Stitched image viewer ─────────────────────────────────────────────────

    def _show_stitch(self, stitch: Image.Image, path: str):
        for w in self._stitch_tab.winfo_children():
            w.destroy()

        # Header bar
        hdr = ttk.Frame(self._stitch_tab)
        hdr.pack(fill=tk.X, padx=6, pady=4)
        ttk.Label(hdr, text=f"Saved → {path}", foreground="#888").pack(side=tk.LEFT)

        # Scrollable canvas
        cont = ttk.Frame(self._stitch_tab)
        cont.pack(fill=tk.BOTH, expand=True)

        hbar = ttk.Scrollbar(cont, orient=tk.HORIZONTAL)
        vbar = ttk.Scrollbar(cont, orient=tk.VERTICAL)
        cv   = tk.Canvas(cont, bg="#0d0d12",
                         xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        hbar.config(command=cv.xview)
        vbar.config(command=cv.yview)
        hbar.pack(side=tk.BOTTOM, fill=tk.X)
        vbar.pack(side=tk.RIGHT,  fill=tk.Y)
        cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Fit to window width while keeping aspect
        win_w = max(self.winfo_width() - 40, 400)
        scale = min(1.0, win_w / stitch.width)
        dw = int(stitch.width  * scale)
        dh = int(stitch.height * scale)

        display = stitch.resize((dw, dh), Image.LANCZOS)
        self._stitch_photo = ImageTk.PhotoImage(display)
        cv.create_image(0, 0, anchor=tk.NW, image=self._stitch_photo)
        cv.configure(scrollregion=(0, 0, dw, dh))

        # don't switch tabs here — graph tab is shown first; user clicks across


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
