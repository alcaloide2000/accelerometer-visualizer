import os
import io
import datetime
import tkinter as tk
import mplcursors
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import numpy as np
from scipy.signal import windows, find_peaks, butter, sosfiltfilt, hilbert
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

# ── palette ───────────────────────────────────────────────────────────────────
TEST_STYLES = [
    {"color": "#f97316", "ls": "-",  "marker": "o"},
    {"color": "#22d3ee", "ls": "--", "marker": "s"},
    {"color": "#4ade80", "ls": ":",  "marker": "^"},
    {"color": "#f472b6", "ls": "-.", "marker": "D"},
    {"color": "#facc15", "ls": "-",  "marker": "v"},
    {"color": "#a78bfa", "ls": "--", "marker": "P"},
    {"color": "#34d399", "ls": ":",  "marker": "*"},
    {"color": "#fb923c", "ls": "-.", "marker": "X"},
]
SENSOR_COLORS = ["#f97316", "#22d3ee", "#a78bfa"]
BG_DARK  = "#12121c"
BG_MID   = "#1a1a2e"
BG_PANEL = "#2a2a3e"
FG_MAIN  = "#e0e0f0"
FG_DIM   = "#a0a0b0"
GRID_COL = "#2a2a4e"

# decay method: an event must peak at least this many times the median
# envelope (ambient level) to count as an excitation
DAMP_MIN_EXCITATION = 8.0
# default decay-method parameters (Damping window, peak tables, reports)
DAMP_DEFAULTS = dict(band=0.5, win_s=5.0, n_events=3)


# ── data helpers ──────────────────────────────────────────────────────────────
def parse_file(filepath):
    metadata, data_lines, in_data = {}, [], False
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if "Timestamp;Measure Value" in line:
                in_data = True
                continue
            if in_data:
                if line.strip():
                    data_lines.append(line.strip())
            elif ":" in line and not line.startswith("-"):
                key, _, val = line.partition(":")
                metadata[key.strip()] = val.strip()
    rows = []
    for line in data_lines:
        parts = line.split(";")
        if len(parts) == 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                pass
    return metadata, pd.DataFrame(rows, columns=["timestamp", "value"])


def _parse_data_only(filepath):
    rows = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(";")
            if len(parts) == 2:
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
    return pd.DataFrame(rows, columns=["timestamp", "value"])


def parse_channel_folder(folder):
    txts       = sorted(f for f in os.listdir(folder) if f.lower().endswith(".txt"))
    main_files = [f for f in txts if "_part" not in f]
    part_files = [f for f in txts if "_part" in f]
    if not main_files:
        raise ValueError(f"No main file in {folder}")
    meta, df_main = parse_file(os.path.join(folder, main_files[0]))
    frames = [df_main] + [
        _parse_data_only(os.path.join(folder, pf)) for pf in part_files
    ]
    return meta, pd.concat(frames, ignore_index=True)


def load_test_folder(test_folder):
    subdirs = sorted(
        d for d in os.listdir(test_folder)
        if os.path.isdir(os.path.join(test_folder, d)) and d.lower().startswith("acel")
    )
    if not subdirs:
        raise ValueError(f"No acel* subfolders found in {test_folder}")
    channels = []
    for d in subdirs:
        meta, df = parse_channel_folder(os.path.join(test_folder, d))
        fs       = float(meta.get("Sampling rate", "250") or "250")
        frq, amp = compute_fft(df["value"].to_numpy(), fs)
        channels.append(dict(label=d, meta=meta, df=df, frq=frq, amp=amp, pidx=None))
    return channels


def compute_fft(values, fs):
    n   = len(values)
    win = windows.hann(n)
    sig = (values - values.mean()) * win
    amp = (2.0 / win.sum()) * np.abs(np.fft.rfft(sig))
    frq = np.fft.rfftfreq(n, d=1.0 / fs)
    return frq, amp


def top_peaks(freqs, amp, n=6):
    min_prom = amp.max() * 0.01
    idx, _   = find_peaks(amp, prominence=min_prom, distance=5)
    idx      = idx[np.argsort(amp[idx])[::-1]][:n]
    return np.sort(idx)


def decay_zeta(ch, f0):
    """Mean decay-method zeta (%) of a channel at f0 with DAMP_DEFAULTS, or
    None if no decay was found.  Cached on the channel (~60 ms per call)."""
    cache = ch.setdefault("zeta_cache", {})
    key = round(float(f0), 4)
    if key not in cache:
        fs  = float(ch["meta"].get("Sampling rate", "250") or "250")
        evs = decay_damping(ch["df"]["value"].to_numpy(), fs, key, **DAMP_DEFAULTS)
        cache[key] = float(np.mean([e["zeta"] for e in evs])) if evs else None
    return cache[key]


def peak_zeta(ch, rank):
    """Damping string for the rank-th detected peak of a channel."""
    pidx = ch["pidx"] if ch["pidx"] is not None else []
    if rank >= len(pidx):
        return "—"
    z = decay_zeta(ch, ch["frq"][pidx[rank]])
    return f"{z:.2f} %" if z is not None else "—"


def decay_damping(values, fs, f0, band=0.5, win_s=5.0, n_events=3):
    """Free-decay damping (envelope / log-decrement method).

    Band-pass the signal around f0, pick the n_events largest excitations
    (envelope maxima at least 3 windows apart), and for each one fit a
    least-squares line to ln(maxima) vs time over the following win_s
    seconds.  zeta % = -slope / (2*pi*f0) * 100.  Events below
    DAMP_MIN_EXCITATION x the median envelope are ignored, and fitting stops
    early when the maxima fall to the ambient level (2x the median envelope)."""
    lo, hi = max(f0 - band, 0.05), min(f0 + band, fs / 2 * 0.99)
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    y   = sosfiltfilt(sos, values - values.mean())
    env = np.abs(hilbert(y))
    floor = 2.0 * np.median(env)
    n_win = int(win_s * fs)

    pk, _ = find_peaks(env[: len(env) - n_win], distance=max(1, 3 * n_win))
    pk = pk[env[pk] >= DAMP_MIN_EXCITATION * np.median(env)]   # real excitations only
    pk = np.sort(pk[np.argsort(env[pk])[::-1]][:n_events])

    events = []
    for p0 in pk:
        seg = y[p0 : p0 + n_win]
        mx, _ = find_peaks(seg, distance=max(1, int(0.7 * fs / f0)))
        mx = mx[seg[mx] > 0]
        below = np.nonzero(seg[mx] < floor)[0]
        if len(below):
            mx = mx[: below[0]]
        if len(mx) < 5:
            continue
        t_mx = (p0 + mx) / fs
        ln_a = np.log(seg[mx])
        slope, icpt = np.polyfit(t_mx, ln_a, 1)
        r2 = float(np.corrcoef(t_mx, ln_a)[0, 1] ** 2)
        ctx = int(1.0 * fs)                   # 1 s of context before the event
        a, b = max(p0 - ctx, 0), min(p0 + n_win, len(y))
        events.append(dict(
            t0=p0 / fs, slope=slope, icpt=icpt, r2=r2,
            zeta=-slope / (2 * np.pi * f0) * 100.0,
            t_seg=np.arange(a, b) / fs, y_seg=y[a:b],
            t_mx=t_mx, a_mx=seg[mx], include=True,
        ))
    return events


def dominant_f0(channels):
    """Median of each channel's dominant FFT peak — the default f0."""
    dom = [ch["frq"][top_peaks(ch["frq"], ch["amp"], 1)[0]] for ch in channels]
    return float(np.median(dom))


def run_damping(channels, f0, band, win_s, n_events):
    """decay_damping for every channel → [(sensor index, channel, event)]."""
    rows = []
    for s_idx, ch in enumerate(channels):
        fs = float(ch["meta"].get("Sampling rate", "250") or "250")
        for ev in decay_damping(ch["df"]["value"].to_numpy(), fs,
                                f0, band, win_s, n_events):
            rows.append((s_idx, ch, ev))
    return rows


def damping_means(channels, rows):
    """Per-sensor mean zeta (a1, a2, a3 …) and overall mean of included events."""
    per, all_z = [], []
    for s_idx in range(len(channels)):
        z = [ev["zeta"] for si, _, ev in rows if si == s_idx and ev["include"]]
        all_z += z
        per.append(float(np.mean(z)) if z else None)
    return per, (float(np.mean(all_z)) if all_z else None)


def plot_decay(ax_sig, ax_reg, ch, ev, f0, band, color):
    """Filtered signal + envelope of maxima, and ln(envelope) regression."""
    unit = ch["meta"].get("Unit for accelerometer", "g")
    ax_sig.clear(); style_ax(ax_sig)
    ax_sig.plot(ev["t_seg"], ev["y_seg"], color="#1e40af", linewidth=0.8,
                label=f"filtered {f0:.2f} ± {band:g} Hz")
    ax_sig.plot(ev["t_mx"], ev["a_mx"], color="red", linewidth=1.2,
                label="envelope of maxima")
    ax_sig.set_title(f"{ch['label'].upper()} — filtered around {f0:.2f} Hz",
                     color=color, fontsize=10, fontweight="bold")
    ax_sig.set_xlabel("Time (s)", fontsize=8)
    ax_sig.set_ylabel(f"Acc ({unit})", fontsize=8)
    ax_sig.legend(fontsize=7, loc="upper right")

    ax_reg.clear(); style_ax(ax_reg)
    ax_reg.plot(ev["t_mx"], np.log(ev["a_mx"]), "^", color="red",
                markerfacecolor="none", label="ln(envelope)")
    ax_reg.plot(ev["t_mx"], ev["slope"] * ev["t_mx"] + ev["icpt"],
                color="blue", linewidth=1.0,
                label=f"fit: m = {ev['slope']:.4f} 1/s,  R² = {ev['r2']:.2f}")
    ax_reg.set_title(f"Linear regression of envelope — ζ = {ev['zeta']:.3f} %",
                     color="#111111", fontsize=10, fontweight="bold")
    ax_reg.set_xlabel("Time (s)", fontsize=8)
    ax_reg.set_ylabel("ln(amplitude)", fontsize=8)
    ax_reg.legend(fontsize=7, loc="upper right")


def style_ax(ax):
    ax.set_facecolor("white")
    ax.tick_params(colors="#333333", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#aaaaaa")
    ax.grid(True, color="#dddddd", linewidth=0.5)


# ── app ───────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Accelerometer Visualizer")
        self.geometry("1400x940")
        self.configure(bg=BG_DARK)
        self.tests       = []   # for comparison mode
        self.single_test = None  # for single mode
        self.overlay_test = None  # for 3-accel overlay mode
        self.mode        = tk.StringVar(value="compare")
        self._cursors    = []   # active mplcursors instances
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── mode toggle bar ──────────────────────────────────────────────────
        mode_bar = tk.Frame(self, bg="#0f0f1a")
        mode_bar.pack(fill="x", padx=0, pady=0)

        for text, val in (("Single Test", "single"), ("Compare Tests", "compare"), ("Compare 3 Accelerometers", "overlay")):
            tk.Radiobutton(
                mode_bar, text=text, variable=self.mode, value=val,
                command=self._on_mode_change,
                bg="#0f0f1a", fg=FG_DIM, selectcolor="#7c3aed",
                activebackground="#0f0f1a", activeforeground=FG_MAIN,
                font=("Segoe UI", 10, "bold"), indicatoron=False,
                relief="flat", padx=18, pady=6, cursor="hand2",
            ).pack(side="left")

        # ── single-test toolbar (hidden initially) ───────────────────────────
        self.single_bar = tk.Frame(self, bg=BG_DARK)
        tk.Button(self.single_bar, text="Select Test Folder",
                  command=self._load_single,
                  bg="#7c3aed", fg="white", relief="flat", padx=14, pady=4,
                  font=("Segoe UI", 10, "bold"), cursor="hand2").pack(side="left")
        self.single_lbl = tk.Label(self.single_bar, text="No test loaded",
                                   bg=BG_DARK, fg=FG_DIM, font=("Segoe UI", 9))
        self.single_lbl.pack(side="left", padx=12)
        tk.Button(self.single_bar, text="📄 Generate Report",
                  command=self._generate_report,
                  bg="#065f46", fg="white", relief="flat", padx=10, pady=4,
                  font=("Segoe UI", 9, "bold"), cursor="hand2").pack(side="left", padx=(10, 0))

        # ── compare toolbar (visible initially) ──────────────────────────────
        self.compare_bar = tk.Frame(self, bg=BG_DARK)
        for text, cmd, bg in (
            ("Load Parent Folder",  self._load_parent,              "#7c3aed"),
            ("Select 2 Tests",      lambda: self._select_n_tests(2), "#0369a1"),
            ("Select 3 Tests",      lambda: self._select_n_tests(3), "#0f766e"),
            ("Add Test",            self._add_test,                 "#374151"),
            ("Clear All",           self._clear,                    "#1f2937"),
        ):
            tk.Button(self.compare_bar, text=text, command=cmd, bg=bg, fg="white",
                      relief="flat", padx=10, pady=4,
                      font=("Segoe UI", 9, "bold"), cursor="hand2").pack(side="left", padx=(0, 5))
        self.compare_lbl = tk.Label(self.compare_bar, text="No tests loaded",
                                    bg=BG_DARK, fg=FG_DIM, font=("Segoe UI", 9))
        self.compare_lbl.pack(side="left", padx=6)
        tk.Button(self.compare_bar, text="📄 Generate Report",
                  command=self._generate_report,
                  bg="#065f46", fg="white", relief="flat", padx=10, pady=4,
                  font=("Segoe UI", 9, "bold"), cursor="hand2").pack(side="left", padx=(6, 0))

        # ── overlay toolbar ──────────────────────────────────────────────────
        self.overlay_bar = tk.Frame(self, bg=BG_DARK)
        tk.Button(self.overlay_bar, text="Select Test Folder",
                  command=self._load_overlay,
                  bg="#7c3aed", fg="white", relief="flat", padx=14, pady=4,
                  font=("Segoe UI", 10, "bold"), cursor="hand2").pack(side="left")
        self.overlay_lbl = tk.Label(self.overlay_bar, text="No test loaded",
                                    bg=BG_DARK, fg=FG_DIM, font=("Segoe UI", 9))
        self.overlay_lbl.pack(side="left", padx=12)
        tk.Button(self.overlay_bar, text="📄 Generate Report",
                  command=self._generate_report,
                  bg="#065f46", fg="white", relief="flat", padx=10, pady=4,
                  font=("Segoe UI", 9, "bold"), cursor="hand2").pack(side="left", padx=(10, 0))

        # shared: damping window button on every bar
        for bar in (self.single_bar, self.compare_bar, self.overlay_bar):
            tk.Button(bar, text="〰 Damping",
                      command=self._show_damping,
                      bg="#9a3412", fg="white", relief="flat", padx=10, pady=4,
                      font=("Segoe UI", 9, "bold"), cursor="hand2").pack(side="left", padx=(6, 0))

        # shared: peaks spinbox on the right of whichever bar is active
        for bar in (self.single_bar, self.compare_bar, self.overlay_bar):
            tk.Label(bar, text="Top peaks:", bg=BG_DARK, fg=FG_DIM,
                     font=("Segoe UI", 9)).pack(side="right", padx=(4, 2))
        self.n_peaks_var = tk.IntVar(value=6)
        # attach one spinbox to compare bar (single bar gets its own reference)
        self._pk_spin_c = tk.Spinbox(self.compare_bar, from_=1, to=20,
                                     textvariable=self.n_peaks_var, width=4,
                                     bg=BG_PANEL, fg="white",
                                     buttonbackground="#3a3a5e",
                                     command=self._replot)
        self._pk_spin_c.pack(side="right")
        self._pk_spin_s = tk.Spinbox(self.single_bar, from_=1, to=20,
                                     textvariable=self.n_peaks_var, width=4,
                                     bg=BG_PANEL, fg="white",
                                     buttonbackground="#3a3a5e",
                                     command=self._replot)
        self._pk_spin_s.pack(side="right")
        self._pk_spin_ov = tk.Spinbox(self.overlay_bar, from_=1, to=20,
                                      textvariable=self.n_peaks_var, width=4,
                                      bg=BG_PANEL, fg="white",
                                      buttonbackground="#3a3a5e",
                                      command=self._replot)
        self._pk_spin_ov.pack(side="right")

        # legend strip
        self.legend_frame = tk.Frame(self, bg=BG_PANEL)
        self.legend_frame.pack(fill="x", padx=10, pady=(0, 4))

        # canvas placeholder — rebuilt on mode change
        self.canvas_frame = tk.Frame(self, bg=BG_DARK)
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=(0, 4))
        self.tb_frame = tk.Frame(self, bg=BG_DARK)
        self.tb_frame.pack(fill="x", padx=10)

        self.fig    = None
        self.canvas = None

        self._on_mode_change()   # initialise correct mode

    # ── mode switching ────────────────────────────────────────────────────────
    def _on_mode_change(self):
        mode = self.mode.get()
        self.single_bar.pack_forget()
        self.compare_bar.pack_forget()
        self.overlay_bar.pack_forget()
        if mode == "single":
            self.single_bar.pack(fill="x", padx=10, pady=6)
        elif mode == "compare":
            self.compare_bar.pack(fill="x", padx=10, pady=6)
        else:
            self.overlay_bar.pack(fill="x", padx=10, pady=6)
        self._rebuild_figure()
        self._replot()

    def _rebuild_figure(self):
        """Destroy and recreate the matplotlib figure for the current mode."""
        # destroy old canvas + toolbar
        for w in self.canvas_frame.winfo_children():
            w.destroy()
        for w in self.tb_frame.winfo_children():
            w.destroy()
        if self.fig:
            plt.close(self.fig)

        if self.mode.get() == "single":
            self.fig = plt.figure(figsize=(14, 8), facecolor="white")
            gs = self.fig.add_gridspec(2, 3, height_ratios=[1, 1.4],
                                       hspace=0.45, wspace=0.32)
            self.ax_time = [self.fig.add_subplot(gs[0, i]) for i in range(3)]
            self.ax_fft  = [self.fig.add_subplot(gs[1, i]) for i in range(3)]
            for ax in self.ax_time + self.ax_fft:
                style_ax(ax)
        elif self.mode.get() == "compare":
            self.fig = plt.figure(figsize=(14, 8), facecolor="white")
            gs = self.fig.add_gridspec(2, 3, height_ratios=[2, 1],
                                       hspace=0.45, wspace=0.32)
            self.ax_fft = [self.fig.add_subplot(gs[0, i]) for i in range(3)]
            self.ax_tbl =  self.fig.add_subplot(gs[1, :])
            self.ax_tbl.axis("off")
            for ax in self.ax_fft:
                style_ax(ax)
        else:  # overlay
            self.fig = plt.figure(figsize=(14, 8), facecolor="white")
            gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1],
                                       hspace=0.35)
            self.ax_ov  = self.fig.add_subplot(gs[0])
            self.ax_tbl = self.fig.add_subplot(gs[1])
            self.ax_tbl.axis("off")
            style_ax(self.ax_ov)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, self.tb_frame)

    # ── single-test loading ───────────────────────────────────────────────────
    def _load_single(self):
        folder = filedialog.askdirectory(title="Select a test folder")
        if not folder:
            return
        try:
            channels = load_test_folder(folder)
            n = self.n_peaks_var.get()
            for ch in channels:
                ch["pidx"] = top_peaks(ch["frq"], ch["amp"], n)
            self.single_test = {"name": os.path.basename(folder), "channels": channels}
            self.single_lbl.config(text=f"Loaded: {self.single_test['name']}")
            self._rebuild_legend_single()
            self._plot_single()
        except Exception as e:
            messagebox.showerror("Error loading test", str(e))

    def _rebuild_legend_single(self):
        for w in self.legend_frame.winfo_children():
            w.destroy()
        if not self.single_test:
            return
        tk.Label(self.legend_frame, text=f"  {self.single_test['name']}  ",
                 bg=BG_PANEL, fg=FG_MAIN,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        for i, ch in enumerate(self.single_test["channels"]):
            c = SENSOR_COLORS[i % len(SENSOR_COLORS)]
            fs = float(ch["meta"].get("Sampling rate", "250") or "250")
            n  = len(ch["df"])
            tk.Label(self.legend_frame,
                     text=f"  ■ {ch['label'].upper()}  {n:,} pts  {n/fs:.0f} s",
                     bg=BG_PANEL, fg=c, font=("Segoe UI", 8)).pack(side="left", padx=6)

    def _load_overlay(self):
        folder = filedialog.askdirectory(title="Select a test folder")
        if not folder:
            return
        try:
            channels = load_test_folder(folder)
            n = self.n_peaks_var.get()
            for ch in channels:
                ch["pidx"] = top_peaks(ch["frq"], ch["amp"], n)
            self.overlay_test = {"name": os.path.basename(folder), "channels": channels}
            self.overlay_lbl.config(text=f"Loaded: {self.overlay_test['name']}")
            self._rebuild_legend_overlay()
            self._plot_overlay()
        except Exception as e:
            messagebox.showerror("Error loading test", str(e))

    def _rebuild_legend_overlay(self):
        for w in self.legend_frame.winfo_children():
            w.destroy()
        if not self.overlay_test:
            return
        tk.Label(self.legend_frame, text=f"  {self.overlay_test['name']}  ",
                 bg=BG_PANEL, fg=FG_MAIN,
                 font=("Segoe UI", 9, "bold")).pack(side="left")
        for i, ch in enumerate(self.overlay_test["channels"]):
            c = SENSOR_COLORS[i % len(SENSOR_COLORS)]
            tk.Label(self.legend_frame,
                     text=f"  ■ {ch['label'].upper()}",
                     bg=BG_PANEL, fg=c, font=("Segoe UI", 8)).pack(side="left", padx=6)

    # ── compare loading ───────────────────────────────────────────────────────
    def _load_parent(self):
        parent = filedialog.askdirectory(title="Select parent folder containing test* folders")
        if not parent:
            return
        test_dirs = sorted(
            d for d in os.listdir(parent)
            if os.path.isdir(os.path.join(parent, d)) and d.lower().startswith("test")
        )
        if not test_dirs:
            messagebox.showwarning("No tests found", "No test* subfolders found.")
            return
        self.tests = []
        for d in test_dirs:
            try:
                channels = load_test_folder(os.path.join(parent, d))
                self.tests.append({"name": d, "channels": channels})
            except Exception as e:
                messagebox.showerror(f"Error loading {d}", str(e))
                return
        self._after_compare_load()

    def _select_n_tests(self, n):
        """Ask for exactly n test folders one by one, then replace current selection."""
        folders = []
        for i in range(n):
            folder = filedialog.askdirectory(
                title=f"Select test {i + 1} of {n}"
            )
            if not folder:
                return   # user cancelled — abort the whole operation
            folders.append(folder)
        self.tests = []
        for folder in folders:
            try:
                channels = load_test_folder(folder)
                self.tests.append({"name": os.path.basename(folder), "channels": channels})
            except Exception as e:
                messagebox.showerror("Error loading test", str(e))
                self.tests = []
                return
        self._after_compare_load()

    def _add_test(self):
        folder = filedialog.askdirectory(title="Select a test folder")
        if not folder:
            return
        try:
            channels = load_test_folder(folder)
            self.tests.append({"name": os.path.basename(folder), "channels": channels})
            self._after_compare_load()
        except Exception as e:
            messagebox.showerror("Error loading test", str(e))

    def _clear(self):
        self.tests = []
        self._after_compare_load()

    def _after_compare_load(self):
        n = self.n_peaks_var.get()
        for test in self.tests:
            for ch in test["channels"]:
                ch["pidx"] = top_peaks(ch["frq"], ch["amp"], n)
        names = ", ".join(t["name"] for t in self.tests) if self.tests else "No tests loaded"
        self.compare_lbl.config(text=f"{len(self.tests)} tests: {names}" if self.tests else names)
        self._rebuild_legend_compare()
        self._plot_compare()

    def _rebuild_legend_compare(self):
        for w in self.legend_frame.winfo_children():
            w.destroy()
        tk.Label(self.legend_frame, text="  Tests:  ", bg=BG_PANEL,
                 fg=FG_DIM, font=("Segoe UI", 8, "bold")).pack(side="left")
        for i, test in enumerate(self.tests):
            st = TEST_STYLES[i % len(TEST_STYLES)]
            tk.Label(self.legend_frame, text=f"— {test['name']}",
                     bg=BG_PANEL, fg=st["color"],
                     font=("Segoe UI", 9, "bold")).pack(side="left", padx=8)

    # ── replot dispatcher ─────────────────────────────────────────────────────
    def _replot(self):
        if self.mode.get() == "single":
            if self.single_test:
                n = self.n_peaks_var.get()
                for ch in self.single_test["channels"]:
                    ch["pidx"] = top_peaks(ch["frq"], ch["amp"], n)
            self._plot_single()
        elif self.mode.get() == "compare":
            n = self.n_peaks_var.get()
            for test in self.tests:
                for ch in test["channels"]:
                    ch["pidx"] = top_peaks(ch["frq"], ch["amp"], n)
            self._plot_compare()
        else:
            if self.overlay_test:
                n = self.n_peaks_var.get()
                for ch in self.overlay_test["channels"]:
                    ch["pidx"] = top_peaks(ch["frq"], ch["amp"], n)
            self._plot_overlay()

    # ── single-test plot ──────────────────────────────────────────────────────
    def _plot_single(self):
        for ax in self.ax_time + self.ax_fft:
            ax.clear(); style_ax(ax)

        if not self.single_test:
            for ax in self.ax_time:
                ax.set_title("Select a test folder", color="#444444", fontsize=9)
            self.canvas.draw()
            return

        channels = self.single_test["channels"]
        for s_idx, ch in enumerate(channels[:3]):
            color = SENSOR_COLORS[s_idx % len(SENSOR_COLORS)]
            fs    = float(ch["meta"].get("Sampling rate", "250") or "250")
            unit  = ch["meta"].get("Unit for accelerometer", "g")
            time  = ch["df"]["timestamp"].to_numpy() / fs
            vals  = ch["df"]["value"].to_numpy()

            # time domain
            ax_t = self.ax_time[s_idx]
            ax_t.plot(time, vals, color=color, linewidth=0.5, alpha=0.85)
            ax_t.set_title(ch["label"].upper(), color=color,
                           fontsize=9, fontweight="bold")
            ax_t.set_xlabel("Time (s)", color="#444444", fontsize=7)
            ax_t.set_ylabel(f"Acc ({unit})", color="#444444", fontsize=7)
            vmin, vmax = vals.min(), vals.max()
            ax_t.annotate(f"min {vmin:.4f}  max {vmax:.4f}",
                          xy=(0.02, 0.97), xycoords="axes fraction",
                          color=color, fontsize=7, va="top",
                          bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))

            # FFT
            ax_f = self.ax_fft[s_idx]
            ax_f.plot(ch["frq"], ch["amp"], color=color, linewidth=0.7, alpha=0.85)
            pidx = ch["pidx"] if ch["pidx"] is not None else []
            for rank, idx in enumerate(pidx):
                f, a = ch["frq"][idx], ch["amp"][idx]
                ax_f.plot(f, a, "o", color=color, markersize=4, zorder=5)
                ax_f.axvline(f, color=color, linewidth=0.6, linestyle="--", alpha=0.4)
                ax_f.annotate(f"f{rank+1}={f:.2f} Hz",
                              xy=(f, a), xytext=(4, 3), textcoords="offset points",
                              color=color, fontsize=7,
                              bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85))
            ax_f.set_xlabel("Frequency (Hz)", color="#444444", fontsize=7)
            ax_f.set_ylabel(f"Amplitude ({unit})", color="#444444", fontsize=7)
            ax_f.set_title(f"{ch['label'].upper()} — FFT", color=color,
                           fontsize=9, fontweight="bold")

        self.fig.suptitle(f"Single Test: {self.single_test['name']}",
                          color="#111111", fontsize=11, y=0.98)
        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.08)
        self.canvas.draw()
        self._attach_hover()

    # ── compare plot ──────────────────────────────────────────────────────────
    def _plot_compare(self):
        for ax in self.ax_fft:
            ax.clear(); style_ax(ax)
        self.ax_tbl.clear(); self.ax_tbl.axis("off")

        if not self.tests:
            for ax in self.ax_fft:
                ax.set_title("Load tests to begin", color="#444444", fontsize=9)
            self.canvas.draw()
            return

        sensors = [ch["label"] for ch in self.tests[0]["channels"]]
        unit    = ""

        for s_idx, sensor in enumerate(sensors[:3]):
            ax = self.ax_fft[s_idx]
            for t_idx, test in enumerate(self.tests):
                if s_idx >= len(test["channels"]):
                    continue
                ch = test["channels"][s_idx]
                st = TEST_STYLES[t_idx % len(TEST_STYLES)]
                if not unit:
                    unit = ch["meta"].get("Unit for accelerometer", "g")
                ax.plot(ch["frq"], ch["amp"],
                        color=st["color"], linewidth=0.8, linestyle=st["ls"],
                        alpha=0.8, label=test["name"])
                pidx = ch["pidx"] if ch["pidx"] is not None else []
                for idx in pidx:
                    ax.plot(ch["frq"][idx], ch["amp"][idx],
                            st["marker"], color=st["color"], markersize=4, zorder=5)
                    ax.axvline(ch["frq"][idx], color=st["color"],
                               linewidth=0.5, linestyle="--", alpha=0.3)
            ax.set_title(sensor.upper(),
                         color=SENSOR_COLORS[s_idx % len(SENSOR_COLORS)],
                         fontsize=10, fontweight="bold")
            ax.set_xlabel("Frequency (Hz)", color="#444444", fontsize=8)
            ax.set_ylabel(f"Amplitude ({unit})", color="#444444", fontsize=8)
            ax.legend(fontsize=7.5, facecolor="white",
                      edgecolor="#aaaaaa", labelcolor="#111111",
                      loc="upper right", framealpha=0.9)

        self._draw_table(sensors, unit)
        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.08)
        self.canvas.draw()
        self._attach_hover()

    # ── overlay plot (Compare 3 Accelerometers) ───────────────────────────────
    def _plot_overlay(self):
        self.ax_ov.clear(); style_ax(self.ax_ov)
        self.ax_tbl.clear(); self.ax_tbl.axis("off")

        if not self.overlay_test:
            self.ax_ov.set_title("Select a test folder", color="#444444", fontsize=9)
            self.canvas.draw()
            return

        channels = self.overlay_test["channels"]
        unit = ""
        n_peaks = self.n_peaks_var.get()

        for i, ch in enumerate(channels[:3]):
            color = SENSOR_COLORS[i % len(SENSOR_COLORS)]
            unit  = ch["meta"].get("Unit for accelerometer", "g")
            frq   = ch["frq"]
            amp   = ch["amp"]

            self.ax_ov.plot(frq, amp, color=color, linewidth=0.9,
                            alpha=0.9, label=ch["label"].upper())

            pidx = ch["pidx"] if ch["pidx"] is not None else []
            for rank, idx in enumerate(pidx):
                f, a = frq[idx], amp[idx]
                self.ax_ov.plot(f, a, "o", color=color, markersize=5, zorder=5)
                self.ax_ov.axvline(f, color=color, linewidth=0.6,
                                   linestyle="--", alpha=0.35)
                self.ax_ov.annotate(
                    f"{ch['label'].upper()} f{rank+1}={f:.2f} Hz",
                    xy=(f, a), xytext=(4, 3), textcoords="offset points",
                    color=color, fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85),
                )

        self.ax_ov.set_xlabel("Frequency (Hz)", color="#444444", fontsize=8)
        self.ax_ov.set_ylabel(f"Amplitude ({unit})", color="#444444", fontsize=8)
        self.ax_ov.set_title(
            f"{self.overlay_test['name']} — All Accelerometers FFT",
            color="#111111", fontsize=10, fontweight="bold",
        )
        self.ax_ov.legend(fontsize=9, facecolor=BG_PANEL,
                          edgecolor="#3a3a5e", labelcolor="#111111",
                          loc="upper right", framealpha=0.9)

        # peak table below the plot
        col_labels = ["Rank"]
        for ch in channels[:3]:
            col_labels += [ch["label"].upper(), f"{ch['label'].upper()} ζ"]
        rows = []
        for rank in range(n_peaks):
            row = [f"f{rank+1}"]
            for ch in channels[:3]:
                pidx = ch["pidx"] if ch["pidx"] is not None else []
                row.append(f"{ch['frq'][pidx[rank]]:.3f} Hz"
                            if rank < len(pidx) else "—")
                row.append(peak_zeta(ch, rank))
            rows.append(row)

        tbl = self.ax_tbl.table(
            cellText=rows, colLabels=col_labels,
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor("white" if r > 0 else "#f0f0f0")
            cell.set_text_props(color="#111111")
            cell.set_edgecolor("#cccccc")

        self.fig.subplots_adjust(left=0.07, right=0.97, top=0.95, bottom=0.05)
        self.canvas.draw()
        self._attach_hover()

    def _draw_table(self, sensors, unit):
        ax = self.ax_tbl
        ax.clear(); ax.axis("off")
        if not self.tests:
            return

        n_peaks = self.n_peaks_var.get()
        n_tests = len(self.tests)
        n_sens  = min(len(sensors), 3)

        col_labels = ["Rank"]
        for s_idx in range(n_sens):
            for test in self.tests:
                col_labels.append(f"{sensors[s_idx].upper()}\n{test['name']}\n(Hz)")
                col_labels.append("ζ (%)")

        def get_hz(t_idx, s_idx, rank):
            if t_idx >= len(self.tests):
                return "—"
            chs = self.tests[t_idx]["channels"]
            if s_idx >= len(chs):
                return "—"
            pidx = chs[s_idx]["pidx"] if chs[s_idx]["pidx"] is not None else []
            if rank >= len(pidx):
                return "—"
            return f"{chs[s_idx]['frq'][pidx[rank]]:.3f}"

        def get_zeta(t_idx, s_idx, rank):
            chs = self.tests[t_idx]["channels"]
            if s_idx >= len(chs):
                return "—"
            return peak_zeta(chs[s_idx], rank).replace(" %", "")

        rows = []
        for rank in range(n_peaks):
            row = [f"f{rank+1}"]
            for s_idx in range(n_sens):
                for t_idx in range(n_tests):
                    row.append(get_hz(t_idx, s_idx, rank))
                    row.append(get_zeta(t_idx, s_idx, rank))
            rows.append(row)

        tbl = ax.table(cellText=rows, colLabels=col_labels,
                       cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)

        n_cols = len(col_labels)
        for j in range(n_cols):
            cell = tbl[0, j]
            if j == 0:
                fc, tc = "#e0e0e0", "#333333"
            else:
                t_idx  = (j - 1) // 2 % n_tests
                st     = TEST_STYLES[t_idx % len(TEST_STYLES)]
                fc, tc = st["color"], "white"
            cell.set_facecolor(fc)
            cell.set_text_props(color=tc, fontweight="bold")
            cell.set_edgecolor("#cccccc")

        for r in range(len(rows)):
            for j in range(n_cols):
                cell = tbl[r + 1, j]
                cell.set_facecolor("#f7f7f7" if r % 2 == 0 else "white")
                tc = "#444444" if j == 0 else TEST_STYLES[(j-1) // 2 % n_tests % len(TEST_STYLES)]["color"]
                cell.set_text_props(color=tc)
                cell.set_edgecolor("#cccccc")

        ax.set_title("Natural Frequency & Damping Comparison — all tests  (Hz / ζ %, ranked by amplitude)",
                     color="#111111", fontsize=9, pad=4)


    # ── damping window (free-decay / envelope method) ─────────────────────────
    def _show_damping(self):
        """Damping per accelerometer following the envelope method: band-pass
        around f0, envelope of maxima after each excitation, least-squares
        fit of ln(envelope) vs time, zeta = -slope / (2*pi*f0)."""
        mode = self.mode.get()
        if mode == "single":
            tests = [self.single_test] if self.single_test else []
        elif mode == "overlay":
            tests = [self.overlay_test] if self.overlay_test else []
        else:
            tests = self.tests
        if not tests:
            messagebox.showinfo("Damping", "Load a test first.")
            return

        win = tk.Toplevel(self)
        win.title("Damping — envelope decay method")
        win.geometry("1250x860")
        win.configure(bg=BG_DARK)

        # ── controls ────────────────────────────────────────────────────────
        ctl = tk.Frame(win, bg=BG_DARK)
        ctl.pack(fill="x", padx=10, pady=(8, 4))

        def lbl(text):
            tk.Label(ctl, text=text, bg=BG_DARK, fg=FG_DIM,
                     font=("Segoe UI", 9)).pack(side="left", padx=(8, 2))

        test_var = tk.StringVar(value=tests[0]["name"])
        f0_var   = tk.StringVar()
        band_var = tk.DoubleVar(value=DAMP_DEFAULTS["band"])
        win_var  = tk.DoubleVar(value=DAMP_DEFAULTS["win_s"])
        nev_var  = tk.IntVar(value=DAMP_DEFAULTS["n_events"])

        lbl("Test:")
        test_cb = ttk.Combobox(ctl, textvariable=test_var, width=14, state="readonly",
                               values=[t["name"] for t in tests])
        test_cb.pack(side="left")
        lbl("f₀ (Hz):")
        f0_cb = ttk.Combobox(ctl, textvariable=f0_var, width=9)
        f0_cb.pack(side="left")
        lbl("Band ± (Hz):")
        tk.Spinbox(ctl, from_=0.1, to=5, increment=0.1, textvariable=band_var,
                   width=5).pack(side="left")
        lbl("Decay window (s):")
        tk.Spinbox(ctl, from_=1, to=30, increment=0.5, textvariable=win_var,
                   width=5).pack(side="left")
        lbl("Events / sensor:")
        tk.Spinbox(ctl, from_=1, to=10, textvariable=nev_var, width=4).pack(side="left")

        def test_by_name():
            return next(t for t in tests if t["name"] == test_var.get())

        def fill_f0_choices(_e=None):
            test = test_by_name()
            freqs = set()
            for ch in test["channels"]:
                pidx = top_peaks(ch["frq"], ch["amp"], self.n_peaks_var.get())
                freqs.update(round(float(ch["frq"][i]), 2) for i in pidx)
            f0_cb["values"] = [f"{f:.2f}" for f in sorted(freqs)]
            f0_var.set(f"{dominant_f0(test['channels']):.2f}")

        test_cb.bind("<<ComboboxSelected>>", lambda e: (fill_f0_choices(), calculate()))
        fill_f0_choices()

        for text, cmd, bg in (("Calculate",              lambda: calculate(),      "#9a3412"),
                              ("Include / exclude row",  lambda: toggle_row(),     "#374151"),
                              ("Copy table",             lambda: copy_table(),     "#0369a1")):
            tk.Button(ctl, text=text, command=cmd, bg=bg, fg="white", relief="flat",
                      padx=10, pady=3, font=("Segoe UI", 9, "bold"),
                      cursor="hand2").pack(side="left", padx=(8, 0))

        # ── results table + summary ─────────────────────────────────────────
        mid = tk.Frame(win, bg=BG_DARK)
        mid.pack(fill="x", padx=10)
        cols  = ("test", "chan", "t0", "units", "slope", "f0", "zeta", "r2", "use")
        heads = ("Test", "Channel", "Event t (s)", "Units", "Slope m (1/s)",
                 "f₀ (Hz)", "Damping ratio %", "R²", "Used")
        tree = ttk.Treeview(mid, columns=cols, show="headings", height=9)
        for c, h in zip(cols, heads):
            tree.heading(c, text=h)
            tree.column(c, width=120, anchor="center")
        tree.tag_configure("off", foreground="#aaaaaa")
        sb = ttk.Scrollbar(mid, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")

        summary = tk.Label(win, bg=BG_PANEL, fg=FG_MAIN, anchor="w",
                           font=("Segoe UI", 10, "bold"), padx=10, pady=4)
        summary.pack(fill="x", padx=10, pady=(4, 0))

        # ── plots: filtered signal + envelope, ln(envelope) regression ──────
        fig = plt.figure(figsize=(11, 4.2), facecolor="white")
        ax_sig = fig.add_subplot(121)
        ax_reg = fig.add_subplot(122)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=6)
        NavigationToolbar2Tk(canvas, win)
        win.protocol("WM_DELETE_WINDOW", lambda: (plt.close(fig), win.destroy()))

        rows = []    # (sensor index, channel, event dict)
        state = {"f0": None}

        def row_values(k):
            s_idx, ch, ev = rows[k]
            return (test_var.get(), ch["label"].upper(), f"{ev['t0']:.1f}",
                    ch["meta"].get("Unit for accelerometer", "g"),
                    f"{ev['slope']:.4f}", f"{state['f0']:.2f}",
                    f"{ev['zeta']:.3f}", f"{ev['r2']:.2f}",
                    "yes" if ev["include"] else "no")

        def update_summary():
            chans = test_by_name()["channels"]
            per, overall = damping_means(chans, rows)
            parts = [f"a{s_idx + 1} ({ch['label'].upper()}) = "
                     + (f"{z:.3f} %" if z is not None else "—")
                     for s_idx, (ch, z) in enumerate(zip(chans, per))]
            mean = f"{overall:.3f} %" if overall is not None else "—"
            summary.config(text="Damping ratio %:   " + "     ".join(parts)
                                + f"          Promedio = {mean}")

        def calculate():
            try:
                f0   = float(f0_var.get())
                band = float(band_var.get())
                wsec = float(win_var.get())
                nev  = int(nev_var.get())
            except (tk.TclError, ValueError):
                messagebox.showerror("Damping", "Invalid parameter value.", parent=win)
                return
            if band >= f0:
                messagebox.showerror("Damping", "Band must be smaller than f₀.", parent=win)
                return
            state["f0"] = f0
            test = test_by_name()
            rows[:] = run_damping(test["channels"], f0, band, wsec, nev)
            # remembered on the test so the Word report uses these results
            test["damping"] = dict(f0=f0, band=band, win_s=wsec, n_events=nev,
                                   rows=rows, reviewed=True)
            tree.delete(*tree.get_children())
            for k in range(len(rows)):
                tree.insert("", "end", iid=str(k), values=row_values(k))
            update_summary()
            if rows:
                tree.selection_set("0")
            else:
                ax_sig.clear(); ax_reg.clear()
                ax_sig.set_title("No decays found — try another f₀ or a wider band",
                                 fontsize=9)
                canvas.draw()

        def toggle_row():
            for iid in tree.selection():
                ev = rows[int(iid)][2]
                ev["include"] = not ev["include"]
                tree.item(iid, values=row_values(int(iid)),
                          tags=() if ev["include"] else ("off",))
            update_summary()

        def copy_table():
            lines = ["\t".join(heads)]
            lines += ["\t".join(row_values(k)) for k in range(len(rows))]
            lines.append(summary.cget("text"))
            win.clipboard_clear()
            win.clipboard_append("\n".join(lines))

        def on_select(_e=None):
            sel = tree.selection()
            if not sel:
                return
            s_idx, ch, ev = rows[int(sel[0])]
            test = test_by_name()
            plot_decay(ax_sig, ax_reg, ch, ev, state["f0"], test["damping"]["band"],
                       SENSOR_COLORS[s_idx % len(SENSOR_COLORS)])
            fig.tight_layout()
            canvas.draw()

        tree.bind("<<TreeviewSelect>>", on_select)
        calculate()

    # ── report generation ─────────────────────────────────────────────────────
    # ── hover tooltips ────────────────────────────────────────────────────────
    def _attach_hover(self):
        """Attach mplcursors hover tooltips to all active plot axes."""
        for c in self._cursors:
            try:
                c.remove()
            except Exception:
                pass
        self._cursors = []

        mode = self.mode.get()
        if mode == "single":
            time_set = set(id(ax) for ax in self.ax_time)
            all_axes = self.ax_time + self.ax_fft
        elif mode == "compare":
            time_set = set()
            all_axes = self.ax_fft
        else:
            time_set = set()
            all_axes = [self.ax_ov]

        lines = [ln for ax in all_axes for ln in ax.get_lines()
                 if ln.get_xdata() is not None and len(ln.get_xdata()) > 1]
        if not lines:
            return

        cursor = mplcursors.cursor(lines, hover=True)

        @cursor.connect("add")
        def on_add(sel):
            ax   = sel.artist.axes
            x, y = sel.target
            # extract unit from y-axis label  e.g. "Acc (g)" → "g"
            ylabel = ax.get_ylabel()
            unit   = ylabel.split("(")[-1].rstrip(")").strip() if "(" in ylabel else ""

            if id(ax) in time_set:
                text = f"t = {x:.3f} s\nacc = {y:.5f} {unit}"
            else:
                text  = f"f = {x:.4f} Hz\namp = {y:.6f} {unit}"
                label = sel.artist.get_label()
                if label and not label.startswith("_"):
                    text += f"\n[{label}]"

            sel.annotation.set_text(text)
            sel.annotation.set_fontsize(8)
            sel.annotation.set_color("#e0e0f0")
            bbox = sel.annotation.get_bbox_patch()
            bbox.set_facecolor("#1a1a2e")
            bbox.set_edgecolor("#7c3aed")
            bbox.set_alpha(0.92)

        self._cursors.append(cursor)

    def _generate_report(self):
        mode = self.mode.get()
        if mode == "single" and not self.single_test:
            messagebox.showwarning("No data", "Load a test first.")
            return
        if mode == "compare" and not self.tests:
            messagebox.showwarning("No data", "Load tests first.")
            return
        if mode == "overlay" and not self.overlay_test:
            messagebox.showwarning("No data", "Load a test first.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".docx",
            filetypes=[("Word document", "*.docx")],
            initialfile=f"report_{datetime.date.today()}.docx",
            title="Save report as…",
        )
        if not path:
            return

        try:
            if mode == "single":
                self._write_single_report(path)
            elif mode == "compare":
                self._write_compare_report(path)
            else:
                self._write_overlay_report(path)
            messagebox.showinfo("Report saved", f"Report saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not generate report:\n{e}")

    # ── shared docx helpers ───────────────────────────────────────────────────
    @staticmethod
    def _new_doc():
        doc = Document()
        # Page margins
        for section in doc.sections:
            section.top_margin    = Inches(1)
            section.bottom_margin = Inches(1)
            section.left_margin   = Inches(1)
            section.right_margin  = Inches(1)
        return doc

    @staticmethod
    def _heading(doc, text, level=1):
        p = doc.add_heading(text, level=level)
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = p.runs[0]
        run.font.color.rgb = RGBColor(0x1e, 0x40, 0xaf) if level == 1 else RGBColor(0x0f, 0x76, 0x6e)
        return p

    @staticmethod
    def _fig_to_buf(fig):
        """Render a matplotlib figure to an in-memory PNG buffer."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        return buf

    @staticmethod
    def _shade(cell, fill):
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), fill)
        shd.set(qn("w:val"),  "clear")
        cell._tc.get_or_add_tcPr().append(shd)

    def _simple_table(self, doc, headers, rows):
        """Blue-header table with alternating row fill, like the peak tables."""
        tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
        tbl.style = "Table Grid"
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        for j, h in enumerate(headers):
            cell = tbl.rows[0].cells[j]
            cell.text = h
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = cell.paragraphs[0].runs[0]
            run.font.bold = True
            run.font.color.rgb = RGBColor(0xff, 0xff, 0xff)
            self._shade(cell, "1e3a8a")
        for i, row in enumerate(rows, 1):
            fill = "e8f4fd" if i % 2 else "ffffff"
            for j, text in enumerate(row):
                cell = tbl.rows[i].cells[j]
                cell.text = text
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                self._shade(cell, fill)
        return tbl

    def _add_damping_section(self, doc, tests):
        """Damping table in the layout of damping_amortiguamiento_guide:
        per-event rows, then a1/a2/a3 per-sensor means and the overall mean."""
        doc.add_paragraph()
        self._heading(doc, "Damping — Envelope Decay Method", level=2)
        doc.add_paragraph(
            "For each accelerometer the largest excitations are selected, the signal "
            "is band-pass filtered around the frequency of interest f₀ and the "
            "envelope of maxima is taken over the following seconds. The envelope is "
            "plotted on a logarithmic scale against time, a least-squares line is "
            "fitted, and the fraction of critical damping is ζ = −m / (2π·f₀) · 100 %, "
            "where m is the slope of the fit.")

        summary = []
        for test in tests:
            chans = test["channels"]
            d = test.get("damping")
            if d is None:
                f0 = dominant_f0(chans)
                d = dict(f0=f0, rows=run_damping(chans, f0, **DAMP_DEFAULTS),
                         reviewed=False, **DAMP_DEFAULTS)
            used = [(si, ch, ev) for si, ch, ev in d["rows"] if ev["include"]]
            n_off = len(d["rows"]) - len(used)

            if len(tests) > 1:
                self._heading(doc, test["name"], level=3)
            note = (f"f₀ = {d['f0']:.2f} Hz, band ± {d['band']:g} Hz, decay window "
                    f"{d['win_s']:g} s, up to {d['n_events']} events per sensor.")
            if n_off:
                note += f" {n_off} event(s) excluded after review."
            if not d["reviewed"]:
                note += " Default parameters — not reviewed in the Damping window."
            doc.add_paragraph(note)

            if not used:
                doc.add_paragraph("No decays found.")
            else:
                self._simple_table(
                    doc,
                    ["TestName", "ChanTitle", "Units", "SlopeM", "Omega0",
                     "DampingRatio%", "R²"],
                    [[test["name"], ch["label"].upper(),
                      ch["meta"].get("Unit for accelerometer", "g"),
                      f"{ev['slope']:.4f}", f"{d['f0']:.2f}",
                      f"{ev['zeta']:.3f}", f"{ev['r2']:.2f}"]
                     for _, ch, ev in used])

                # example plots: the best-fitting event, as in the guide
                si, ch, ev = max(used, key=lambda r: r[2]["r2"])
                fig = plt.figure(figsize=(11, 4), facecolor="white")
                plot_decay(fig.add_subplot(121), fig.add_subplot(122), ch, ev,
                           d["f0"], d["band"], SENSOR_COLORS[si % len(SENSOR_COLORS)])
                fig.tight_layout()
                doc.add_paragraph()
                doc.add_picture(self._fig_to_buf(fig), width=Inches(6.3))
                plt.close(fig)
                doc.add_paragraph(f"Best-fitting event: {ch['label'].upper()} "
                                  f"at t = {ev['t0']:.1f} s (R² = {ev['r2']:.2f}).")

            per, overall = damping_means(chans, d["rows"])
            summary.append((test["name"], per, overall))

        n_sens = max(len(per) for _, per, _ in summary)
        labels = [ch["label"].upper() for ch in tests[0]["channels"]]
        fmt = lambda z: f"{z:.3f}" if z is not None else "—"
        self._heading(doc, "Damping ratio % — mean per accelerometer", level=3)
        self._simple_table(
            doc,
            ["Test"] + [f"a{i + 1} ({labels[i]})" if i < len(labels) else f"a{i + 1}"
                        for i in range(n_sens)] + ["Promedio"],
            [[name] + [fmt(per[i]) if i < len(per) else "—" for i in range(n_sens)]
             + [fmt(overall)] for name, per, overall in summary])

    @staticmethod
    def _add_peak_table(doc, channels, n_peaks):
        """Add a peak-frequency table to the document."""
        col_headers = ["Rank"]
        for ch in channels:
            col_headers += [ch["label"].upper(), f"{ch['label'].upper()} ζ"]
        table = doc.add_table(rows=1 + n_peaks, cols=len(col_headers))
        table.style = "Table Grid"
        table.alignment = WD_TABLE_ALIGNMENT.CENTER

        # header row
        hdr = table.rows[0].cells
        for j, h in enumerate(col_headers):
            hdr[j].text = h
            run = hdr[j].paragraphs[0].runs[0]
            run.font.bold = True
            run.font.color.rgb = RGBColor(0xff, 0xff, 0xff)
            hdr[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            # blue fill
            from docx.oxml.ns import qn
            from docx.oxml import OxmlElement
            tc   = hdr[j]._tc
            tcPr = tc.get_or_add_tcPr()
            shd  = OxmlElement("w:shd")
            shd.set(qn("w:fill"), "1e3a8a")
            shd.set(qn("w:color"), "auto")
            shd.set(qn("w:val"),   "clear")
            tcPr.append(shd)

        for rank in range(n_peaks):
            row_cells = table.rows[rank + 1].cells
            row_cells[0].text = f"f{rank + 1}"
            row_cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            fill = "e8f4fd" if rank % 2 == 0 else "ffffff"
            texts = []
            for ch in channels:
                pidx = ch["pidx"] if ch["pidx"] is not None else []
                texts.append(f"{ch['frq'][pidx[rank]]:.3f} Hz" if rank < len(pidx) else "—")
                texts.append(peak_zeta(ch, rank))
            for j, text in enumerate(texts, 1):
                row_cells[j].text = text
                row_cells[j].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                # alternating row colour
                from docx.oxml.ns import qn
                from docx.oxml import OxmlElement
                tc   = row_cells[j]._tc
                tcPr = tc.get_or_add_tcPr()
                shd  = OxmlElement("w:shd")
                shd.set(qn("w:fill"), fill)
                shd.set(qn("w:val"),  "clear")
                tcPr.append(shd)

    # ── single-test report ────────────────────────────────────────────────────
    def _write_single_report(self, path):
        doc  = self._new_doc()
        test = self.single_test
        n    = self.n_peaks_var.get()

        # Title
        self._heading(doc, "Accelerometer Analysis Report")
        doc.add_paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M')}")
        doc.add_paragraph(f"Test: {test['name']}")
        doc.add_paragraph()

        # Metadata table
        self._heading(doc, "Sensor Information", level=2)
        meta = test["channels"][0]["meta"]
        info = [
            ("Device",       meta.get("BeanDevice", "—")),
            ("Firmware",     meta.get("Firmware version", "—")),
            ("Sampling rate",f"{float(meta.get('Sampling rate','250')):.0f} Hz"),
            ("Range",        meta.get("Range(g)", "—")),
            ("Date",         meta.get("Date", "—")),
        ]
        tbl = doc.add_table(rows=len(info), cols=2)
        tbl.style = "Table Grid"
        for i, (k, v) in enumerate(info):
            tbl.rows[i].cells[0].text = k
            tbl.rows[i].cells[1].text = v
            tbl.rows[i].cells[0].paragraphs[0].runs[0].font.bold = True
        doc.add_paragraph()

        # Channel summary
        self._heading(doc, "Channel Summary", level=2)
        ch_hdr = ["Channel", "Samples", "Duration (s)", "Fs (Hz)"]
        cht = doc.add_table(rows=1 + len(test["channels"]), cols=4)
        cht.style = "Table Grid"
        for j, h in enumerate(ch_hdr):
            cht.rows[0].cells[j].text = h
            cht.rows[0].cells[j].paragraphs[0].runs[0].font.bold = True
        for i, ch in enumerate(test["channels"], 1):
            fs = ch["meta"].get("Sampling rate", "250")
            n_s = len(ch["df"])
            cht.rows[i].cells[0].text = ch["label"].upper()
            cht.rows[i].cells[1].text = f"{n_s:,}"
            cht.rows[i].cells[2].text = f"{n_s / float(fs):.1f}"
            cht.rows[i].cells[3].text = f"{float(fs):.0f}"
        doc.add_paragraph()

        # Time-domain plot
        self._heading(doc, "Time-Domain Signals", level=2)
        fig_t, axes = plt.subplots(1, len(test["channels"]),
                                   figsize=(12, 3), facecolor="white")
        if len(test["channels"]) == 1:
            axes = [axes]
        for ax, ch in zip(axes, test["channels"]):
            fs   = float(ch["meta"].get("Sampling rate", "250"))
            time = ch["df"]["timestamp"].to_numpy() / fs
            vals = ch["df"]["value"].to_numpy()
            step = max(1, len(time) // 8000)
            ax.plot(time[::step], vals[::step], linewidth=0.6, color="#1e40af")
            ax.set_title(ch["label"].upper(), fontsize=9)
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel(f"Acc ({ch['meta'].get('Unit for accelerometer','g')})", fontsize=8)
            ax.grid(True, linewidth=0.4, alpha=0.5)
        fig_t.tight_layout()
        doc.add_picture(self._fig_to_buf(fig_t), width=Inches(6))
        plt.close(fig_t)
        doc.add_paragraph()

        # FFT plot
        self._heading(doc, "FFT — Frequency Spectrum", level=2)
        fig_f, axes = plt.subplots(1, len(test["channels"]),
                                   figsize=(12, 4), facecolor="white")
        if len(test["channels"]) == 1:
            axes = [axes]
        for ax, ch in zip(axes, test["channels"]):
            frq  = np.array(ch["frq"])
            amp  = np.array(ch["amp"])
            pidx = ch["pidx"] if ch["pidx"] is not None else []
            ax.plot(frq, amp, linewidth=0.7, color="#1e40af")
            for rank, idx in enumerate(pidx):
                ax.plot(frq[idx], amp[idx], "ro", markersize=5)
                ax.axvline(frq[idx], color="red", linewidth=0.6, linestyle="--", alpha=0.5)
                ax.annotate(f"f{rank+1}={frq[idx]:.2f} Hz",
                            xy=(frq[idx], amp[idx]), xytext=(4, 4),
                            textcoords="offset points", fontsize=7, color="red")
            ax.set_title(ch["label"].upper(), fontsize=9)
            ax.set_xlabel("Frequency (Hz)", fontsize=8)
            ax.set_ylabel(f"Amplitude ({ch['meta'].get('Unit for accelerometer','g')})", fontsize=8)
            ax.grid(True, linewidth=0.4, alpha=0.5)
        fig_f.tight_layout()
        doc.add_picture(self._fig_to_buf(fig_f), width=Inches(6))
        plt.close(fig_f)
        doc.add_paragraph()

        # Peak table
        self._heading(doc, "Identified Natural Frequencies", level=2)
        self._add_peak_table(doc, test["channels"], n)

        self._add_damping_section(doc, [test])
        doc.save(path)

    # ── compare report ────────────────────────────────────────────────────────
    def _write_compare_report(self, path):
        doc    = self._new_doc()
        n      = self.n_peaks_var.get()
        tests  = self.tests
        sensors = [ch["label"] for ch in tests[0]["channels"]]

        # Title
        self._heading(doc, "Accelerometer Comparison Report")
        doc.add_paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M')}")
        doc.add_paragraph(f"Tests compared: {', '.join(t['name'] for t in tests)}")
        doc.add_paragraph()

        # Per-sensor sections
        colors_plt = ["#1e40af", "#0f766e", "#9f1239", "#92400e",
                      "#4d7c0f", "#6d28d9", "#0e7490", "#b45309"]

        for s_idx, sensor in enumerate(sensors[:3]):
            self._heading(doc, f"Sensor: {sensor.upper()}", level=2)

            # FFT comparison plot for this sensor
            fig, ax = plt.subplots(figsize=(11, 4), facecolor="white")
            for t_idx, test in enumerate(tests):
                if s_idx >= len(test["channels"]):
                    continue
                ch   = test["channels"][s_idx]
                frq  = np.array(ch["frq"])
                amp  = np.array(ch["amp"])
                pidx = ch["pidx"] if ch["pidx"] is not None else []
                c    = colors_plt[t_idx % len(colors_plt)]
                ls   = ["-", "--", ":", "-."][t_idx % 4]
                ax.plot(frq, amp, linewidth=0.9, color=c, linestyle=ls,
                        label=test["name"])
                for idx in pidx:
                    ax.plot(frq[idx], amp[idx], "o", color=c, markersize=5)
                    ax.axvline(frq[idx], color=c, linewidth=0.5,
                               linestyle="--", alpha=0.35)

            unit = tests[0]["channels"][s_idx]["meta"].get("Unit for accelerometer", "g")
            ax.set_xlabel("Frequency (Hz)", fontsize=9)
            ax.set_ylabel(f"Amplitude ({unit})", fontsize=9)
            ax.set_title(f"{sensor.upper()} — FFT Comparison  "
                         f"(Hann window, DC removed)", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, linewidth=0.4, alpha=0.5)
            fig.tight_layout()
            doc.add_picture(self._fig_to_buf(fig), width=Inches(6.2))
            plt.close(fig)
            doc.add_paragraph()

            # Peak table for this sensor (one column per test)
            self._heading(doc, f"Peak Frequencies — {sensor.upper()}", level=3)
            col_headers = ["Rank"]
            for t in tests:
                col_headers += [t["name"], f"{t['name']} ζ"]
            tbl = doc.add_table(rows=1 + n, cols=len(col_headers))
            tbl.style = "Table Grid"

            # header
            from docx.oxml.ns import qn
            from docx.oxml import OxmlElement
            for j, h in enumerate(col_headers):
                cell = tbl.rows[0].cells[j]
                cell.text = h
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = cell.paragraphs[0].runs[0]
                run.font.bold = True
                run.font.color.rgb = RGBColor(0xff, 0xff, 0xff)
                tc   = cell._tc
                tcPr = tc.get_or_add_tcPr()
                shd  = OxmlElement("w:shd")
                shd.set(qn("w:fill"), "1e3a8a")
                shd.set(qn("w:val"),  "clear")
                tcPr.append(shd)

            # data rows
            for rank in range(n):
                row_cells = tbl.rows[rank + 1].cells
                row_cells[0].text = f"f{rank + 1}"
                row_cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                fill = "e8f4fd" if rank % 2 == 0 else "ffffff"
                texts = []
                for test in tests:
                    if s_idx < len(test["channels"]):
                        ch   = test["channels"][s_idx]
                        frq  = np.array(ch["frq"])
                        pidx = ch["pidx"] if ch["pidx"] is not None else []
                        texts.append(f"{frq[pidx[rank]]:.3f} Hz"
                                     if rank < len(pidx) else "—")
                        texts.append(peak_zeta(ch, rank))
                    else:
                        texts += ["—", "—"]
                for j, text in enumerate(texts, 1):
                    cell = row_cells[j]
                    cell.text = text
                    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    tc   = cell._tc
                    tcPr = tc.get_or_add_tcPr()
                    shd  = OxmlElement("w:shd")
                    shd.set(qn("w:fill"), fill)
                    shd.set(qn("w:val"),  "clear")
                    tcPr.append(shd)

            doc.add_paragraph()

        self._add_damping_section(doc, tests)
        doc.save(path)

    # ── overlay report (Compare 3 Accelerometers) ─────────────────────────────
    def _write_overlay_report(self, path):
        doc  = self._new_doc()
        test = self.overlay_test
        n    = self.n_peaks_var.get()
        channels = test["channels"]

        # Title
        self._heading(doc, "3-Accelerometer Comparison Report")
        doc.add_paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M')}")
        doc.add_paragraph(f"Test: {test['name']}")
        doc.add_paragraph()

        # Sensor info table
        self._heading(doc, "Sensor Information", level=2)
        meta = channels[0]["meta"]
        info = [
            ("Device",        meta.get("BeanDevice", "—")),
            ("Firmware",      meta.get("Firmware version", "—")),
            ("Sampling rate", f"{float(meta.get('Sampling rate', '250')):.0f} Hz"),
            ("Range",         meta.get("Range(g)", "—")),
            ("Date",          meta.get("Date", "—")),
        ]
        tbl = doc.add_table(rows=len(info), cols=2)
        tbl.style = "Table Grid"
        for i, (k, v) in enumerate(info):
            tbl.rows[i].cells[0].text = k
            tbl.rows[i].cells[1].text = v
            tbl.rows[i].cells[0].paragraphs[0].runs[0].font.bold = True
        doc.add_paragraph()

        # Channel summary
        self._heading(doc, "Channel Summary", level=2)
        ch_hdr = ["Channel", "Samples", "Duration (s)", "Fs (Hz)"]
        cht = doc.add_table(rows=1 + len(channels), cols=4)
        cht.style = "Table Grid"
        for j, h in enumerate(ch_hdr):
            cht.rows[0].cells[j].text = h
            cht.rows[0].cells[j].paragraphs[0].runs[0].font.bold = True
        for i, ch in enumerate(channels, 1):
            fs  = float(ch["meta"].get("Sampling rate", "250"))
            n_s = len(ch["df"])
            cht.rows[i].cells[0].text = ch["label"].upper()
            cht.rows[i].cells[1].text = f"{n_s:,}"
            cht.rows[i].cells[2].text = f"{n_s / fs:.1f}"
            cht.rows[i].cells[3].text = f"{fs:.0f}"
        doc.add_paragraph()

        # Overlay FFT plot
        self._heading(doc, "Frequency Spectrum — All Accelerometers", level=2)
        fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
        colors = ["#f97316", "#0ea5e9", "#7c3aed"]
        for i, ch in enumerate(channels):
            color = colors[i % len(colors)]
            frq   = np.array(ch["frq"])
            amp   = np.array(ch["amp"])
            unit  = ch["meta"].get("Unit for accelerometer", "g")
            ax.plot(frq, amp, color=color, linewidth=0.9,
                    label=ch["label"].upper())
            pidx = ch["pidx"] if ch["pidx"] is not None else []
            for rank, idx in enumerate(pidx):
                ax.plot(frq[idx], amp[idx], "o", color=color, markersize=5)
                ax.axvline(frq[idx], color=color, linewidth=0.6,
                           linestyle="--", alpha=0.4)
                ax.annotate(
                    f"{ch['label'].upper()} f{rank+1}={frq[idx]:.2f} Hz",
                    xy=(frq[idx], amp[idx]), xytext=(4, 4),
                    textcoords="offset points", fontsize=7, color=color,
                )
        ax.set_xlabel("Frequency (Hz)", fontsize=9)
        ax.set_ylabel(f"Amplitude ({unit})", fontsize=9)
        ax.set_title(f"{test['name']} — FFT Overlay", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, linewidth=0.4, alpha=0.5)
        fig.tight_layout()
        doc.add_picture(self._fig_to_buf(fig), width=Inches(6))
        plt.close(fig)
        doc.add_paragraph()

        # Peak table
        self._heading(doc, "Identified Natural Frequencies", level=2)
        self._add_peak_table(doc, channels, n)

        self._add_damping_section(doc, [test])
        doc.save(path)


if __name__ == "__main__":
    app = App()
    app.mainloop()
