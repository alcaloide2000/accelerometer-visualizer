import os
import io
import datetime
import tkinter as tk
import mplcursors
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import numpy as np
from scipy.signal import windows, find_peaks
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
        col_labels = ["Rank"] + [ch["label"].upper() for ch in channels[:3]]
        rows = []
        for rank in range(n_peaks):
            row = [f"f{rank+1}"]
            for ch in channels[:3]:
                pidx = ch["pidx"] if ch["pidx"] is not None else []
                row.append(f"{ch['frq'][pidx[rank]]:.3f} Hz"
                            if rank < len(pidx) else "—")
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

        rows = []
        for rank in range(n_peaks):
            row = [f"f{rank+1}"]
            for s_idx in range(n_sens):
                for t_idx in range(n_tests):
                    row.append(get_hz(t_idx, s_idx, rank))
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
                t_idx  = (j - 1) % n_tests
                st     = TEST_STYLES[t_idx % len(TEST_STYLES)]
                fc, tc = st["color"], "white"
            cell.set_facecolor(fc)
            cell.set_text_props(color=tc, fontweight="bold")
            cell.set_edgecolor("#cccccc")

        for r in range(len(rows)):
            for j in range(n_cols):
                cell = tbl[r + 1, j]
                cell.set_facecolor("#f7f7f7" if r % 2 == 0 else "white")
                tc = "#444444" if j == 0 else TEST_STYLES[(j-1) % n_tests % len(TEST_STYLES)]["color"]
                cell.set_text_props(color=tc)
                cell.set_edgecolor("#cccccc")

        ax.set_title("Natural Frequency Comparison — all tests  (Hz, ranked by amplitude)",
                     color="#111111", fontsize=9, pad=4)


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
    def _add_peak_table(doc, channels, n_peaks):
        """Add a peak-frequency table to the document."""
        col_headers = ["Rank"] + [ch["label"].upper() for ch in channels]
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

        frq_all = [np.array(ch["frq"]) for ch in channels]
        amp_all = [np.array(ch["amp"]) for ch in channels]
        pidx_all = [ch.get("pidx") if ch.get("pidx") is not None else [] for ch in channels]

        for rank in range(n_peaks):
            row_cells = table.rows[rank + 1].cells
            row_cells[0].text = f"f{rank + 1}"
            row_cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            fill = "e8f4fd" if rank % 2 == 0 else "ffffff"
            for j, (frq, pidx) in enumerate(zip(frq_all, pidx_all), 1):
                hz = f"{frq[pidx[rank]]:.3f} Hz" if rank < len(pidx) else "—"
                row_cells[j].text = hz
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
            col_headers = ["Rank"] + [t["name"] for t in tests]
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
                for j, test in enumerate(tests, 1):
                    if s_idx < len(test["channels"]):
                        ch   = test["channels"][s_idx]
                        frq  = np.array(ch["frq"])
                        pidx = ch["pidx"] if ch["pidx"] is not None else []
                        hz   = (f"{frq[pidx[rank]]:.3f} Hz"
                                if rank < len(pidx) else "—")
                    else:
                        hz = "—"
                    cell = row_cells[j]
                    cell.text = hz
                    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    tc   = cell._tc
                    tcPr = tc.get_or_add_tcPr()
                    shd  = OxmlElement("w:shd")
                    shd.set(qn("w:fill"), fill)
                    shd.set(qn("w:val"),  "clear")
                    tcPr.append(shd)

            doc.add_paragraph()

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

        doc.save(path)


if __name__ == "__main__":
    app = App()
    app.mainloop()
