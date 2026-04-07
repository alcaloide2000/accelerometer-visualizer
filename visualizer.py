import os
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
import numpy as np
from scipy.signal import windows, find_peaks

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
    ax.set_facecolor(BG_MID)
    ax.tick_params(colors=FG_DIM, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#3a3a5e")
    ax.grid(True, color=GRID_COL, linewidth=0.5)


# ── app ───────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Accelerometer Visualizer")
        self.geometry("1400x940")
        self.configure(bg=BG_DARK)
        self.tests      = []   # for comparison mode
        self.single_test = None  # for single mode
        self.mode        = tk.StringVar(value="compare")
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── mode toggle bar ──────────────────────────────────────────────────
        mode_bar = tk.Frame(self, bg="#0f0f1a")
        mode_bar.pack(fill="x", padx=0, pady=0)

        for text, val in (("Single Test", "single"), ("Compare Tests", "compare")):
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

        # shared: peaks spinbox on the right of whichever bar is active
        for bar in (self.single_bar, self.compare_bar):
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
        if mode == "single":
            self.compare_bar.pack_forget()
            self.single_bar.pack(fill="x", padx=10, pady=6)
        else:
            self.single_bar.pack_forget()
            self.compare_bar.pack(fill="x", padx=10, pady=6)
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
            self.fig = plt.figure(figsize=(14, 8), facecolor=BG_DARK)
            gs = self.fig.add_gridspec(2, 3, height_ratios=[1, 1.4],
                                       hspace=0.45, wspace=0.32)
            self.ax_time = [self.fig.add_subplot(gs[0, i]) for i in range(3)]
            self.ax_fft  = [self.fig.add_subplot(gs[1, i]) for i in range(3)]
            for ax in self.ax_time + self.ax_fft:
                style_ax(ax)
        else:
            self.fig = plt.figure(figsize=(14, 8), facecolor=BG_DARK)
            gs = self.fig.add_gridspec(2, 3, height_ratios=[2, 1],
                                       hspace=0.45, wspace=0.32)
            self.ax_fft = [self.fig.add_subplot(gs[0, i]) for i in range(3)]
            self.ax_tbl =  self.fig.add_subplot(gs[1, :])
            self.ax_tbl.axis("off")
            for ax in self.ax_fft:
                style_ax(ax)

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
        else:
            n = self.n_peaks_var.get()
            for test in self.tests:
                for ch in test["channels"]:
                    ch["pidx"] = top_peaks(ch["frq"], ch["amp"], n)
            self._plot_compare()

    # ── single-test plot ──────────────────────────────────────────────────────
    def _plot_single(self):
        for ax in self.ax_time + self.ax_fft:
            ax.clear(); style_ax(ax)

        if not self.single_test:
            for ax in self.ax_time:
                ax.set_title("Select a test folder", color=FG_DIM, fontsize=9)
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
            ax_t.set_xlabel("Time (s)", color=FG_DIM, fontsize=7)
            ax_t.set_ylabel(f"Acc ({unit})", color=FG_DIM, fontsize=7)
            vmin, vmax = vals.min(), vals.max()
            ax_t.annotate(f"min {vmin:.4f}  max {vmax:.4f}",
                          xy=(0.02, 0.97), xycoords="axes fraction",
                          color=color, fontsize=7, va="top",
                          bbox=dict(boxstyle="round,pad=0.2", fc=BG_DARK, alpha=0.7))

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
                              bbox=dict(boxstyle="round,pad=0.2", fc=BG_DARK, alpha=0.7))
            ax_f.set_xlabel("Frequency (Hz)", color=FG_DIM, fontsize=7)
            ax_f.set_ylabel(f"Amplitude ({unit})", color=FG_DIM, fontsize=7)
            ax_f.set_title(f"{ch['label'].upper()} — FFT", color=color,
                           fontsize=9, fontweight="bold")

        self.fig.suptitle(f"Single Test: {self.single_test['name']}",
                          color=FG_MAIN, fontsize=11, y=1.01)
        self.fig.tight_layout(pad=1.8)
        self.canvas.draw()

    # ── compare plot ──────────────────────────────────────────────────────────
    def _plot_compare(self):
        for ax in self.ax_fft:
            ax.clear(); style_ax(ax)
        self.ax_tbl.clear(); self.ax_tbl.axis("off")

        if not self.tests:
            for ax in self.ax_fft:
                ax.set_title("Load tests to begin", color=FG_DIM, fontsize=9)
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
            ax.set_xlabel("Frequency (Hz)", color=FG_DIM, fontsize=8)
            ax.set_ylabel(f"Amplitude ({unit})", color=FG_DIM, fontsize=8)
            ax.legend(fontsize=7.5, facecolor=BG_PANEL,
                      edgecolor="#3a3a5e", labelcolor=FG_MAIN,
                      loc="upper right", framealpha=0.9)

        self._draw_table(sensors, unit)
        self.fig.tight_layout(pad=1.8)
        self.canvas.draw()

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
                fc, tc = BG_MID, FG_DIM
            else:
                t_idx  = (j - 1) % n_tests
                st     = TEST_STYLES[t_idx % len(TEST_STYLES)]
                fc, tc = st["color"], BG_DARK
            cell.set_facecolor(fc)
            cell.set_text_props(color=tc, fontweight="bold")
            cell.set_edgecolor("#3a3a5e")

        for r in range(len(rows)):
            for j in range(n_cols):
                cell = tbl[r + 1, j]
                cell.set_facecolor("#20203a" if r % 2 == 0 else BG_PANEL)
                tc = FG_DIM if j == 0 else TEST_STYLES[(j-1) % n_tests % len(TEST_STYLES)]["color"]
                cell.set_text_props(color=tc)
                cell.set_edgecolor("#3a3a5e")

        ax.set_title("Natural Frequency Comparison — all tests  (Hz, ranked by amplitude)",
                     color=FG_MAIN, fontsize=9, pad=4)


if __name__ == "__main__":
    app = App()
    app.mainloop()
