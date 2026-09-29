# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Accelerometer FFT visualizer for vibration tests recorded with BeanDevice AX3D sensors (up to 3 accelerometers per test). It computes a Hann-windowed FFT per channel, detects dominant frequency peaks, and compares them across tests/sensors. There are **two independent front-ends that duplicate the same processing logic** — no shared module:

- `visualizer.py` — Tkinter + matplotlib desktop app (with `mplcursors` hover and `.docx` report export via `python-docx`). Reads test folders from disk.
- `app.py` — Streamlit + Plotly web app, deployed on Render (`render.yaml`). Works from uploaded files, not the filesystem.

When changing parsing, FFT, or peak-detection behavior, update **both** files (`parse_file`/`compute_fft`/`top_peaks` in `visualizer.py` vs. `_parse_bytes`/`process_test`/`_peaks` in `app.py`).

## Commands

```bash
pip install -r requirements.txt          # web app deps only
python visualizer.py                     # desktop app (also needs matplotlib, mplcursors, python-docx — not in requirements.txt)
streamlit run app.py                     # web app locally
```

Build a standalone Windows exe of the desktop app (output `dist/AccelerometerVisualizer.exe`, ignored by git):

```bash
python -m PyInstaller --noconfirm --onefile --windowed --name AccelerometerVisualizer \
  --hidden-import scipy._cyutility \
  --exclude-module streamlit --exclude-module plotly --exclude-module IPython \
  --exclude-module PyQt5 --exclude-module PySide6 visualizer.py
```

`scipy._cyutility` must be listed explicitly (scipy ≥ 1.16; without it the exe dies on import with `ModuleNotFoundError`).

No tests or linter exist. `requirements.txt` is the Render deploy manifest, so don't add desktop-only dependencies to it.

## Data format

A test is a folder (e.g. `test1/`, untracked sample data) containing `acel1/`, `acel2/`, `acel3/` subfolders. Each channel folder has:
- one main `.txt` file: `key : value` header lines (metadata, incl. `Sampling rate`, default 250 Hz if missing), then a `Timestamp;Measure Value` line, then `timestamp;value` rows;
- optional `*_part001.txt`, `*_part002.txt`, … files that contain data rows only and are concatenated after the main file in sorted order.

The desktop app discovers channels by `acel*` subfolder names. The web app has no folders, so it groups uploaded files by `Acel_1`/`Acel_2`/`Acel_3` in the filename and detects part files by `_part` in the name. Timestamps are sample indices (time = timestamp / fs).

## Processing (shared semantics)

- FFT: subtract mean, apply Hann window, amplitude = `2/sum(window) * |rfft|`.
- Peaks: `scipy.signal.find_peaks` with prominence ≥ 1% of max amplitude and `distance=5`, keep top N by amplitude, returned sorted by frequency. N is user-selectable.
- Damping (ζ column in every peak table/report): half-power bandwidth on a Welch PSD (Hann, `nperseg=DAMP_NPERSEG`=4096, 50% overlap), not on the raw FFT — the raw FFT of these long ambient records is too noisy. For each FFT peak, take the PSD max within ±`DAMP_SEARCH_HZ` (0.25 Hz), find where PSD drops to ½, ζ = (f₂−f₁)/(2·f₀). Returns None ("—") if the band is < 2 PSD bins or runs off the spectrum. `compute_psd`/`damping_ratio`/`peak_zeta` in `visualizer.py` vs. PSD in `process_test` + `_damping`/`_peak_cells` in `app.py`.
- Damping window (desktop "〰 Damping" button → `_show_damping`, maths in `decay_damping`; web "〰 Damping" tab, maths in `_decay_damping`, cached via `run_damping`, exclusions via the `Used` column of `st.data_editor`): the free-decay envelope method from `damping_amortiguamiento_guide.pdf`. 4th-order Butterworth band-pass f₀ ± band (zero-phase `sosfiltfilt`); events = largest Hilbert-envelope maxima ≥ `DAMP_MIN_EXCITATION`× the median envelope, spaced ≥ 3 windows apart; positive maxima over the next `win_s` s (cut where they drop below 2× the median envelope) → least-squares fit of ln(maxima) vs t → ζ % = −m/(2π·f₀)·100. The window shows per-event rows, a1/a2/a3 per-sensor means and the overall "Promedio". Calculating in the window stores the results (incl. excluded rows) as `test["damping"]`; every Word report ends with `_add_damping_section`, which uses those results or, for unreviewed tests, runs `DAMP_DEFAULTS` at `dominant_f0` and says so.

## Views

Both apps expose the same analysis modes: **Single Test** (one test, its channels), **Compare Tests** (multiple tests, per-sensor comparison plus peak/mean-frequency tables), and **Compare 3 Accelerometers** (sensors of one test overlaid). `app.py` adds a fourth tab, "Sensors per Test", and a fifth, "Damping" (the desktop app has it as a button/window instead). In `visualizer.py`, the mode is `self.mode` (`single`/`compare`/`overlay`); each has `_load_*`, `_plot_*`, `_rebuild_legend_*` and `_write_*_report` methods, dispatched from `_replot` and `_generate_report`.

In `app.py`, `process_test` is `@st.cache_data`-cached and takes a tuple of `(name, bytes)` so it's hashable; time-domain plots are downsampled to `MAX_TIME_PTS` via `_ds`.
