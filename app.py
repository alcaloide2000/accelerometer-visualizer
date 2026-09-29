import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import windows, find_peaks, welch, butter, sosfiltfilt, hilbert
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── palette ───────────────────────────────────────────────────────────────────
SENSOR_COLORS = ["#f97316", "#22d3ee", "#a78bfa"]
TEST_COLORS   = ["#f97316", "#22d3ee", "#4ade80", "#f472b6", "#facc15", "#a78bfa"]
TEST_DASH     = ["solid",   "dash",    "dot",     "dashdot", "solid",   "dash"]
MAX_TIME_PTS  = 6000
BG            = "#12121c"
PLOT_BG       = "#1a1a2e"
GRID_COL      = "rgba(60,60,100,0.5)"
# damping: Welch PSD segment length (4096 @ 250 Hz ≈ 16 s, 0.06 Hz bins) and
# how far (Hz) from an FFT peak to look for the matching PSD peak
DAMP_NPERSEG   = 4096
DAMP_SEARCH_HZ = 0.25
# decay method: an event must peak at least this many times the median
# envelope (ambient level) to count as an excitation
DAMP_MIN_EXCITATION = 8.0


# ── data processing ───────────────────────────────────────────────────────────

def _parse_bytes(name, raw):
    """Parse a txt file (header or data-only) from raw bytes."""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    lines    = text.splitlines()
    metadata = {}
    rows     = []
    in_data  = "_part" in name.lower()   # part files start with data lines

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if not in_data and "Timestamp;Measure Value" in line:
            in_data = True
            continue
        if in_data:
            parts = line.split(";")
            if len(parts) == 2:
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
        elif ":" in line and not line.startswith("-"):
            key, _, val = line.partition(":")
            metadata[key.strip()] = val.strip()

    return metadata, pd.DataFrame(rows, columns=["timestamp", "value"])


def _group_by_channel(file_data):
    """
    Group (name, bytes) pairs by channel.
    Channel is detected from 'Acel_1', 'Acel_2', 'Acel_3' in the filename.
    Returns {channel_label: [(name, bytes), ...]}, main file first per group.
    """
    groups = {}
    for name, raw in file_data:
        upper = name.upper()
        for ch in ["ACEL_1", "ACEL_2", "ACEL_3"]:
            if ch in upper:
                key = ch.lower()
                groups.setdefault(key, []).append((name, raw))
                break
    for key in groups:
        # main file (no _part) first, then parts in order
        groups[key].sort(key=lambda x: (1 if "_part" in x[0].lower() else 0, x[0]))
    return dict(sorted(groups.items()))


@st.cache_data(show_spinner="Processing files…")
def process_test(file_data):
    """
    file_data: tuple of (name, bytes) — hashable for Streamlit cache.
    Returns list of channel dicts with FFT pre-computed.
    """
    groups   = _group_by_channel(list(file_data))
    channels = []

    for label, files in groups.items():
        meta_all, dfs = {}, []
        for name, raw in files:
            meta, df = _parse_bytes(name, raw)
            if meta:
                meta_all = meta
            dfs.append(df)

        df_all = pd.concat(dfs, ignore_index=True)
        fs     = float(meta_all.get("Sampling rate", "250") or "250")
        n      = len(df_all)
        vals   = df_all["value"].to_numpy()
        win    = windows.hann(n)
        sig    = (vals - vals.mean()) * win
        amp    = (2.0 / win.sum()) * np.abs(np.fft.rfft(sig))
        frq    = np.fft.rfftfreq(n, d=1.0 / fs)
        nseg   = min(DAMP_NPERSEG, n)
        psd_f, psd_p = welch(vals - vals.mean(), fs, window="hann",
                             nperseg=nseg, noverlap=nseg // 2)

        channels.append(dict(
            label    = label,
            meta     = meta_all,
            n_samples= n,
            fs       = fs,
            duration = n / fs,
            time     = (df_all["timestamp"].to_numpy() / fs).tolist(),
            values   = vals.tolist(),
            frq      = frq.tolist(),
            amp      = amp.tolist(),
            psd_f    = psd_f.tolist(),
            psd_p    = psd_p.tolist(),
        ))

    return channels


def _peaks(frq, amp, n):
    frq, amp = np.array(frq), np.array(amp)
    min_prom = amp.max() * 0.01
    idx, _   = find_peaks(amp, prominence=min_prom, distance=5)
    idx      = idx[np.argsort(amp[idx])[::-1]][:n]
    return np.sort(idx)


def _damping(psd_f, psd_p, f0):
    """Half-power bandwidth damping ratio (%) of the PSD peak nearest f0.

    Returns None when the peak can't be resolved (band runs off the search
    range or is narrower than 2 PSD bins)."""
    psd_f, psd_p = np.array(psd_f), np.array(psd_p)
    lo, hi = np.searchsorted(psd_f, [f0 - DAMP_SEARCH_HZ, f0 + DAMP_SEARCH_HZ])
    if hi <= lo or len(psd_f) < 2:
        return None
    i  = lo + int(np.argmax(psd_p[lo:hi]))
    hp = psd_p[i] / 2                     # half power
    if hp <= 0:
        return None
    l = i
    while l > 0 and psd_p[l] > hp:
        l -= 1
    r = i
    while r < len(psd_p) - 1 and psd_p[r] > hp:
        r += 1
    if psd_p[l] > hp or psd_p[r] > hp:
        return None
    f1 = np.interp(hp, [psd_p[l], psd_p[l + 1]], [psd_f[l], psd_f[l + 1]])
    f2 = np.interp(hp, [psd_p[r], psd_p[r - 1]], [psd_f[r], psd_f[r - 1]])
    if f2 - f1 < 2 * (psd_f[1] - psd_f[0]):
        return None
    return 100.0 * (f2 - f1) / (2.0 * f0)


def _peak_cells(ch, n):
    """[(freq_str, zeta_str), ...] for ranks 0..n-1 of a channel."""
    frq  = np.array(ch["frq"])
    pidx = _peaks(frq, ch["amp"], n)
    out  = []
    for rank in range(n):
        if rank >= len(pidx):
            out.append(("—", "—"))
            continue
        f0 = frq[pidx[rank]]
        z  = _damping(ch["psd_f"], ch["psd_p"], f0)
        out.append((f"{f0:.3f} Hz", f"{z:.2f} %" if z is not None else "—"))
    return out


def _decay_damping(values, fs, f0, band=0.5, win_s=5.0, n_events=3):
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
            t_mx=t_mx, a_mx=seg[mx],
        ))
    return events


def _dominant_f0(channels):
    """Median of each channel's dominant FFT peak — the default f0."""
    dom = [np.array(ch["frq"])[_peaks(ch["frq"], ch["amp"], 1)[0]] for ch in channels]
    return float(np.median(dom))


@st.cache_data(show_spinner="Computing damping…")
def run_damping(file_data, f0, band, win_s, n_events):
    """_decay_damping for every channel of a test → flat list of event dicts."""
    rows = []
    for s_idx, ch in enumerate(process_test(file_data)):
        for ev in _decay_damping(np.array(ch["values"]), ch["fs"],
                                 f0, band, win_s, n_events):
            rows.append(dict(ev, s_idx=s_idx, label=ch["label"],
                             unit=ch["meta"].get("Unit for accelerometer", "g")))
    return rows


def _ds(arr, mx):
    arr = np.array(arr)
    if len(arr) <= mx:
        return arr
    return arr[:: len(arr) // mx]


# ── plot styling ──────────────────────────────────────────────────────────────

def _style(fig, height=600):
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=PLOT_BG,
        font_color="#e0e0f0", height=height,
        margin=dict(l=55, r=20, t=55, b=40),
        legend=dict(bgcolor="#2a2a3e", bordercolor="#3a3a5e",
                    font=dict(color="#e0e0f0")),
    )
    fig.update_xaxes(gridcolor=GRID_COL, zerolinecolor=GRID_COL,
                     title_font_color="#a0a0b0", tickfont_color="#a0a0b0")
    fig.update_yaxes(gridcolor=GRID_COL, zerolinecolor=GRID_COL,
                     title_font_color="#a0a0b0", tickfont_color="#a0a0b0")
    return fig


# ── Streamlit app ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Accelerometer Visualizer",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #12121c; }
[data-testid="stSidebar"]          { background: #1a1a2e; }
.stTabs [data-baseweb="tab-list"]  { background: #1a1a2e; gap: 4px; }
.stTabs [data-baseweb="tab"]       { color: #a0a0b0; border-radius: 4px 4px 0 0; }
.stTabs [aria-selected="true"]     { color: #e0e0f0 !important;
                                     border-bottom: 2px solid #7c3aed !important; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Accelerometer FFT Visualizer")

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    n_peaks = st.slider("Peaks to detect", 1, 15, 6)
    st.markdown("---")
    st.markdown("""
**How to upload**

Upload all `.txt` files for a test at once.
The app groups them automatically by channel
(`Acel_1`, `Acel_2`, `Acel_3` in the filename).
Part files (`_part001`, `_part002`, …) are
concatenated in order.
""")

tab_single, tab_compare, tab_3accel, tab_overlay, tab_damp = st.tabs(["🔬 Single Test", "📊 Compare Tests", "📡 Compare 3 Accelerometers", "🎛️ Sensors per Test", "〰 Damping"])

# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE TEST
# ═══════════════════════════════════════════════════════════════════════════════
with tab_single:
    st.caption("Upload all .txt files for one test (all channels + parts).")

    uploaded = st.file_uploader(
        "Drop files here", type=["txt"],
        accept_multiple_files=True, key="single",
    )

    if uploaded:
        file_data = tuple((f.name, f.read()) for f in uploaded)
        channels  = process_test(file_data)

        if not channels:
            st.error("No channels detected. Filenames must contain 'Acel_1', 'Acel_2', or 'Acel_3'.")
        else:
            # info strip
            info_cols = st.columns(len(channels))
            for col, ch in zip(info_cols, channels):
                col.metric(
                    ch["label"].upper(),
                    f"{ch['n_samples']:,} samples",
                    f"{ch['duration']:.0f} s  @  {ch['fs']:.0f} Hz",
                )

            n_ch   = len(channels)
            titles = (
                [f"{ch['label'].upper()} — Time" for ch in channels]
              + [f"{ch['label'].upper()} — FFT"  for ch in channels]
            )
            fig = make_subplots(
                rows=2, cols=n_ch,
                subplot_titles=titles,
                vertical_spacing=0.14,
                horizontal_spacing=0.06,
            )

            for i, ch in enumerate(channels, 1):
                color = SENSOR_COLORS[(i - 1) % len(SENSOR_COLORS)]
                unit  = ch["meta"].get("Unit for accelerometer", "g")
                frq   = np.array(ch["frq"])
                amp   = np.array(ch["amp"])

                # time-domain (downsampled)
                fig.add_trace(go.Scatter(
                    x=_ds(ch["time"],   MAX_TIME_PTS),
                    y=_ds(ch["values"], MAX_TIME_PTS),
                    mode="lines", line=dict(color=color, width=0.7),
                    showlegend=False,
                ), row=1, col=i)
                fig.update_xaxes(title_text="Time (s)",        row=1, col=i)
                fig.update_yaxes(title_text=f"Acc ({unit})",   row=1, col=i)

                # FFT
                fig.add_trace(go.Scatter(
                    x=frq, y=amp, mode="lines",
                    line=dict(color=color, width=0.8),
                    showlegend=False,
                ), row=2, col=i)
                fig.update_xaxes(title_text="Frequency (Hz)",        row=2, col=i)
                fig.update_yaxes(title_text=f"Amplitude ({unit})",   row=2, col=i)

                pidx = _peaks(frq, amp, n_peaks)
                for rank, idx in enumerate(pidx):
                    f, a = frq[idx], amp[idx]
                    fig.add_trace(go.Scatter(
                        x=[f], y=[a], mode="markers+text",
                        marker=dict(color=color, size=7),
                        text=[f"f{rank+1}={f:.2f} Hz"],
                        textposition="top right",
                        textfont=dict(size=8, color=color),
                        showlegend=False,
                    ), row=2, col=i)
                    fig.add_shape(
                        type="line", x0=f, x1=f, y0=0, y1=a,
                        line=dict(color=color, width=0.8, dash="dash"),
                        opacity=0.5, row=2, col=i,
                    )

            st.plotly_chart(_style(fig, 720), use_container_width=True)

            # ── combined overlay: all sensors on one FFT ──────────────────────
            st.markdown("#### All Sensors — FFT Overlay")
            fig_ov = go.Figure()
            for i, ch in enumerate(channels):
                color = SENSOR_COLORS[i % len(SENSOR_COLORS)]
                frq   = np.array(ch["frq"])
                amp   = np.array(ch["amp"])
                unit  = ch["meta"].get("Unit for accelerometer", "g")

                fig_ov.add_trace(go.Scatter(
                    x=frq, y=amp, mode="lines",
                    line=dict(color=color, width=1.0),
                    name=ch["label"].upper(),
                ))

                pidx = _peaks(frq, amp, n_peaks)
                for rank, idx in enumerate(pidx):
                    f, a = float(frq[idx]), float(amp[idx])
                    fig_ov.add_trace(go.Scatter(
                        x=[f], y=[a], mode="markers+text",
                        marker=dict(color=color, size=7),
                        text=[f"{ch['label'].upper()} f{rank+1}={f:.2f} Hz"],
                        textposition="top right",
                        textfont=dict(size=8, color=color),
                        showlegend=False,
                    ))
                    fig_ov.add_shape(
                        type="line", x0=f, x1=f, y0=0, y1=a,
                        line=dict(color=color, width=0.7, dash="dash"),
                        opacity=0.4,
                    )

            fig_ov.update_xaxes(title_text="Frequency (Hz)")
            fig_ov.update_yaxes(title_text=f"Amplitude ({unit})")
            st.plotly_chart(_style(fig_ov, 450), use_container_width=True)

            # peak table
            st.markdown("#### Peak Frequencies & Damping")
            cells = [_peak_cells(ch, n_peaks) for ch in channels]
            rows = []
            for rank in range(n_peaks):
                row = {"Rank": f"f{rank+1}"}
                for ch, c in zip(channels, cells):
                    row[ch["label"].upper()]        = c[rank][0]
                    row[f"{ch['label'].upper()} ζ"] = c[rank][1]
                rows.append(row)
            st.dataframe(pd.DataFrame(rows).set_index("Rank"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARE TESTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    n_tests = st.radio(
        "Number of tests to compare", [2, 3, 4],
        horizontal=True, key="n_tests",
    )

    st.caption("Upload all .txt files for each test (all channels + parts).")

    upload_cols = st.columns(n_tests)
    tests = []

    for i, col in enumerate(upload_cols):
        with col:
            test_name = st.text_input(
                "Test name", value=f"Test {i + 1}", key=f"tname_{i}",
            )
            ups = st.file_uploader(
                f"Files for {test_name}", type=["txt"],
                accept_multiple_files=True, key=f"compare_{i}",
            )
            if ups:
                file_data = tuple((f.name, f.read()) for f in ups)
                channels  = process_test(file_data)
                if channels:
                    tests.append({"name": test_name, "channels": channels})
                    st.success(
                        f"✓ {len(channels)} ch · "
                        f"{channels[0]['n_samples']:,} pts · "
                        f"{channels[0]['duration']:.0f} s"
                    )
                else:
                    st.error("No channels detected in these files.")

    if len(tests) == n_tests:
        sensors = [ch["label"] for ch in tests[0]["channels"]]
        n_sens  = min(len(sensors), 3)

        # FFT comparison
        fig = make_subplots(
            rows=1, cols=n_sens,
            subplot_titles=[s.upper() for s in sensors[:n_sens]],
            horizontal_spacing=0.07,
        )

        for t_idx, test in enumerate(tests):
            color = TEST_COLORS[t_idx % len(TEST_COLORS)]
            dash  = TEST_DASH[t_idx  % len(TEST_DASH)]

            for s_idx, ch in enumerate(test["channels"][:n_sens], 1):
                frq  = np.array(ch["frq"])
                amp  = np.array(ch["amp"])
                unit = ch["meta"].get("Unit for accelerometer", "g")

                fig.add_trace(go.Scatter(
                    x=frq, y=amp, mode="lines",
                    line=dict(color=color, width=1.0, dash=dash),
                    name=test["name"],
                    legendgroup=test["name"],
                    showlegend=(s_idx == 1),
                ), row=1, col=s_idx)

                pidx = _peaks(frq, amp, n_peaks)
                for idx in pidx:
                    f, a = frq[idx], amp[idx]
                    fig.add_trace(go.Scatter(
                        x=[f], y=[a], mode="markers",
                        marker=dict(color=color, size=6),
                        showlegend=False, legendgroup=test["name"],
                    ), row=1, col=s_idx)
                    fig.add_shape(
                        type="line", x0=f, x1=f, y0=0, y1=a,
                        line=dict(color=color, width=0.7, dash="dash"),
                        opacity=0.4, row=1, col=s_idx,
                    )

                fig.update_xaxes(title_text="Frequency (Hz)",      row=1, col=s_idx)
                fig.update_yaxes(title_text=f"Amplitude ({unit})", row=1, col=s_idx)

        st.plotly_chart(_style(fig, 480), use_container_width=True)

        # peak comparison table
        st.markdown("#### Peak Frequency & Damping Comparison")
        cells = {id(ch): _peak_cells(ch, n_peaks)
                 for test in tests for ch in test["channels"]}
        rows = []
        for rank in range(n_peaks):
            row = {"Rank": f"f{rank+1}"}
            for test in tests:
                for ch in test["channels"]:
                    key  = f"{ch['label'].upper()} — {test['name']}"
                    row[key], row[f"{key} ζ"] = cells[id(ch)][rank]
            rows.append(row)
        st.dataframe(
            pd.DataFrame(rows).set_index("Rank"),
            use_container_width=True,
        )

        # ── mean frequency summary per accelerometer ──────────────────────────
        st.markdown("#### Mean Peak Frequencies & Damping per Accelerometer")
        mean_rows = []
        for rank in range(n_peaks):
            mean_row = {"Rank": f"f{rank+1}"}
            for s_idx in range(n_sens):
                sensor_label = sensors[s_idx].upper()
                freqs, zetas = [], []
                for test in tests:
                    if s_idx < len(test["channels"]):
                        ch   = test["channels"][s_idx]
                        frq  = np.array(ch["frq"])
                        amp  = np.array(ch["amp"])
                        pidx = _peaks(frq, amp, n_peaks)
                        if rank < len(pidx):
                            freqs.append(frq[pidx[rank]])
                            z = _damping(ch["psd_f"], ch["psd_p"], frq[pidx[rank]])
                            if z is not None:
                                zetas.append(z)
                mean_row[sensor_label] = (
                    f"{np.mean(freqs):.3f} Hz" if freqs else "—"
                )
                mean_row[f"{sensor_label} ζ"] = (
                    f"{np.mean(zetas):.2f} %" if zetas else "—"
                )
            mean_rows.append(mean_row)
        st.dataframe(
            pd.DataFrame(mean_rows).set_index("Rank"),
            use_container_width=True,
        )

    elif tests:
        st.info(f"Upload files for all {n_tests} tests to see the comparison "
                f"({len(tests)}/{n_tests} loaded).")

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARE 3 ACCELEROMETERS  (all 3 sensors on one FFT plot, one test)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_3accel:
    st.caption("Upload all .txt files for one test. All three accelerometers are shown on the same frequency plot.")

    uploaded_3a = st.file_uploader(
        "Drop files here", type=["txt"],
        accept_multiple_files=True, key="three_accel",
    )

    if uploaded_3a:
        file_data_3a = tuple((f.name, f.read()) for f in uploaded_3a)
        channels_3a  = process_test(file_data_3a)

        if not channels_3a:
            st.error("No channels detected. Filenames must contain 'Acel_1', 'Acel_2', or 'Acel_3'.")
        else:
            # info strip
            info_cols = st.columns(len(channels_3a))
            for col, ch in zip(info_cols, channels_3a):
                col.metric(
                    ch["label"].upper(),
                    f"{ch['n_samples']:,} samples",
                    f"{ch['duration']:.0f} s  @  {ch['fs']:.0f} Hz",
                )

            fig_3a = go.Figure()
            unit_3a = "g"

            for i, ch in enumerate(channels_3a):
                color   = SENSOR_COLORS[i % len(SENSOR_COLORS)]
                frq     = np.array(ch["frq"])
                amp     = np.array(ch["amp"])
                unit_3a = ch["meta"].get("Unit for accelerometer", "g")

                fig_3a.add_trace(go.Scatter(
                    x=frq, y=amp, mode="lines",
                    line=dict(color=color, width=1.2),
                    name=ch["label"].upper(),
                ))

                pidx = _peaks(frq, amp, n_peaks)
                for rank, idx in enumerate(pidx):
                    f, a = float(frq[idx]), float(amp[idx])
                    fig_3a.add_trace(go.Scatter(
                        x=[f], y=[a], mode="markers+text",
                        marker=dict(color=color, size=8),
                        text=[f"{ch['label'].upper()} f{rank+1}={f:.2f} Hz"],
                        textposition="top right",
                        textfont=dict(size=9, color=color),
                        showlegend=False,
                    ))
                    fig_3a.add_shape(
                        type="line", x0=f, x1=f, y0=0, y1=a,
                        line=dict(color=color, width=0.8, dash="dash"),
                        opacity=0.4,
                    )

            fig_3a.update_xaxes(title_text="Frequency (Hz)")
            fig_3a.update_yaxes(title_text=f"Amplitude ({unit_3a})")
            fig_3a.update_layout(legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="left", x=0,
            ))
            st.plotly_chart(_style(fig_3a, 520), use_container_width=True)

            # peak table
            st.markdown("#### Peak Frequencies & Damping")
            cells_3a = [_peak_cells(ch, n_peaks) for ch in channels_3a]
            rows_3a = []
            for rank in range(n_peaks):
                row = {"Rank": f"f{rank+1}"}
                for ch, c in zip(channels_3a, cells_3a):
                    row[ch["label"].upper()]        = c[rank][0]
                    row[f"{ch['label'].upper()} ζ"] = c[rank][1]
                rows_3a.append(row)
            st.dataframe(
                pd.DataFrame(rows_3a).set_index("Rank"),
                use_container_width=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
# SENSORS PER TEST  (all sensors overlaid, one plot per test)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_overlay:
    n_tests_ov = st.radio(
        "Number of tests", [1, 2, 3, 4],
        horizontal=True, key="n_tests_ov",
    )

    st.caption("Upload all .txt files for each test. Each test produces one FFT plot with all sensors overlaid.")

    upload_cols_ov = st.columns(n_tests_ov)
    tests_ov = []

    for i, col in enumerate(upload_cols_ov):
        with col:
            test_name = st.text_input(
                "Test name", value=f"Test {i + 1}", key=f"ovname_{i}",
            )
            ups = st.file_uploader(
                f"Files for {test_name}", type=["txt"],
                accept_multiple_files=True, key=f"overlay_{i}",
            )
            if ups:
                file_data = tuple((f.name, f.read()) for f in ups)
                channels  = process_test(file_data)
                if channels:
                    tests_ov.append({"name": test_name, "channels": channels})
                    st.success(
                        f"✓ {len(channels)} ch · "
                        f"{channels[0]['n_samples']:,} pts · "
                        f"{channels[0]['duration']:.0f} s"
                    )
                else:
                    st.error("No channels detected in these files.")

    if tests_ov:
        for test in tests_ov:
            st.markdown(f"#### {test['name']}")
            fig = go.Figure()

            for i, ch in enumerate(test["channels"]):
                color = SENSOR_COLORS[i % len(SENSOR_COLORS)]
                frq   = np.array(ch["frq"])
                amp   = np.array(ch["amp"])
                unit  = ch["meta"].get("Unit for accelerometer", "g")

                fig.add_trace(go.Scatter(
                    x=frq, y=amp, mode="lines",
                    line=dict(color=color, width=1.0),
                    name=ch["label"].upper(),
                ))

                pidx = _peaks(frq, amp, n_peaks)
                for rank, idx in enumerate(pidx):
                    f, a = float(frq[idx]), float(amp[idx])
                    fig.add_trace(go.Scatter(
                        x=[f], y=[a], mode="markers+text",
                        marker=dict(color=color, size=7),
                        text=[f"{ch['label'].upper()} f{rank+1}={f:.2f} Hz"],
                        textposition="top right",
                        textfont=dict(size=8, color=color),
                        showlegend=False,
                        legendgroup=ch["label"],
                    ))
                    fig.add_shape(
                        type="line", x0=f, x1=f, y0=0, y1=a,
                        line=dict(color=color, width=0.7, dash="dash"),
                        opacity=0.4,
                    )

            fig.update_xaxes(title_text="Frequency (Hz)")
            fig.update_yaxes(title_text=f"Amplitude ({unit})")
            st.plotly_chart(_style(fig, 450), use_container_width=True)

        # combined peak table across all loaded tests
        if len(tests_ov) > 1:
            st.markdown("#### Peak Frequencies & Damping — All Tests")
            cells = {id(ch): _peak_cells(ch, n_peaks)
                     for test in tests_ov for ch in test["channels"]}
            rows = []
            for rank in range(n_peaks):
                row = {"Rank": f"f{rank+1}"}
                for test in tests_ov:
                    for ch in test["channels"]:
                        key  = f"{ch['label'].upper()} — {test['name']}"
                        row[key], row[f"{key} ζ"] = cells[id(ch)][rank]
                rows.append(row)
            st.dataframe(
                pd.DataFrame(rows).set_index("Rank"),
                use_container_width=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
# DAMPING  (free-decay envelope method, one test)
# ═══════════════════════════════════════════════════════════════════════════════
with tab_damp:
    st.caption(
        "Upload all .txt files for one test. For each accelerometer the largest "
        "excitations are band-pass filtered around f₀, the envelope of maxima is "
        "fitted on a log scale, and ζ = −m / (2π·f₀) · 100 %, where m is the slope."
    )

    uploaded_d = st.file_uploader(
        "Drop files here", type=["txt"],
        accept_multiple_files=True, key="damping",
    )

    if uploaded_d:
        file_data_d = tuple((f.name, f.read()) for f in uploaded_d)
        channels_d  = process_test(file_data_d)

        if not channels_d:
            st.error("No channels detected. Filenames must contain 'Acel_1', 'Acel_2', or 'Acel_3'.")
        else:
            # f0 choices: detected FFT peaks of every sensor
            peak_freqs = sorted({
                round(float(np.array(ch["frq"])[i]), 2)
                for ch in channels_d for i in _peaks(ch["frq"], ch["amp"], n_peaks)
            })
            f0_default = _dominant_f0(channels_d)
            nearest    = int(np.argmin([abs(f - f0_default) for f in peak_freqs]))

            c1, c2, c3, c4 = st.columns(4)
            f0_choice = c1.selectbox(
                "f₀ (Hz) — FFT peaks", peak_freqs + ["Custom…"], index=nearest,
                format_func=lambda f: f if isinstance(f, str) else f"{f:.2f}",
                key="damp_f0_choice",
            )
            if f0_choice == "Custom…":
                f0_d = c1.number_input("Custom f₀ (Hz)", 0.1, channels_d[0]["fs"] / 2,
                                       round(f0_default, 2), 0.01, format="%.2f",
                                       key="damp_f0_custom")
            else:
                f0_d = float(f0_choice)
            band_d = c2.number_input("Band ± (Hz)", 0.1, 5.0, 0.5, 0.1, key="damp_band")
            win_d  = c3.number_input("Decay window (s)", 1.0, 30.0, 5.0, 0.5, key="damp_win")
            nev_d  = int(c4.number_input("Events / sensor", 1, 10, 3, key="damp_nev"))

            if band_d >= f0_d:
                st.error("Band must be smaller than f₀.")
            else:
                rows_d = run_damping(file_data_d, f0_d, band_d, win_d, nev_d)

                if not rows_d:
                    st.warning("No decays found — try another f₀ or a wider band.")
                else:
                    st.markdown("#### Damping per Event")
                    st.caption("Untick **Used** to exclude a weak fit (low R²) from the means.")
                    df_d = pd.DataFrame([{
                        "Channel":         r["label"].upper(),
                        "Event t (s)":     round(r["t0"], 1),
                        "Units":           r["unit"],
                        "Slope m (1/s)":   round(r["slope"], 4),
                        "f₀ (Hz)":         round(f0_d, 2),
                        "Damping ratio %": round(r["zeta"], 3),
                        "R²":              round(r["r2"], 2),
                        "Used":            True,
                    } for r in rows_d])
                    # key tied to the inputs so exclusions reset when they change
                    editor_key = (f"damp_editor_{hash(tuple(n for n, _ in file_data_d))}"
                                  f"_{f0_d}_{band_d}_{win_d}_{nev_d}")
                    df_d = st.data_editor(
                        df_d, hide_index=True, use_container_width=True,
                        disabled=[c for c in df_d.columns if c != "Used"],
                        key=editor_key,
                    )

                    # per-accelerometer means (a1, a2, a3) and overall mean
                    used   = df_d["Used"].to_numpy()
                    zetas  = np.array([r["zeta"] for r in rows_d])
                    s_idxs = np.array([r["s_idx"] for r in rows_d])
                    cols_m = st.columns(len(channels_d) + 1)
                    for s_idx, (col, ch) in enumerate(zip(cols_m, channels_d)):
                        z = zetas[used & (s_idxs == s_idx)]
                        col.metric(f"a{s_idx + 1} ({ch['label'].upper()})",
                                   f"{z.mean():.3f} %" if len(z) else "—")
                    cols_m[-1].metric("Promedio",
                                      f"{zetas[used].mean():.3f} %" if used.any() else "—")

                    st.download_button(
                        "⬇️ Download table (CSV)",
                        df_d.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"damping_{f0_d:.2f}Hz.csv", mime="text/csv",
                    )

                    # plots for one event: filtered signal + envelope, regression
                    st.markdown("#### Envelope Fit")
                    best = int(np.argmax([r["r2"] for r in rows_d]))
                    k = st.selectbox(
                        "Event", range(len(rows_d)), index=best,
                        format_func=lambda i: (
                            f"{rows_d[i]['label'].upper()} · t = {rows_d[i]['t0']:.1f} s · "
                            f"ζ = {rows_d[i]['zeta']:.3f} % · R² = {rows_d[i]['r2']:.2f}"),
                        key="damp_event",
                    )
                    ev    = rows_d[k]
                    color = SENSOR_COLORS[ev["s_idx"] % len(SENSOR_COLORS)]
                    fig_d = make_subplots(
                        rows=1, cols=2, horizontal_spacing=0.08,
                        subplot_titles=[
                            f"{ev['label'].upper()} — filtered around {f0_d:.2f} Hz",
                            f"Linear regression of envelope — ζ = {ev['zeta']:.3f} %",
                        ],
                    )
                    fig_d.add_trace(go.Scatter(
                        x=ev["t_seg"], y=ev["y_seg"], mode="lines",
                        line=dict(color=color, width=1.0),
                        name=f"filtered {f0_d:.2f} ± {band_d:g} Hz",
                    ), row=1, col=1)
                    fig_d.add_trace(go.Scatter(
                        x=ev["t_mx"], y=ev["a_mx"], mode="lines",
                        line=dict(color="#ef4444", width=2),
                        name="envelope of maxima",
                    ), row=1, col=1)
                    fig_d.add_trace(go.Scatter(
                        x=ev["t_mx"], y=np.log(ev["a_mx"]), mode="markers",
                        marker=dict(symbol="triangle-up-open", color="#ef4444", size=9),
                        name="ln(envelope)",
                    ), row=1, col=2)
                    fig_d.add_trace(go.Scatter(
                        x=ev["t_mx"], y=ev["slope"] * ev["t_mx"] + ev["icpt"],
                        mode="lines", line=dict(color="#60a5fa", width=1.5),
                        name=f"fit: m = {ev['slope']:.4f} 1/s, R² = {ev['r2']:.2f}",
                    ), row=1, col=2)
                    fig_d.update_xaxes(title_text="Time (s)")
                    fig_d.update_yaxes(title_text=f"Acc ({ev['unit']})", row=1, col=1)
                    fig_d.update_yaxes(title_text="ln(amplitude)",       row=1, col=2)
                    fig_d.update_layout(legend=dict(
                        orientation="h", yanchor="bottom", y=-0.3,
                        xanchor="left", x=0,
                    ))
                    st.plotly_chart(_style(fig_d, 480), use_container_width=True)
