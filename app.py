import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import windows, find_peaks
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
        ))

    return channels


def _peaks(frq, amp, n):
    frq, amp = np.array(frq), np.array(amp)
    min_prom = amp.max() * 0.01
    idx, _   = find_peaks(amp, prominence=min_prom, distance=5)
    idx      = idx[np.argsort(amp[idx])[::-1]][:n]
    return np.sort(idx)


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

tab_single, tab_compare = st.tabs(["🔬 Single Test", "📊 Compare Tests"])

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

            # peak table
            st.markdown("#### Peak Frequencies")
            rows = []
            for rank in range(n_peaks):
                row = {"Rank": f"f{rank+1}"}
                for ch in channels:
                    pidx = _peaks(np.array(ch["frq"]), np.array(ch["amp"]), n_peaks)
                    row[ch["label"].upper()] = (
                        f"{np.array(ch['frq'])[pidx[rank]]:.3f} Hz"
                        if rank < len(pidx) else "—"
                    )
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
        st.markdown("#### Peak Frequency Comparison")
        rows = []
        for rank in range(n_peaks):
            row = {"Rank": f"f{rank+1}"}
            for test in tests:
                for ch in test["channels"]:
                    frq  = np.array(ch["frq"])
                    amp  = np.array(ch["amp"])
                    pidx = _peaks(frq, amp, n_peaks)
                    key  = f"{ch['label'].upper()} — {test['name']}"
                    row[key] = (f"{frq[pidx[rank]]:.3f} Hz"
                                if rank < len(pidx) else "—")
            rows.append(row)
        st.dataframe(
            pd.DataFrame(rows).set_index("Rank"),
            use_container_width=True,
        )

    elif tests:
        st.info(f"Upload files for all {n_tests} tests to see the comparison "
                f"({len(tests)}/{n_tests} loaded).")
