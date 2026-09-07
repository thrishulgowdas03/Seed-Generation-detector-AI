import json
import tempfile
from pathlib import Path

import cv2
import streamlit as st

from pipeline import analyze


# -----------------------------------------------------------------------------
# PAGE
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SeedVision AI | Germination Intelligence",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# PROFESSIONAL THEME
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 88% 3%, rgba(43, 170, 99, 0.10), transparent 25%),
            radial-gradient(circle at 4% 30%, rgba(21, 101, 192, 0.06), transparent 24%),
            #f7f9f8;
    }

    .block-container {
        max-width: 1380px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
    }

    .hero {
        background: linear-gradient(135deg, #0c3325 0%, #155c3c 58%, #1f7a4e 100%);
        border-radius: 24px;
        padding: 30px 34px;
        color: white;
        box-shadow: 0 18px 45px rgba(13, 61, 40, 0.18);
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
    }

    .hero:after {
        content: '✦';
        position: absolute;
        right: 38px;
        top: 12px;
        font-size: 92px;
        opacity: 0.08;
    }

    .hero-kicker {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        opacity: 0.78;
        margin-bottom: 8px;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 34px;
        line-height: 1.08;
        font-weight: 700;
        margin: 0;
    }

    .hero-subtitle {
        margin-top: 10px;
        font-size: 15px;
        max-width: 760px;
        opacity: 0.88;
        line-height: 1.55;
    }

    .pill-row { margin-top: 18px; }
    .pill {
        display: inline-block;
        padding: 6px 11px;
        margin-right: 7px;
        margin-bottom: 5px;
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 999px;
        background: rgba(255,255,255,0.09);
        font-size: 12px;
        font-weight: 600;
    }

    .section-label {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 20px;
        font-weight: 700;
        margin: 14px 0 10px 0;
        color: #163126;
    }

    .upload-card {
        background: white;
        border: 1px solid #e1e8e4;
        border-radius: 18px;
        padding: 19px 22px 8px 22px;
        box-shadow: 0 8px 25px rgba(20, 40, 30, 0.055);
        margin-bottom: 18px;
    }

    .metric-card {
        background: white;
        border: 1px solid #e1e8e4;
        border-radius: 16px;
        padding: 17px 18px;
        min-height: 112px;
        box-shadow: 0 7px 20px rgba(20, 40, 30, 0.045);
    }

    .metric-label {
        font-size: 12px;
        color: #687871;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .metric-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 29px;
        font-weight: 700;
        color: #14271f;
        margin-top: 8px;
    }

    .metric-note {
        font-size: 11px;
        color: #87948e;
        margin-top: 3px;
    }

    .status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border-radius: 999px;
        padding: 7px 12px;
        font-size: 12px;
        font-weight: 700;
        background: #e8f7ee;
        color: #17683d;
        border: 1px solid #cdebd9;
    }

    .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #20a464;
        display: inline-block;
    }

    .class-card {
        background: white;
        border: 1px solid #e1e8e4;
        border-radius: 15px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }

    .class-name {
        font-weight: 700;
        color: #183128;
        font-size: 13px;
    }

    .bar-bg {
        height: 8px;
        background: #edf1ef;
        border-radius: 20px;
        overflow: hidden;
        margin-top: 9px;
    }

    .bar-fill {
        height: 100%;
        border-radius: 20px;
        background: linear-gradient(90deg, #1e8f58, #62bd82);
    }

    .class-meta {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        color: #718078;
        margin-top: 6px;
    }

    .pie-panel {
        background: white;
        border: 1px solid #e1e8e4;
        border-radius: 18px;
        padding: 20px;
        box-shadow: 0 8px 25px rgba(20, 40, 30, 0.055);
        height: 100%;
    }

    .pie-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 17px;
        font-weight: 700;
        color: #183128;
        margin-bottom: 2px;
    }

    .pie-subtitle {
        font-size: 12px;
        color: #7a8881;
        margin-bottom: 14px;
    }

    .pie-wrap {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 24px;
        flex-wrap: wrap;
    }

    .pie-chart {
        width: 178px;
        height: 178px;
        border-radius: 50%;
        position: relative;
        flex: 0 0 auto;
        box-shadow: inset 0 0 0 1px rgba(20, 40, 30, 0.06);
    }

    .pie-chart::after {
        content: '';
        position: absolute;
        width: 94px;
        height: 94px;
        left: 42px;
        top: 42px;
        border-radius: 50%;
        background: white;
        box-shadow: 0 1px 4px rgba(20, 40, 30, 0.08);
    }

    .pie-center {
        position: absolute;
        z-index: 2;
        inset: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        pointer-events: none;
    }

    .pie-center-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 24px;
        font-weight: 700;
        color: #163126;
        line-height: 1;
    }

    .pie-center-label {
        font-size: 10px;
        color: #7a8881;
        margin-top: 5px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .pie-legend { min-width: 155px; }
    .pie-legend-row {
        display: flex;
        align-items: center;
        gap: 9px;
        margin: 10px 0;
        font-size: 12px;
        color: #43544c;
    }
    .pie-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        flex: 0 0 auto;
    }
    .pie-legend-name { flex: 1; font-weight: 600; }
    .pie-legend-value { font-weight: 700; color: #183128; }
    .quality-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 7px 11px;
        border-radius: 999px;
        background: #f1f7f3;
        border: 1px solid #dcebe2;
        color: #315044;
        font-size: 11px;
        font-weight: 700;
        margin-top: 10px;
    }

    .info-card {
        background: #f1f7f3;
        border: 1px solid #dcebe2;
        border-radius: 15px;
        padding: 16px 18px;
        color: #315044;
        font-size: 13px;
        line-height: 1.55;
    }

    .footer {
        text-align: center;
        color: #89958f;
        font-size: 11px;
        padding-top: 28px;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 14px;
    }

    div[data-testid="stSidebar"] {
        background: #10251c;
    }

    div[data-testid="stSidebar"] * {
        color: #edf5f0;
    }

    .sidebar-brand {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 21px;
        font-weight: 700;
        margin-bottom: 2px;
    }

    .sidebar-muted {
        color: #a9bbb1 !important;
        font-size: 12px;
        line-height: 1.5;
    }

    .step {
        display: flex;
        gap: 10px;
        margin: 11px 0;
        align-items: center;
    }

    .step-num {
        width: 24px;
        height: 24px;
        border-radius: 50%;
        background: rgba(111, 203, 145, 0.16);
        border: 1px solid rgba(111, 203, 145, 0.28);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 700;
        color: #bfe8cc;
    }

    .step-text { font-size: 12px; color: #dce8e1; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# SIDEBAR
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-brand">🌱 SeedVision AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-muted">Automated seed germination intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**How it works**")
    steps = [
        ("1", "Upload one tray image"),
        ("2", "Identify crop automatically"),
        ("3", "Detect individual seeds"),
        ("4", "Classify germination stage"),
        ("5", "Generate visual + numeric results"),
    ]
    for n, text in steps:
        st.markdown(
            f'<div class="step"><div class="step-num">{n}</div><div class="step-text">{text}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Supported crops**")
    st.markdown("🌽 Maize  ·  🌾 Ragi  ·  🌱 Paddy")
    st.markdown("---")
    st.markdown("**Analysis design**")
    st.markdown('<div class="sidebar-muted">Crop-specific OpenCV detection with pretrained CLIP classification. No manual crop selection.</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HERO
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">AI-powered seed analysis platform</div>
        <div class="hero-title">From one photograph to<br>germination intelligence.</div>
        <div class="hero-subtitle">
            Automatically identify the crop, locate individual seeds, classify germination stage,
            and present clear statistics with visual evidence — all from a single image.
        </div>
        <div class="pill-row">
            <span class="pill">⚡ Automated</span>
            <span class="pill">🧠 Computer Vision + AI</span>
            <span class="pill">🌽 Maize</span>
            <span class="pill">🌾 Ragi</span>
            <span class="pill">🌱 Paddy</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Start an analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Upload a clear tray / paper image",
    type=["jpg", "jpeg", "png"],
    help="For best results, keep the seeds visible, reasonably separated, and well lit.",
)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# ANALYSIS
# -----------------------------------------------------------------------------
if uploaded:
    data = uploaded.getvalue()
    suffix = Path(uploaded.name).suffix or ".jpg"

    with tempfile.TemporaryDirectory() as td:
        input_path = Path(td) / f"input{suffix}"
        input_path.write_bytes(data)

        try:
            progress = st.progress(0, text="Preparing image analysis…")
            progress.progress(15, text="Detecting crop and seed candidates…")
            annotated, result = analyze(input_path, td)
            progress.progress(100, text="Analysis complete")
            progress.empty()
        except Exception as e:
            st.error("Analysis could not be completed.")
            with st.expander("Technical error details"):
                st.exception(e)
            st.stop()

        crop = result.get("crop", "Unknown")
        counts = result.get("counts", {})
        total = int(result.get("total_seeds", sum(counts.values())))
        germi = int(counts.get("GERMI", 0))
        semi = int(counts.get("SEMI GERMI", 0))
        non = int(counts.get("NON GERMI", 0))
        germinated = int(result.get("germinated_total", germi + semi))
        strict_rate = float(result.get("germination_rate", germinated / total if total else 0))
        weighted_rate = result.get("weighted_germination_rate")
        if weighted_rate is None:
            weighted_rate = (germi + 0.5 * semi) / total if total else 0

        # Result header
        left, right = st.columns([3.8, 1.2])
        with left:
            st.markdown('<div class="section-label">Analysis complete</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="status"><span class="dot"></span> Crop automatically identified: {crop}</div>',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown("**File**")
            st.code(uploaded.name, language=None)

        # KPI cards
        c1, c2, c3, c4 = st.columns(4)
        cards = [
            ("Total seeds", f"{total}", "Detected candidates"),
            ("GERMI", f"{germi}", "Clearly germinated"),
            ("SEMI GERMI", f"{semi}", "Early / partial growth"),
            ("Germination", f"{strict_rate * 100:.1f}%", "GERMI + SEMI GERMI"),
        ]
        for col, (label, value, note) in zip([c1, c2, c3, c4], cards):
            with col:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">{label}</div>'
                    f'<div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
                    unsafe_allow_html=True,
                )

        st.write("")

        # ---------------------------------------------------------------------
        # PROFESSIONAL GERMINATION CHARTS
        # Shown directly after the KPI cards so judges see the quality
        # distribution immediately after analysis.
        # ---------------------------------------------------------------------
        g_pct = (germi / total * 100) if total else 0
        s_pct = (semi / total * 100) if total else 0
        n_pct = (non / total * 100) if total else 0
        germinated_pct = (germinated / total * 100) if total else 0

        stop1 = g_pct
        stop2 = g_pct + s_pct

        quality_gradient = (
            f"conic-gradient(#1f9d61 0% {stop1:.2f}%, "
            f"#e2b84b {stop1:.2f}% {stop2:.2f}%, "
            f"#d76b6b {stop2:.2f}% 100%)"
        )
        overall_gradient = (
            f"conic-gradient(#176b43 0% {germinated_pct:.2f}%, "
            f"#d76b6b {germinated_pct:.2f}% 100%)"
        )

        st.markdown(
            '<div class="section-label">Germination quality overview</div>',
            unsafe_allow_html=True,
        )

        p1, p2 = st.columns(2)

        with p1:
            st.markdown(
                f"""<div class="pie-panel">
                    <div class="pie-title">Seed quality distribution</div>
                    <div class="pie-subtitle">Classification of all detected seeds</div>
                    <div class="pie-wrap">
                        <div class="pie-chart" style="background:{quality_gradient};">
                            <div class="pie-center">
                                <div class="pie-center-value">{total}</div>
                                <div class="pie-center-label">total seeds</div>
                            </div>
                        </div>
                        <div class="pie-legend">
                            <div class="pie-legend-row">
                                <span class="pie-dot" style="background:#1f9d61"></span>
                                <span class="pie-legend-name">GERMI</span>
                                <span class="pie-legend-value">{germi} · {g_pct:.1f}%</span>
                            </div>
                            <div class="pie-legend-row">
                                <span class="pie-dot" style="background:#e2b84b"></span>
                                <span class="pie-legend-name">SEMI GERMI</span>
                                <span class="pie-legend-value">{semi} · {s_pct:.1f}%</span>
                            </div>
                            <div class="pie-legend-row">
                                <span class="pie-dot" style="background:#d76b6b"></span>
                                <span class="pie-legend-name">NON GERMI</span>
                                <span class="pie-legend-value">{non} · {n_pct:.1f}%</span>
                            </div>
                        </div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        with p2:
            st.markdown(
                f"""<div class="pie-panel">
                    <div class="pie-title">Overall germination</div>
                    <div class="pie-subtitle">GERMI + SEMI GERMI counted as germinated</div>
                    <div class="pie-wrap">
                        <div class="pie-chart" style="background:{overall_gradient};">
                            <div class="pie-center">
                                <div class="pie-center-value">{germinated_pct:.1f}%</div>
                                <div class="pie-center-label">germinated</div>
                            </div>
                        </div>
                        <div class="pie-legend">
                            <div class="pie-legend-row">
                                <span class="pie-dot" style="background:#176b43"></span>
                                <span class="pie-legend-name">Germinated</span>
                                <span class="pie-legend-value">{germinated} · {germinated_pct:.1f}%</span>
                            </div>
                            <div class="pie-legend-row">
                                <span class="pie-dot" style="background:#d76b6b"></span>
                                <span class="pie-legend-name">Non-germinated</span>
                                <span class="pie-legend-value">{non} · {100-germinated_pct:.1f}%</span>
                            </div>
                            <div class="quality-badge">
                                ● Weighted quality: {float(weighted_rate) * 100:.1f}%
                            </div>
                        </div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.write("")

        # Main result area
        tab_result, tab_details = st.tabs(
            ["📸 Visual analysis", "🔍 Technical details"]
        )

        with tab_result:
            img_col, summary_col = st.columns([2.25, 1])
            with img_col:
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption="Annotated output — each detected seed is marked with its predicted germination stage.",
                    use_container_width=True,
                )
            with summary_col:
                st.markdown('<div class="section-label">Quick interpretation</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="info-card"><b>{germinated}</b> of <b>{total}</b> detected seeds '
                    f'are classified as germinated when GERMI and SEMI GERMI are counted together.<br><br>'
                    f'<b>{non}</b> are classified as non-germinated.<br><br>'
                    f'Weighted germination: <b>{float(weighted_rate) * 100:.1f}%</b>.</div>',
                    unsafe_allow_html=True,
                )

                st.write("")
                st.download_button(
                    "⬇ Download annotated image",
                    data=cv2.imencode(".jpg", annotated)[1].tobytes(),
                    file_name=f"{Path(uploaded.name).stem}_annotated.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )
                st.download_button(
                    "⬇ Download analysis JSON",
                    data=json.dumps(result, indent=2, default=str),
                    file_name=f"{Path(uploaded.name).stem}_result.json",
                    mime="application/json",
                    use_container_width=True,
                )
