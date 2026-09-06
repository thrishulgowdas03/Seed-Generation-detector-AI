import json
import tempfile
from pathlib import Path
import cv2
import streamlit as st
from pipeline import analyze

st.set_page_config(page_title="Seed Germination AI", page_icon="🌱", layout="wide")

st.title("🌱 Seed Germination AI")
st.caption("Automatic crop detection • Seed detection • Germination classification")
st.info("Upload one tray image. The system automatically identifies Maize, Ragi, or Paddy — no crop selection is required.")

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded:
    data = uploaded.getvalue()
    suffix = Path(uploaded.name).suffix or ".jpg"
    with tempfile.TemporaryDirectory() as td:
        input_path = Path(td) / f"input{suffix}"
        input_path.write_bytes(data)
        try:
            with st.spinner("Analyzing image…"):
                annotated, result = analyze(input_path, td)
        except Exception as e:
            st.error("Analysis failed")
            st.exception(e)
            st.stop()

        crop = result["crop"]
        counts = result["counts"]
        total = result["total_seeds"]
        germinated = result.get(
            "germinated_total",
            counts.get("GERMI", counts.get("germinated", 0)) +
            counts.get("SEMI GERMI", counts.get("semi_germinated", 0)),
        )
        rate = result.get("germination_rate", germinated / total if total else 0)

        st.success(f"Automatically detected crop: **{crop}**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total seeds", total)
        c2.metric("GERMI", counts.get("GERMI", counts.get("germinated", 0)))
        c3.metric("SEMI GERMI", counts.get("SEMI GERMI", counts.get("semi_germinated", 0)))
        c4.metric("Germination %", f"{rate * 100:.1f}%")

        st.subheader("Analysis result")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.subheader("Class distribution")
        rows = []
        for k, v in counts.items():
            rows.append({
                "Class": k,
                "Count": v,
                "Percentage": round(100 * v / total, 2) if total else 0,
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        st.download_button(
            "Download JSON",
            json.dumps(result, indent=2),
            file_name=f"{Path(uploaded.name).stem}_result.json",
            mime="application/json",
        )

st.markdown("---")
st.caption(
    "Maize: HSV seed/shoot detection + skeleton tracing + CLIP tie-break. "
    "Ragi: OpenCV seed-body detection + Ragi-tailored CLIP. "
    "Paddy: OpenCV seed detection + Paddy-tailored CLIP with growth evidence."
)
