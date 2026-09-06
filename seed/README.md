# Seed Germination AI Dashboard — Maize + Ragi

## What it does
Upload a single tray/paper-towel image. The dashboard automatically:
1. identifies Maize vs Ragi,
2. detects seed locations,
3. classifies each seed as GERMI / SEMI GERMI / NON GERMI,
4. calculates counts and germination percentage,
5. shows an annotated image,
6. lets you download the JSON result.

No crop dropdown is used.

## Architecture
- **Maize:** preserved validated HSV seed detection + yellow-green shoot mask + skeleton graph tracing + CLIP tie-break for ambiguous semi/non cases.
- **Ragi:** OpenCV seed-body detector that was tuned on the supplied Ragi images, followed by the validated V9-style growth classifier.

## Windows setup
Open Command Prompt in this folder:

```bat
python -m pip install -r requirements.txt
streamlit run app.py
```

A browser window should open. If it does not, open the local address printed by Streamlit.

### First Maize run
The Maize tie-break uses CLIP only for ambiguous seeds. Its weights may download the first time. Do this before the demo if possible.

## Important
The Ragi detector is tuned to the supplied white-paper/towel photo setup. Different lighting/backgrounds may require threshold tuning.

The system is a hackathon prototype, not a laboratory-grade germination assay.
