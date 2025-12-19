# Diabetes Prediction (Streamlit)

Interactive Streamlit app that estimates diabetes risk from clinical inputs using a trained model and preprocessing pipeline stored in `artifacts/`. The UI mirrors a simple intake form and returns a probabilistic risk category (low / moderate / high).

## Features

- Two-column clinical intake form (pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, age, diabetes pedigree).
- Cached loading of model and preprocessor for fast reruns.
- Risk bands with messaging (low, moderate, high) plus a clinical disclaimer.
- Sidebar guidance describing intended use and realistic input ranges.

## Inputs and output

- Inputs: numeric fields shown in the form. Keep within typical clinical ranges for best fidelity.
- Output: probability (0–100%) and a risk category. This is **not** a diagnosis and should be paired with clinical judgement.

## Quick start

1. Python 3.8+ recommended.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure artifacts exist:
   - `artifacts/model.pkl`
   - `artifacts/preprocessed.pkl`
4. Run the app (example port 8501):

   ```bash
   python -m streamlit run app.py --server.headless true --server.port 8501
   ```

5. Open the URL printed by Streamlit (e.g., <http://localhost:8501>).

## Project layout

- [app.py](app.py) — Streamlit UI and inference helper.
- [src/](src) — training pipeline components (ingestion, transformation, modeling, utilities).
- [artifacts/](artifacts) — serialized model, preprocessor, and sample splits.
- [data/](data) — raw and imputed datasets for experimentation.
- [requirements.txt](requirements.txt) — Python dependencies.

## Development notes

- Keep feature ordering in sync with `FEATURE_ORDER` in [app.py](app.py).
- If you retrain, regenerate both the model and the preprocessor and place them in `artifacts/`.
- Use `flake8` / `pylint` to keep the app lint-clean; update `requirements.txt` when adding libs.

## Troubleshooting

- Port already in use: run with `--server.port 0` to pick a random port or choose a free one.
- Missing artifacts: ensure `model.pkl` and `preprocessed.pkl` exist; the app will stop with an error if not found.

## Disclaimer

This tool provides probabilistic estimates only. It does not constitute medical advice or diagnosis. Always consult a qualified clinician for medical decisions.
