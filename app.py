"""Streamlit UI for diabetes risk estimation."""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Diabetes Risk Estimator",
    page_icon="🩺",
    layout="wide",
)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
FEATURE_ORDER = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "Age",
    "Pregnancies",
    "DiabetesPedigreeFunction",
]

DEFAULT_FEATURES = {
    "Pregnancies": 0,
    "Age": 33,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 20,
    "Insulin": 79,
    "BMI": 25.0,
    "DiabetesPedigreeFunction": 0.5,
}

PRESETS = {
    "Custom": DEFAULT_FEATURES,
    "Low risk sample": {
        "Pregnancies": 1,
        "Age": 28,
        "Glucose": 95,
        "BloodPressure": 70,
        "SkinThickness": 22,
        "Insulin": 50,
        "BMI": 22.5,
        "DiabetesPedigreeFunction": 0.3,
    },
    "Moderate risk sample": {
        "Pregnancies": 3,
        "Age": 42,
        "Glucose": 135,
        "BloodPressure": 82,
        "SkinThickness": 28,
        "Insulin": 120,
        "BMI": 29.0,
        "DiabetesPedigreeFunction": 0.6,
    },
    "High risk sample": {
        "Pregnancies": 6,
        "Age": 55,
        "Glucose": 180,
        "BloodPressure": 92,
        "SkinThickness": 35,
        "Insulin": 210,
        "BMI": 34.5,
        "DiabetesPedigreeFunction": 1.1,
    },
}


@st.cache_resource(show_spinner=False)
def load_pickle(name: str):
    """Load a pickle artifact once and reuse across reruns."""
    artifact_path = ARTIFACT_DIR / name
    if not artifact_path.exists():
        st.error(f"Missing artifact: {artifact_path}")
        st.stop()
    with artifact_path.open("rb") as f:
        return pickle.load(f)


@st.cache_resource(show_spinner=True)
def load_artifacts():
    return {
        "model": load_pickle("model.pkl"),
        "preprocessor": load_pickle("preprocessed.pkl"),
    }


def predict_diabetes(features: dict) -> float:
    artifacts = load_artifacts()
    input_frame = pd.DataFrame(
        [[features[col] for col in FEATURE_ORDER]],
        columns=FEATURE_ORDER,
    )
    processed = artifacts["preprocessor"].transform(input_frame)
    return float(artifacts["model"].predict_proba(processed)[0][1])


def risk_bucket(percent: float):
    if percent >= 50:
        return "High", "#dc2626"
    if percent >= 20:
        return "Moderate", "#d97706"
    return "Low", "#16a34a"


st.markdown(
    """
    <style>
        .card {
            padding: 1rem;
            border-radius: 0.9rem;
            border: 1px solid #e5e7eb;
            background: #f8fafc;
        }
        .pill {
            padding: 0.15rem 0.65rem;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩺 Diabetes Prediction App")
st.subheader("Interactive risk estimate from clinical inputs")
st.write(
    "Use recent measurements to generate a probabilistic diabetes risk. "
    "Results do not replace clinical judgement."
)

if "form_values" not in st.session_state:
    st.session_state.form_values = DEFAULT_FEATURES.copy()

col_info, col_preset = st.columns([3, 2])
with col_info:
    st.info(
        "Fill in the clinical intake form below or apply a preset scenario "
        "to see the model response."
    )
with col_preset:
    preset_choice = st.selectbox(
        "Presets",
        list(PRESETS.keys()),
        index=list(PRESETS.keys()).index("Custom"),
    )
    preset_key = preset_choice or "Custom"
    if st.button("Apply preset"):
        selected_preset = PRESETS.get(preset_key, DEFAULT_FEATURES)
        st.session_state.form_values = selected_preset.copy()
        st.experimental_rerun()

with st.sidebar:
    st.markdown(
        """
        **How to use**
        - Provide recent measurements.
        - Keep inputs within realistic ranges.
        - Output is a probability, not a diagnosis.
        """
    )

with st.form("diabetes_form"):
    col1, col2 = st.columns(2)
    pregnancies = col1.number_input(
        "Pregnancies",
        min_value=0,
        max_value=17,
        value=st.session_state.form_values["Pregnancies"],
        step=1,
    )
    age = col2.number_input(
        "Age (years)",
        min_value=18,
        max_value=90,
        value=st.session_state.form_values["Age"],
        step=1,
    )

    glucose = col1.number_input(
        "Glucose (mg/dL)",
        min_value=40,
        max_value=260,
        value=st.session_state.form_values["Glucose"],
        step=1,
    )
    bloodpressure = col2.number_input(
        "Blood Pressure (mm Hg)",
        min_value=40,
        max_value=160,
        value=st.session_state.form_values["BloodPressure"],
        step=1,
    )

    skinthickness = col1.number_input(
        "Skin Thickness (mm)",
        min_value=0,
        max_value=99,
        value=st.session_state.form_values["SkinThickness"],
        step=1,
    )
    insulin = col2.number_input(
        "Insulin (mu U/mL)",
        min_value=0,
        max_value=900,
        value=st.session_state.form_values["Insulin"],
        step=1,
    )

    bmi = col1.number_input(
        "BMI",
        min_value=10.0,
        max_value=70.0,
        value=st.session_state.form_values["BMI"],
        step=0.1,
        format="%.1f",
    )
    diabetes_pedigree = col2.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=st.session_state.form_values["DiabetesPedigreeFunction"],
        step=0.01,
        format="%.2f",
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    st.session_state.form_values = {
        "Glucose": glucose,
        "BloodPressure": bloodpressure,
        "SkinThickness": skinthickness,
        "Insulin": insulin,
        "BMI": bmi,
        "Age": age,
        "Pregnancies": pregnancies,
        "DiabetesPedigreeFunction": diabetes_pedigree,
    }

    user_features = {
        "Glucose": glucose,
        "BloodPressure": bloodpressure,
        "SkinThickness": skinthickness,
        "Insulin": insulin,
        "BMI": bmi,
        "Age": age,
        "Pregnancies": pregnancies,
        "DiabetesPedigreeFunction": diabetes_pedigree,
    }

    try:
        probability = predict_diabetes(user_features)
    except Exception as exc:  # pragma: no cover - defensive catch for UI
        st.error(f"Prediction failed: {exc}")
        st.stop()

    percent = round(probability * 100, 2)
    bucket, tone = risk_bucket(percent)

    main_col, side_col = st.columns([3, 2])
    with main_col:
        card_html = (
            "<div class='card'>"
            f"<h3>Estimated risk: {percent}%</h3>"
            f"<span class='pill' style='background:{tone}20; color:{tone};'>"
            f"{bucket} risk</span>"
            "</div>"
        )
        st.markdown(card_html, unsafe_allow_html=True)
        st.progress(min(max(probability, 0.0), 1.0))

        m1, m2, m3 = st.columns(3)
        m1.metric("Glucose", f"{glucose} mg/dL")
        m2.metric("BMI", f"{bmi:.1f}")
        m3.metric("Insulin", f"{insulin} mu U/mL")

    with side_col:
        st.markdown("**Next steps**")
        if bucket == "High":
            st.write("- Seek clinical review promptly.")
            st.write("- Consider further lab tests and monitoring.")
        elif bucket == "Moderate":
            st.write("- Discuss lifestyle and screening with a clinician.")
            st.write("- Recheck measurements if outdated.")
        else:
            st.write("- Maintain healthy lifestyle; periodic screening.")

    st.caption(
        "Model output is probabilistic and complements, not replaces, "
        "medical judgement."
    )

with st.expander("About this app"):
    st.write(
        "Trained on the PIMA diabetes dataset with preprocessing stored in "
        "`artifacts/preprocessed.pkl` and the model in `artifacts/model.pkl`."
    )
    st.write(
        "Inputs are standardized before inference. Always validate new "
        "models before clinical use."
    )
