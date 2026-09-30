
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# AgriFlow AI — 3D Smart Irrigation Dashboard
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "irrigation_model.pkl"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

st.set_page_config(
    page_title="AgriFlow AI | Smart Irrigation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------- CSS ------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 80% 5%, rgba(76,175,80,.16), transparent 25%),
        radial-gradient(circle at 15% 80%, rgba(0,150,136,.12), transparent 28%),
        #07130f;
    color: #eefbf4;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a2118 0%, #07150f 100%);
    border-right: 1px solid rgba(125, 255, 177, .12);
}

section[data-testid="stSidebar"] * {
    color: #eafff1 !important;
}

.hero {
    padding: 24px 28px;
    border: 1px solid rgba(123, 255, 174, .14);
    border-radius: 24px;
    background:
        linear-gradient(135deg, rgba(24,75,48,.82), rgba(8,31,23,.92));
    box-shadow: 0 20px 70px rgba(0,0,0,.25);
    margin-bottom: 18px;
}

.hero-title {
    font-size: clamp(30px, 4vw, 52px);
    font-weight: 800;
    letter-spacing: -1.8px;
    margin: 0;
}

.hero-sub {
    color: #a9c9b7;
    margin-top: 7px;
    font-size: 15px;
}

.badge {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(83, 230, 133, .12);
    border: 1px solid rgba(83, 230, 133, .25);
    color: #78f19a;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 12px;
}

/* ---------- 3D FARM SCENE ---------- */
.scene {
    position: relative;
    height: 455px;
    overflow: hidden;
    border-radius: 28px;
    border: 1px solid rgba(137, 255, 183, .15);
    background:
        linear-gradient(180deg, #091f29 0%, #123d3b 47%, #1c5736 48%, #123a26 100%);
    box-shadow: inset 0 0 80px rgba(0,0,0,.28), 0 18px 55px rgba(0,0,0,.2);
    perspective: 1000px;
}

.sun {
    position:absolute;
    width:90px;
    height:90px;
    right:8%;
    top:11%;
    border-radius:50%;
    background: radial-gradient(circle, #fff4b1 0 20%, #ffd56a 42%, rgba(255,213,106,.05) 72%);
    filter: blur(.2px);
    box-shadow: 0 0 60px rgba(255,213,106,.32);
}

.cloud {
    position:absolute;
    width:110px;
    height:28px;
    border-radius:30px;
    background:rgba(226,255,246,.12);
    filter:blur(1px);
    top:18%;
    left:13%;
}
.cloud:before, .cloud:after {
    content:"";
    position:absolute;
    border-radius:50%;
    background:rgba(226,255,246,.14);
}
.cloud:before { width:45px;height:45px;left:20px;top:-19px; }
.cloud:after { width:56px;height:56px;right:14px;top:-28px; }

.ground {
    position:absolute;
    left:-8%;
    right:-8%;
    bottom:-100px;
    height:260px;
    border-radius:50% 50% 0 0;
    background:
      repeating-linear-gradient(165deg, rgba(255,255,255,.035) 0 2px, transparent 2px 18px),
      linear-gradient(180deg, #754522, #3b2416);
    transform: rotateX(55deg);
    box-shadow: 0 -18px 30px rgba(0,0,0,.2);
}

.field-row {
    position:absolute;
    bottom:100px;
    left:8%;
    right:8%;
    height:32px;
    transform: rotateX(45deg);
    background:linear-gradient(180deg,#8d592c,#563017);
    border-radius:50%;
    opacity:.7;
}

.pipe {
    position:absolute;
    left:8%;
    right:8%;
    bottom:128px;
    height:13px;
    border-radius:20px;
    background:linear-gradient(180deg,#3d8f79,#155544);
    box-shadow:0 5px 12px rgba(0,0,0,.3);
    z-index:4;
}

.pipe:before {
    content:"";
    position:absolute;
    left:12%;
    right:12%;
    top:4px;
    height:3px;
    background:rgba(181,255,224,.38);
    border-radius:10px;
}

.water {
    position:absolute;
    width:7px;
    height:7px;
    border-radius:50%;
    background:#69dfff;
    box-shadow:0 0 10px #42cfff;
    z-index:8;
    animation: drip 1.35s linear infinite;
}

.w1 {left:18%;bottom:122px;animation-delay:.1s}
.w2 {left:31%;bottom:122px;animation-delay:.45s}
.w3 {left:45%;bottom:122px;animation-delay:.8s}
.w4 {left:59%;bottom:122px;animation-delay:1.05s}
.w5 {left:74%;bottom:122px;animation-delay:.25s}

@keyframes drip {
    0% {transform:translateY(0) scale(.7); opacity:0}
    15% {opacity:1}
    100% {transform:translateY(90px) scale(.95); opacity:0}
}

/* 3D plant */
.plant {
    position:absolute;
    left:50%;
    bottom:130px;
    width:180px;
    height:270px;
    transform:translateX(-50%);
    z-index:6;
}

.stem {
    position:absolute;
    width:14px;
    height:170px;
    left:83px;
    bottom:12px;
    border-radius:12px;
    background:linear-gradient(90deg,#124d2e,#43b65e,#176a37);
    box-shadow: 7px 0 12px rgba(0,0,0,.2);
}

.leaf {
    position:absolute;
    width:92px;
    height:48px;
    border-radius:100% 0 100% 0;
    background:linear-gradient(135deg,#8bf26f,#198c45 65%,#0b542b);
    box-shadow:inset -10px -8px 15px rgba(0,0,0,.16), 0 9px 14px rgba(0,0,0,.18);
}
.leaf:after {
    content:"";
    position:absolute;
    width:70%;
    height:2px;
    background:rgba(225,255,211,.45);
    left:14%;
    top:52%;
    transform:rotate(-17deg);
}
.l1 {left:1px;top:95px;transform:rotate(22deg)}
.l2 {right:1px;top:75px;transform:scaleX(-1) rotate(19deg)}
.l3 {left:12px;top:42px;transform:rotate(37deg) scale(.78)}
.l4 {right:9px;top:24px;transform:scaleX(-1) rotate(30deg) scale(.78)}

.flower {
    position:absolute;
    left:61px;
    top:0;
    width:60px;
    height:60px;
    border-radius:50%;
    background:
      radial-gradient(circle at 50% 50%, #ffe48b 0 12px, transparent 13px),
      conic-gradient(#ff83c8 0 14%, transparent 14% 18%, #ff83c8 18% 32%, transparent 32% 36%, #ff83c8 36% 50%, transparent 50% 54%, #ff83c8 54% 68%, transparent 68% 72%, #ff83c8 72% 86%, transparent 86% 90%, #ff83c8 90% 100%);
    filter:drop-shadow(0 7px 8px rgba(0,0,0,.22));
}

.plant-tag {
    position:absolute;
    left:50%;
    bottom:36px;
    transform:translateX(-50%);
    padding:7px 12px;
    border-radius:12px;
    background:rgba(5,24,17,.75);
    border:1px solid rgba(143,255,181,.18);
    color:#dffff0;
    font-size:12px;
    z-index:10;
    backdrop-filter:blur(8px);
}

.sensor {
    position:absolute;
    right:14%;
    bottom:132px;
    width:34px;
    height:66px;
    border-radius:10px;
    background:linear-gradient(180deg,#d9f3e7,#6aa994);
    border:3px solid #1c4f40;
    z-index:7;
    box-shadow:0 8px 15px rgba(0,0,0,.3);
}
.sensor:before {
    content:"";
    position:absolute;
    width:9px;height:9px;
    border-radius:50%;
    background:#42e68a;
    left:10px;top:10px;
    box-shadow:0 0 12px #42e68a;
}
.sensor:after {
    content:"SOIL";
    position:absolute;
    font-size:7px;
    font-weight:800;
    color:#123b2e;
    left:6px;
    bottom:9px;
}

.scene-label {
    position:absolute;
    left:24px;
    top:24px;
    z-index:20;
}
.scene-label strong {
    display:block;
    font-size:20px;
}
.scene-label span {
    color:#a9c9b7;
    font-size:12px;
}

.status-pill {
    position:absolute;
    right:24px;
    top:24px;
    z-index:20;
    padding:9px 13px;
    border-radius:999px;
    background:rgba(42, 211, 111,.12);
    border:1px solid rgba(42, 211, 111,.25);
    color:#7ff0a5;
    font-size:12px;
    font-weight:700;
}

.card {
    background:rgba(14, 36, 27, .78);
    border:1px solid rgba(147,255,187,.12);
    border-radius:20px;
    padding:18px;
    min-height:105px;
}

.metric-label {
    color:#8fac9c;
    font-size:12px;
    margin-bottom:5px;
}
.metric-value {
    font-size:27px;
    font-weight:800;
}

.section-title {
    font-size:24px;
    font-weight:800;
    margin:14px 0 10px;
}

div[data-testid="stButton"] button {
    border-radius:14px;
    font-weight:700;
}

[data-testid="stMetric"] {
    background:rgba(17,42,31,.65);
    border:1px solid rgba(147,255,187,.10);
    padding:14px;
    border-radius:16px;
}

@media (max-width: 800px) {
    .scene {height:390px}
    .plant {transform:translateX(-50%) scale(.8); transform-origin:bottom center;}
    .sensor {right:7%}
    .pipe {left:4%;right:4%}
}
</style>
""", unsafe_allow_html=True)

# ---------------------- DATA / MODEL ---------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

model = load_model()
metrics = load_metrics()

# ---------------------- SIDEBAR --------------------------
st.sidebar.markdown("## 🌱 AgriFlow AI")
st.sidebar.caption("3D Smart Irrigation Intelligence")
page = st.sidebar.radio(
    "Navigation",
    ["🌾 Irrigation Control", "📊 Model Insights", "ℹ️ About Project"]
)
st.sidebar.divider()
st.sidebar.success("● AI System Online")
st.sidebar.caption("Educational prototype • LearnDepth Academy Capstone")

# ---------------------- HERO -----------------------------
st.markdown("""
<div class="hero">
  <div class="badge">● SMART FARMING • AI + ML</div>
  <h1 class="hero-title">AgriFlow AI</h1>
  <div class="hero-sub">
    Intelligent irrigation requirement classification with a live 3D field visualization.
    Adjust the field conditions and let the model assess irrigation need.
  </div>
</div>
""", unsafe_allow_html=True)

if page == "🌾 Irrigation Control":
    left, right = st.columns([1.05, 1.65], gap="large")

    with left:
        st.markdown('<div class="section-title">🌱 Field Conditions</div>', unsafe_allow_html=True)

        soil = st.slider("Soil Moisture (%)", 0.0, 100.0, 32.0, 1.0)
        temp = st.slider("Temperature (°C)", 0.0, 50.0, 31.0, 0.5)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 48.0, 1.0)
        crop_stage = st.selectbox(
            "Crop Stage",
            ["Seedling", "Vegetative", "Flowering", "Fruiting", "Maturity"]
        )
        rainfall = st.slider("Recent Rainfall (mm)", 0.0, 100.0, 4.0, 1.0)

        input_df = pd.DataFrame([{
            "soil_moisture": soil,
            "temperature": temp,
            "humidity": humidity,
            "crop_stage": crop_stage,
            "recent_rainfall": rainfall
        }])

        # The trained model may return numeric labels (0/1) or text labels
        # such as "Yes"/"No". Normalize both formats safely.
        raw_prediction = model.predict(input_df)[0]
        prediction_text = str(raw_prediction).strip().lower()

        if prediction_text in {"1", "yes", "true", "required", "irrigation required"}:
            prediction = 1
        elif prediction_text in {"0", "no", "false", "not required", "irrigation not required"}:
            prediction = 0
        else:
            # Fallback for unexpected labels: use the model's class ordering
            # instead of trying to cast a text label directly to int.
            classes = getattr(model, "classes_", [])
            prediction = int(raw_prediction == classes[-1]) if len(classes) else 0

        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(getattr(model, "classes_", []))

            # Find the probability belonging to the positive irrigation class.
            positive_indexes = [
                i for i, cls in enumerate(classes)
                if str(cls).strip().lower() in {
                    "1", "yes", "true", "required", "irrigation required"
                }
            ]
            if positive_indexes:
                probability = float(probabilities[positive_indexes[0]])
            elif len(probabilities) == 2:
                # For a conventional binary 0/1 model, the second class is positive.
                probability = float(probabilities[1])

        if prediction == 1:
            label = "IRRIGATION REQUIRED"
            status_text = "Water flow recommended"
            status_class = "rgba(42,211,111,.18)"
        else:
            label = "IRRIGATION NOT REQUIRED"
            status_text = "Field moisture looks sufficient"
            status_class = "rgba(91,160,255,.16)"

        st.markdown(
            f"""
            <div style="
                margin-top:16px;padding:18px;border-radius:18px;
                background:{status_class};
                border:1px solid rgba(255,255,255,.10);">
                <div style="font-size:12px;color:#9ab8a7;">AI DECISION</div>
                <div style="font-size:23px;font-weight:800;margin-top:4px;">{label}</div>
                <div style="font-size:12px;color:#9ab8a7;margin-top:5px;">{status_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if probability is not None:
            st.progress(probability, text=f"Irrigation probability: {probability*100:.1f}%")

    with right:
        irrigation_class = "Watering active" if prediction == 1 else "Monitoring"
        water_display = "FLOWING" if prediction == 1 else "STANDBY"

        # Repeat water streams when irrigation is predicted.
        water_html = ""
        if prediction == 1:
            water_html = """
              <span class="water w1"></span><span class="water w2"></span>
              <span class="water w3"></span><span class="water w4"></span>
              <span class="water w5"></span>
            """

        st.markdown(
            f"""
            <div class="scene">
              <div class="sun"></div>
              <div class="cloud"></div>

              <div class="scene-label">
                <strong>3D Smart Field</strong>
                <span>Sensor-driven irrigation simulation</span>
              </div>

              <div class="status-pill">● {irrigation_class}</div>

              <div class="ground"></div>
              <div class="field-row"></div>
              <div class="pipe"></div>
              {water_html}

              <div class="plant">
                <div class="flower"></div>
                <div class="leaf l1"></div>
                <div class="leaf l2"></div>
                <div class="leaf l3"></div>
                <div class="leaf l4"></div>
                <div class="stem"></div>
              </div>

              <div class="sensor"></div>
              <div class="plant-tag">🌱 Crop • {crop_stage}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        if prediction == 1:
            st.info("💧 **Irrigation simulation is active:** water droplets are flowing from the smart pipe toward the crop.")
        else:
            st.success("🌤️ **Irrigation is on standby:** the model currently does not recommend watering.")

    st.markdown('<div class="section-title">📡 Live Field Snapshot</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f'<div class="card"><div class="metric-label">SOIL MOISTURE</div><div class="metric-value">{soil:.0f}%</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="card"><div class="metric-label">TEMPERATURE</div><div class="metric-value">{temp:.1f}°C</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="card"><div class="metric-label">HUMIDITY</div><div class="metric-value">{humidity:.0f}%</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="card"><div class="metric-label">RAINFALL</div><div class="metric-value">{rainfall:.0f} mm</div></div>', unsafe_allow_html=True)

elif page == "📊 Model Insights":
    st.markdown('<div class="section-title">📊 Model Performance</div>', unsafe_allow_html=True)
    a, b, c, d = st.columns(4)

    metric_map = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1_score"),
    ]

    for col, (name, key) in zip([a,b,c,d], metric_map):
        with col:
            value = float(metrics.get(key, 0))
            st.metric(name, f"{value*100:.1f}%")

    st.markdown("---")
    st.markdown("### 🔍 Input Features")
    st.dataframe(pd.DataFrame({
        "Feature": ["Soil Moisture", "Temperature", "Humidity", "Crop Stage", "Recent Rainfall"],
        "Type": ["Numeric", "Numeric", "Numeric", "Categorical", "Numeric"],
        "Purpose": [
            "Represents water availability in soil",
            "Environmental heat condition",
            "Atmospheric moisture",
            "Growth-stage context",
            "Recent natural water input"
        ]
    }), use_container_width=True, hide_index=True)

    st.caption("Model: Decision Tree classifier with preprocessing pipeline. Metrics are based on the included educational dataset.")

else:
    st.markdown('<div class="section-title">ℹ️ About AgriFlow AI</div>', unsafe_allow_html=True)
    st.markdown("""
    **AgriFlow AI** is an educational smart-agriculture prototype for the
    LearnDepth Academy Track 1 capstone problem **“Irrigation Requirement Classification.”**

    The system takes:
    - Soil moisture
    - Temperature
    - Humidity
    - Crop stage
    - Recent rainfall

    and predicts whether irrigation is likely to be required.

    ### 🧠 ML Pipeline
    1. Input validation
    2. Missing-value handling
    3. Numerical scaling
    4. Categorical encoding
    5. Decision Tree classification
    6. Prediction + probability
    7. 3D irrigation visualization

    ### ⚠️ Educational note
    The bundled project uses a reproducible synthetic educational dataset because
    the capstone problem statement specifies expected inputs but does not provide
    a specific dataset. This prototype should not be used as a real agricultural
    irrigation controller without field validation and domain expertise.
    """)

st.markdown("""
<div style="margin-top:30px;text-align:center;color:#6f8d7d;font-size:11px;">
AgriFlow AI • Smart Irrigation Classification • Educational Prototype
</div>
""", unsafe_allow_html=True)
