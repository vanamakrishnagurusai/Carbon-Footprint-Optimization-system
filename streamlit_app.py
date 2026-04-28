# ────────────────────────────────────────────────────────────
#  Carbon Footprint Optimization — Streamlit Cloud App
# ────────────────────────────────────────────────────────────
"""
Self-contained Streamlit dashboard that loads ML models directly
(no FastAPI backend needed). Ready for Streamlit Cloud deployment.
"""

import os
import sys
import joblib
import numpy as np
import streamlit as st

# ── resolve project root so sibling packages are importable ─
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from utils.helper import calculate_co2, get_optimization_suggestions

# ── paths to saved artefacts ────────────────────────────────
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ── load models (cached so they load only once) ─────────────
@st.cache_resource
def load_models():
    """Load all ML artefacts once and cache them across reruns."""
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    cluster_bundle = joblib.load(os.path.join(MODEL_DIR, "kmeans.pkl"))

    kmeans = cluster_bundle["kmeans"]
    cluster_scaler = cluster_bundle["scaler"]
    cluster_label_map = cluster_bundle["label_map"]

    return model, scaler, label_encoder, kmeans, cluster_scaler, cluster_label_map


model, scaler, label_encoder, kmeans, cluster_scaler, cluster_label_map = load_models()


# ── page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="🌱 Carbon Footprint Optimizer",
    page_icon="🌿",
    layout="centered",
)

st.title("🌱 Carbon Footprint Optimization Dashboard")
st.markdown(
    "Enter device parameters below to **predict energy usage**, "
    "**estimate CO₂ emissions**, and get **optimisation suggestions**."
)

# ── sidebar inputs ──────────────────────────────────────────
st.sidebar.header("⚙️ Device Parameters")

DEVICES = [
    "Agricultural Sensor", "EV Charger", "Edge Gateway",
    "Health Monitor", "Industrial Sensor", "Smart Camera",
    "Smart Lighting", "Smart Meter", "Smart Thermostat",
    "Wearable Device",
]

device      = st.sidebar.selectbox("Device", DEVICES)
temperature = st.sidebar.slider("Temperature (°C)", 10.0, 50.0, 30.0, 0.5)
humidity    = st.sidebar.slider("Humidity (%)", 20.0, 100.0, 60.0, 1.0)
power       = st.sidebar.slider("Power (W)", 50.0, 5000.0, 1500.0, 50.0)
duration    = st.sidebar.slider("Duration (hours)", 0.5, 12.0, 3.0, 0.5)
hour        = st.sidebar.slider("Hour of day (0-23)", 0, 23, 19)
day         = st.sidebar.slider("Day of month (1-31)", 1, 31, 15)

# ── predict button ──────────────────────────────────────────
if st.sidebar.button("🔍 Analyse", use_container_width=True):

    # ── encode device ───────────────────────────────────────
    try:
        device_enc = label_encoder.transform([device])[0]
    except ValueError:
        st.error(f"❌ Unknown device '{device}'.")
        st.stop()

    # ── build feature array ─────────────────────────────────
    features = np.array([[
        device_enc, hour, day,
        temperature, humidity,
        power, duration,
    ]])
    features_scaled = scaler.transform(features)

    # ── predict energy & CO₂ ────────────────────────────────
    predicted_energy = float(model.predict(features_scaled)[0])
    co2 = calculate_co2(predicted_energy)

    # ── cluster assignment ──────────────────────────────────
    cluster_input = np.array([[predicted_energy, duration, hour]])
    cluster_input_scaled = cluster_scaler.transform(cluster_input)
    cluster_id = int(kmeans.predict(cluster_input_scaled)[0])
    cluster_label = cluster_label_map.get(cluster_id, "unknown")

    # ── optimisation suggestions ────────────────────────────
    suggestions = get_optimization_suggestions(
        duration=duration,
        hour=hour,
        energy=predicted_energy,
    )

    # ── display results ─────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Prediction Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("⚡ Predicted Energy", f"{round(predicted_energy, 4)} kWh")
    col2.metric("🌫️ Estimated CO₂", f"{co2} kg")
    col3.metric("🏷️ Cluster", cluster_label.title())

    # Cluster colour badge
    if cluster_label == "efficient":
        st.success("✅ This usage pattern is **efficient**.")
    elif cluster_label == "moderate":
        st.warning("⚠️ This usage pattern is **moderate**.")
    elif cluster_label == "high usage":
        st.error("🔴 This usage pattern is **high usage** — consider optimising.")

    # Suggestions
    st.markdown("---")
    st.subheader("💡 Optimisation Suggestions")
    for i, tip in enumerate(suggestions, 1):
        st.info(f"**{i}.** {tip}")

else:
    st.info("👈 Set device parameters in the sidebar and click **Analyse**.")
