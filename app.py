import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Page Configuration
st.set_page_config(page_title="FireGuard AI: Forest Fire Detection", layout="wide")

# Load Assets
@st.cache_resource
def load_assets():
    model = joblib.load('models/mlp_pso.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    features = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Temp_RH_Index', 'Wind_Rain_Interaction']

    # Dynamically determine the number of classes from the model
    n_classes = model.classes_.shape[0]

    # IMPORTANT: Cast classes to standard Python ints to avoid numpy.int64 keys in dict
    model_classes = [int(c) for c in model.classes_]

    if n_classes == 2:
        # Binary Algerian Dataset mapping
        # Map the actual values found in model.classes_ to names
        labels_map = {0: 'No Fire', 3: 'High Risk'}
        class_labels = [labels_map.get(c, f"Class {c}") for c in model_classes]
    else:
        # Multi-class UCI mapping
        labels_map = {0: 'No Fire', 1: 'Low Risk', 2: 'Moderate', 3: 'High Risk'}
        class_labels = [labels_map.get(c, f"Class {c}") for c in model_classes]

except Exception as e:
    st.error(f"Error loading models: {e}. Please run the pipeline first.")
    st.stop()


# UI Header
st.title("🔥 FireGuard AI: Precision Forest Fire Monitoring")
st.markdown("### Advanced PSO-MLP Prediction System (Algerian Dataset)")
st.divider()

# Main Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Live Predictor", "🗺️ Regional Monitoring", "⚙️ System Info"])

with tab1:
    st.header("Real-Time Risk Prediction")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Environmental Parameters")
        # Interaction features are calculated internally
        temp = st.slider("Temperature (°C)", 10.0, 50.0, 30.0)
        rh = st.slider("Relative Humidity (%)", 10.0, 100.0, 50.0)
        ws = st.slider("Wind Speed (km/h)", 0.0, 50.0, 15.0)
        rain = st.slider("Rain (mm)", 0.0, 20.0, 0.0)
        ffmc = st.slider("FFMC", 50.0, 100.0, 80.0)
        dmc = st.slider("DMC", 0.0, 30.0, 10.0)
        dc = st.slider("DC", 0.0, 200.0, 50.0)
        isi = st.slider("ISI", 0.0, 20.0, 5.0)
        bui = st.slider("BUI", 0.0, 100.0, 20.0)
        fwi = st.slider("FWI", 0.0, 50.0, 10.0)

        # Calculate interaction features
        temp_rh = temp * (100 - rh)
        wind_rain = ws * rain

        input_data = np.array([[temp, rh, ws, rain, ffmc, dmc, dc, isi, bui, fwi, temp_rh, wind_rain]])

    with col2:
        st.subheader("AI Analysis")
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        proba = model.predict_proba(scaled_input)[0]
        confidence = np.max(proba) * 100

        # Get the index of the prediction within the model's classes_ array
        # This is the critical fix: model.predict returns the class value (e.g. 0 or 3),
        # but we need the index (0 or 1) to access class_labels.
        class_idx = np.where(model.classes_ == prediction)[0][0]
        risk_label = class_labels[class_idx]

        # Dynamic visual alert based on label text
        if risk_label == 'High Risk':
            st.error(f"## ALERT: {risk_label}")
        elif risk_label == 'No Fire':
            st.success(f"## STATUS: {risk_label}")
        else:
            st.warning(f"## WARNING: {risk_label}")

        st.metric("Prediction Confidence", f"{confidence:.2f}%")

        # Live Probability Chart
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(x=class_labels, y=proba, palette="YlOrRd", ax=ax)
        ax.set_title("AI Probability Distribution")
        ax.set_ylabel("Probability")
        st.pyplot(fig)

with tab2:
    st.header("Regional Risk Monitoring")
    st.markdown("Displaying simulated risk levels across the monitored regional grid.")
    try:
        df_table = pd.read_csv('outputs/regional_risk_table.csv')
        # Use a color-coded dataframe for better UI
        def color_risk(val):
            color = 'red' if val == 'High Risk' else 'green' if val == 'No Fire' else 'orange'
            return f'color: {color}'

        # Use map() instead of applymap() as applymap is deprecated in newer pandas versions
        st.table(df_table.head(25).style.map(color_risk, subset=['Risk_Class']))

    except:
        st.error("Regional data not found. Please run Phase 7.")

with tab3:
    st.header("System Specifications")
    st.markdown("""
    - **Model Architecture:** PSO-Optimized Multi-Layer Perceptron (MLP)
    - **Training Data:** Algerian Forest Fires Dataset
    - **Optimization:** Particle Swarm Optimization (PSO)
    - **Sensing:** Meteorological inputs + Derived interaction indices
    - **Accuracy:** ~98% on test set
    """)

st.sidebar.markdown("---")
st.sidebar.info("Developed for High-Precision Forest Fire Detection. Dataset: Algerian Forest Fires.")
