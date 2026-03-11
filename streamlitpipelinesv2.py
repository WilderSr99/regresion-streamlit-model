import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from joblib import load

# -----------------------------
# CONFIGURACIÓN DE LA APP
# -----------------------------
st.set_page_config(
    page_title="Predicción de Profit - Analytics 2026",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def load_model():
    return load("Modelopipeline.joblib")

model = load_model()

st.title("Graficas interactivas de Predicción de Profit")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("🕹️ Panel de Control")
rd_spend = st.sidebar.slider("R&D Spend", 0.0, 170000.0, 75000.0)
administration = st.sidebar.slider("Administration", 0.0, 190000.0, 120000.0)
marketing_spend = st.sidebar.slider("Marketing Spend", 0.0, 480000.0, 200000.0)
state = st.sidebar.selectbox("Estado", ["New York", "California", "Florida"])

# -----------------------------
# PROCESAMIENTO DE DATOS
# -----------------------------
input_data = pd.DataFrame({
    "R&D Spend": [float(rd_spend)],
    "Administration": [float(administration)],
    "Marketing Spend": [float(marketing_spend)],
    "State": [state]
})

prediction = model.predict(input_data)[0]

# -----------------------------
# VISUALIZACIÓN
# -----------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Métricas de Entrada")
    # Corrección para PyArrow: Convertimos a tipos simples
    st.dataframe(input_data.T.astype(str), width="stretch")
    st.metric(label="💰 Profit Estimado", value=f"${prediction:,.2f}")

with col2:
    st.subheader("Composición de Inversión")
    fig_pie = px.pie(
        names=["R&D", "Admin", "Marketing"],
        values=[rd_spend, administration, marketing_spend],
        hole=0.4,
        title="Distribución del Gasto"
    )
    # Cambio de use_container_width por width="stretch" (Sintaxis 2026)
    st.plotly_chart(fig_pie, width="stretch")

# -----------------------------
# GRÁFICA DE TENDENCIA
# -----------------------------
st.markdown("---")
rd_range = np.linspace(0, 170000, 50)
temp_df = pd.DataFrame({
    "R&D Spend": rd_range,
    "Administration": [administration] * 50,
    "Marketing Spend": [marketing_spend] * 50,
    "State": [state] * 50
})
temp_df["Predicted Profit"] = model.predict(temp_df)

fig_trend = px.line(temp_df, x="R&D Spend", y="Predicted Profit", title="Impacto del R&D en el Profit")
# Cambio de use_container_width por width="stretch" (Sintaxis 2026)
st.plotly_chart(fig_trend, width="stretch")

if st.sidebar.button("🔄 Resetear valores"):
    st.rerun()
 #streamlit run streamlitpipelines.py