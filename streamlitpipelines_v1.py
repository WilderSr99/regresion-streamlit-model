import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from joblib import load

# -----------------------------
# CONFIGURACIÓN DE LA APP
# -----------------------------
st.set_page_config(
    page_title="Predicción de Profit - Entrada Manual",
    page_icon="💰",
    layout="wide"
)

@st.cache_resource
def load_model():
    return load("Modelopipeline.joblib")

model = load_model()

st.title("📈 Dashboard de Predicción de Profit")
st.markdown("Introduce los montos de inversión de forma manual para obtener una predicción exacta.")

# -----------------------------
# SIDEBAR - ENTRADA MANUAL DE DATOS
# -----------------------------
st.sidebar.header("📝 Datos de Inversión")

# Usamos number_input para permitir escritura manual exacta
rd_spend = st.sidebar.number_input(
    "Inversión en R&D", 
    min_value=0.0, 
    max_value=200000.0, 
    value=75000.0,
    step=500.0,
    help="Ingrese el monto exacto invertido en Investigación y Desarrollo."
)

administration = st.sidebar.number_input(
    "Gastos Administrativos", 
    min_value=0.0, 
    max_value=250000.0, 
    value=120000.0,
    step=500.0
)

marketing_spend = st.sidebar.number_input(
    "Inversión en Marketing", 
    min_value=0.0, 
    max_value=500000.0, 
    value=200000.0,
    step=1000.0
)

state = st.sidebar.selectbox(
    "Estado de Operación", 
    ["New York", "California", "Florida"]
)

# -----------------------------
# LÓGICA DE NEGOCIO
# -----------------------------
input_data = pd.DataFrame({
    "R&D Spend": [float(rd_spend)],
    "Administration": [float(administration)],
    "Marketing Spend": [float(marketing_spend)],
    "State": [state]
})

# Predicción
prediction = model.predict(input_data)[0]

# -----------------------------
# DISEÑO DE RESULTADOS
# -----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📌 Resumen de Entradas")
    # Mostramos los datos de forma limpia
    st.table(input_data.T.rename(columns={0: "Monto ($)"}))
    
    st.metric(
        label="💰 Profit Estimado", 
        value=f"${prediction:,.2f}",
        delta=f"{((prediction/rd_spend)-1)*100:.1f}% ROI vs R&D" if rd_spend > 0 else None
    )

with col2:
    st.subheader("📊 Distribución del Presupuesto")
    fig_pie = px.pie(
        names=["R&D", "Admin", "Marketing"],
        values=[rd_spend, administration, marketing_spend],
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    # Sintaxis 2026: width="stretch"
    st.plotly_chart(fig_pie, width="stretch")

# -----------------------------
# ANÁLISIS DE SENSIBILIDAD
# -----------------------------
st.markdown("---")
st.subheader("📉 Análisis Proyectado")

# Generamos una curva basada en tu entrada manual de R&D
rd_range = np.linspace(0, rd_spend * 1.5, 50)
temp_df = pd.DataFrame({
    "R&D Spend": rd_range,
    "Administration": [administration] * 50,
    "Marketing Spend": [marketing_spend] * 50,
    "State": [state] * 50
})
temp_df["Predicted Profit"] = model.predict(temp_df)

fig_trend = px.line(
    temp_df, 
    x="R&D Spend", 
    y="Predicted Profit", 
    title="¿Qué pasaría si varías la inversión en R&D?",
    labels={"Predicted Profit": "Profit Esperado ($)", "R&D Spend": "Inversión R&D ($)"}
)
fig_trend.add_scatter(x=[rd_spend], y=[prediction], mode='markers', name='Tu Valor Actual', marker=dict(size=15, color='red'))

st.plotly_chart(fig_trend, width="stretch")

# -----------------------------
# BOTÓN DE REINICIO
# -----------------------------
if st.sidebar.button("🔄 Limpiar Formulario"):
    st.rerun()

st.markdown("---")
st.caption("Enei 2026 - Herramienta de Soporte a Decisiones Financieras")

#streamlit run streamlitpipelines_v1.py