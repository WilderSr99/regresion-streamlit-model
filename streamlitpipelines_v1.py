import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from joblib import load

# -----------------------------
# CONFIGURACIÓN DE LA APP
# -----------------------------
st.set_page_config(
    page_title="Predicción de Profit - Analytics",
    page_icon="📈",
    layout="wide" # Cambiado a wide para que las gráficas luzcan mejor
)

# -----------------------------
# CARGAR MODELO (CACHE)
# -----------------------------
@st.cache_resource
def load_model():
    # Asegúrate de que el archivo esté en la misma carpeta
    return load("Modelopipeline.joblib")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# -----------------------------
# TÍTULO Y DESCRIPCIÓN
# -----------------------------
st.title("📈 Dashboard de Predicción de Profit")
st.markdown("""
Esta aplicación utiliza un modelo de **Machine Learning (Pipeline)** para estimar el rendimiento económico 
basado en la inversión en diferentes áreas de la empresa.
""")

# -----------------------------
# SIDEBAR - ENTRADA DE DATOS
# -----------------------------
st.sidebar.header("🕹️ Panel de Control")

rd_spend = st.sidebar.slider("R&D Spend", 0.0, 170000.0, 75000.0)
administration = st.sidebar.slider("Administration", 0.0, 190000.0, 120000.0)
marketing_spend = st.sidebar.slider("Marketing Spend", 0.0, 480000.0, 200000.0)
state = st.sidebar.selectbox("Estado de Operación", ["New York", "California", "Florida"])

# -----------------------------
# LÓGICA DE PREDICCIÓN
# -----------------------------
input_data = pd.DataFrame({
    "R&D Spend": [rd_spend],
    "Administration": [administration],
    "Marketing Spend": [marketing_spend],
    "State": [state]
})

# Realizar la predicción automáticamente o mediante botón
prediction = model.predict(input_data)[0]

# -----------------------------
# VISUALIZACIÓN DE RESULTADOS (LAYOUT)
# -----------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Métricas de Entrada")
    st.dataframe(input_data.T.rename(columns={0: "Valor"}))
    
    st.metric(label="💰 Profit Estimado", value=f"${prediction:,.2f}")

with col2:
    st.subheader("Análisis de Composición de Inversión")
    # Gráfico de pastel interactivo con Plotly
    fig_pie = px.pie(
        names=["R&D", "Admin", "Marketing"],
        values=[rd_spend, administration, marketing_spend],
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.RdBu,
        title="Distribución del Gasto"
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------
# SECCIÓN DE GRÁFICAS INTERACTIVAS
# -----------------------------
st.markdown("---")
st.subheader("📊 Simulación de Escenarios")

# Crear datos sintéticos para ver la tendencia de R&D vs Profit
rd_range = np.linspace(0, 170000, 50)
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
    title="Impacto del R&D en el Profit (Manteniendo otros valores fijos)",
    labels={"Predicted Profit": "Profit ($)", "R&D Spend": "Inversión en I+D ($)"},
    template="plotly_dark"
)
# Añadir un punto que represente la predicción actual
fig_trend.add_scatter(x=[rd_spend], y=[prediction], mode='markers', name='Punto Actual', marker=dict(size=12, color='yellow'))

st.plotly_chart(fig_trend, use_container_width=True)

# -----------------------------
# BOTÓN RESET (CORREGIDO)
# -----------------------------
if st.sidebar.button("🔄 Resetear valores"):
    st.rerun()

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption(f"Entorno: VS Code | Localhost:8501 | ML Engine: Scikit-Learn Pipeline")
st.caption("Desarrollado por: [WilderSr99] - Curso ENEI 2026: ML en Producción Web")