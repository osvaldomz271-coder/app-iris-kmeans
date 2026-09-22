import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Panel Ejecutivo - Predicción Logística E-commerce", 
    page_icon="📦", 
    layout="centered"
)

st.title("📦 Centro de Control Logístico: Predicción de Riesgo de Entrega")
st.markdown("""
Herramienta analítica para la toma de decisiones gerenciales. Evalúa en tiempo real la probabilidad 
de cumplimiento del SLA (Service Level Agreement) basada en machine learning (**Random Forest**).
""")

@st.cache_resource
def load_ecom_artifacts():
    model = joblib.load('rf_ecom_model.joblib')
    scaler = joblib.load('scaler_ecom.joblib')
    return model, scaler

try:
    model, scaler = load_ecom_artifacts()
except Exception as e:
    st.error(f"Error al cargar los artefactos del modelo: {e}")

st.sidebar.header("🎛️ Variables Operativas del Envío")
customer_care_calls = st.sidebar.slider("Llamadas de Atención al Cliente", 2, 7, 4)
customer_rating = st.sidebar.slider("Calificación del Cliente (1-5)", 1, 5, 3)
cost_of_product = st.sidebar.number_input("Costo del Producto ($)", 10.0, 300.0, 150.0)
prior_purchases = st.sidebar.slider("Compras Previas", 2, 10, 3)
discount_offered = st.sidebar.slider("Descuento Ofrecido ($)", 0, 65, 10)
weight_in_gms = st.sidebar.number_input("Peso del Paquete (gramos)", 1000, 8000, 4000)

if st.button("🔍 Evaluar Riesgo del Envío", type="primary"):
    # Procesamiento del vector
    input_data = np.array([[customer_care_calls, customer_rating, cost_of_product, prior_purchases, discount_offered, weight_in_gms]])
    input_scaled = scaler.transform(input_data)

    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    
    # NOTA TÉCNICA CORREGIDA: 
    # proba[0] = Probabilidad de Retraso (Clase 0)
    # proba[1] = Probabilidad de Éxito / A Tiempo (Clase 1)
    prob_retraso = float(proba[0])
    prob_atiempo = float(proba[1])

    st.markdown("---")
    st.subheader("📊 Diagnóstico Gerencial del Envío")

    # Alerta orientada a negocio
    if prob_retraso > 0.5:
        st.error(f"🚨 **ALERTA CRÍTICA DE RETRASO** (Probabilidad de fallo: **{prob_retraso*100:.1f}%**)")
        st.markdown("""
        * **Recomendación para la Empresa:** Este envío presenta alta vulnerabilidad de incumplir el SLA. 
          Se sugiere derivar a una ruta express de última milla o activar una alerta preventiva con el operador logístico.
        """)
    else:
        st.success(f"✅ **ENVÍO VIABLE - BAJO RIESGO** (Probabilidad de éxito a tiempo: **{prob_atiempo*100:.1f}%**)")
        st.markdown("""
        * **Recomendación para la Empresa:** El envío cumple con los patrones operativos estándar. Puede proceder 
          bajo el flujo logístico regular sin incurrir en costos de aceleración adicionales.
        """)

    # Visualización limpia para el consultor
    st.markdown("#### Distribución de Probabilidades del Modelo:")
    df_res = pd.DataFrame({
        'Estado Logístico': ['Riesgo de Retraso (0)', 'A Tiempo / Éxito (1)'],
        'Probabilidad': [prob_retraso, prob_atiempo]
    }).set_index('Estado Logístico')
    
    st.bar_chart(df_res)
    
    # Tabla ejecutiva sin datos técnicos confusos
    st.markdown("##### Resumen de Probabilidades:")
    st.dataframe(df_res.style.format("{:.2%}}"))