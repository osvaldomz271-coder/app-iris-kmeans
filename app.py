import streamlit as st
import numpy as np
import joblib
import pandas as pd

st.set_page_config(page_title="Predicción Logística de E-commerce", page_icon="📦", layout="centered")

st.title("📦 Predicción de Envíos en E-commerce (A Tiempo vs. Retraso)")
st.markdown("""
Aplicación web interactiva conectada al modelo definitivo **Random Forest** entrenado con las 
variables logísticas y comerciales del dataset de comercio electrónico.
""")

@st.cache_resource
def load_ecom_artifacts():
    model = joblib.load('rf_ecom_model.joblib')
    scaler = joblib.load('scaler_ecom.joblib')
    return model, scaler

model, scaler = load_ecom_artifacts()

st.sidebar.header("🎛️ Parámetros Operativos del Envío")
customer_care_calls = st.sidebar.slider("Llamadas de Atención al Cliente", 2, 7, 4)
customer_rating = st.sidebar.slider("Calificación del Cliente (1-5)", 1, 5, 3)
cost_of_product = st.sidebar.number_input("Costo del Producto ($)", 10.0, 300.0, 150.0)
prior_purchases = st.sidebar.slider("Compras Previas", 2, 10, 3)
discount_offered = st.sidebar.slider("Descuento Ofrecido ($)", 0, 65, 10)
weight_in_gms = st.sidebar.number_input("Peso del Paquete (gramos)", 1000, 8000, 4000)

input_data = np.array([[customer_care_calls, customer_rating, cost_of_product, prior_purchases, discount_offered, weight_in_gms]])
input_scaled = scaler.transform(input_data)

if st.button("🚀 Predecir Estado del Envío", type="primary"):
    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    st.success("### Resultado de la Inferencia Logística:")
    if pred == 1:
        st.warning("⚠️ **Predicción:** El envío corre el **riesgo de retrasarse** (No llegará a tiempo según el SLA).")
    else:
        st.info("✅ **Predicción:** El envío **sí llegará a tiempo** a su destino.")

    st.markdown("#### 📊 Probabilidades Asociadas:")
    df_res = pd.DataFrame({'Estado': ['Retrasado (0)', 'A Tiempo (1)'], 'Probabilidad': proba}).set_index('Estado')
    st.bar_chart(df_res)
