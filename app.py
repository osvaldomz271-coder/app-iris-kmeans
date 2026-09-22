import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Configuración de la página
st.set_page_config(
    page_title="Predicción Logística de E-commerce", 
    page_icon="📦", 
    layout="centered"
)

st.title("📦 Predicción de Envíos en E-commerce (A Tiempo vs. Retraso)")
st.markdown("""
Aplicación web interactiva conectada al modelo definitivo **Random Forest** entrenado con las 
variables logísticas y comerciales del dataset de comercio electrónico (ICYA3004 - Universidad de los Andes).
""")

@st.cache_resource
def load_ecom_artifacts():
    model = joblib.load('rf_ecom_model.joblib')
    scaler = joblib.load('scaler_ecom.joblib')
    return model, scaler

# Carga segura de artefactos
try:
    model, scaler = load_ecom_artifacts()
except Exception as e:
    st.error(f"Error al cargar los modelos. Asegúrate de tener 'rf_ecom_model.joblib' y 'scaler_ecom.joblib' en el directorio. Detalle: {e}")

st.sidebar.header("🎛️ Parámetros Operativos del Envío")
customer_care_calls = st.sidebar.slider("Llamadas de Atención al Cliente", 2, 7, 4)
customer_rating = st.sidebar.slider("Calificación del Cliente (1-5)", 1, 5, 3)
cost_of_product = st.sidebar.number_input("Costo del Producto ($)", 10.0, 300.0, 150.0)
prior_purchases = st.sidebar.slider("Compras Previas", 2, 10, 3)
discount_offered = st.sidebar.slider("Descuento Ofrecido ($)", 0, 65, 10)
weight_in_gms = st.sidebar.number_input("Peso del Paquete (gramos)", 1000, 8000, 4000)

if st.button("🚀 Predecir Estado del Envío", type="primary"):
    # Vector de entrada estructurado para el modelo
    input_data = np.array([[customer_care_calls, customer_rating, cost_of_product, prior_purchases, discount_offered, weight_in_gms]])
    input_scaled = scaler.transform(input_data)
    
    # Depuración visual: Verificar que los datos escalados cambian al mover los sliders
    st.info("🔍 **Valores escalados ingresados al modelo:**")
    st.write(input_scaled)

    pred = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    
    # Depuración visual: Mostrar probabilidades brutas del modelo [Clase 0, Clase 1]
    st.write(f"📊 **Probabilidades brutas devueltas por el modelo [Retrasado (0), A Tiempo (1)]:** `[{proba[0]:.4f}, {proba[1]:.4f}]`")

    st.success("### Resultado de la Inferencia Logística:")
    if pred == 1:
        st.warning("⚠️ **Predicción:** El envío corre el **riesgo de retrasarse** (No llegará a tiempo según el SLA).")
    else:
        st.info("✅ **Predicción:** El envío **sí llegará a tiempo** a su destino.")

    st.markdown("#### 📊 Gráfica de Probabilidades Asociadas:")
    
    # Estructura limpia y compatible para la gráfica de barras en Streamlit
    df_res = pd.DataFrame({
        'Estado del Envío': ['Retrasado (0)', 'A Tiempo (1)'],
        'Probabilidad': [float(proba[0]), float(proba[1])]
    }).set_index('Estado del Envío')
    
    # Visualización estable de la gráfica
    st.bar_chart(df_res)
    
    # Desglose numérico complementario
    st.write("Detalle numérico de las probabilidades:")
    st.dataframe(df_res.style.format("{:.4f}"))