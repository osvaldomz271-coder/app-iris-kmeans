import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Configuración de la Página Streamlit (Estética y Estándares Profesionales)
st.set_page_config(
    page_title="Clasificador K-NN (Dataset Iris & K-Means)",
    page_icon="🌸",
    layout="centered"
)

# Título y Descripción Metodológica
st.title("🌸 Aplicación Web Interactiva de Predicción con K-NN")
st.markdown("""
Esta aplicación web implementa un modelo de aprendizaje supervisado **K-Nearest Neighbors (K-NN)** 
entrenado sobre las pseudo-etiquetas descubiertas mediante segmentación no supervisada (**K-Means**) 
utilizando el problema canónico **Iris** de Scikit-Learn.
""")

# Carga de Artefactos Serializados con caché para optimizar rendimiento
@st.cache_resource
def load_artifacts():
    model = joblib.load('knn_iris_model.joblib')
    scaler = joblib.load('scaler_iris.joblib')
    return model, scaler

try:
    knn_model, scaler_obj = load_artifacts()
except Exception as e:
    st.error(f"Error al cargar los artefactos del modelo: {e}")
    st.stop()

# Panel Lateral (Sidebar) para Ingreso de Nuevas Muestras
st.sidebar.header("🎛️ Parámetros Morfológicos de Entrada")
st.sidebar.markdown("Ingrese las dimensiones de la muestra botánica:")

sepal_length = st.sidebar.slider("Longitud del Sépalo (cm)", 4.0, 8.0, 5.8, 0.1)
sepal_width = st.sidebar.slider("Ancho del Sépalo (cm)", 2.0, 4.5, 3.0, 0.1)
petal_length = st.sidebar.slider("Longitud del Pétalo (cm)", 1.0, 7.0, 4.3, 0.1)
petal_width = st.sidebar.slider("Ancho del Pétalo (cm)", 0.1, 2.5, 1.3, 0.1)

# Procesamiento de la Muestra del Usuario
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# ⚠️ CRÍTICO: Aplicar el mismo transformador de escala (StandardScaler) guardado en el entrenamiento
input_scaled = scaler_obj.transform(input_data)

# Botón de Predicción Activa
if st.button("🚀 Predecir Clúster / Categoría", type="primary"):
    prediccion_cluster = knn_model.predict(input_scaled)[0]
    probabilidades = knn_model.predict_proba(input_scaled)[0]

    # Diccionario de interpretación gerencial / operativa de los clústeres
    descripciones_clusters = {
        0: "Clúster 0: Perfil Morfológico Compacto (Especie Setosa - Alta Separación)",
        1: "Clúster 1: Perfil Intermedio de Transición (Especie Versicolor)",
        2: "Clúster 2: Perfil de Dimensiones Extensas (Especie Virginica)"
    }

    st.success("### Resultado de la Inferencia:")
    st.markdown(f"**Categoría Asignada:** `{descripciones_clusters.get(prediccion_cluster, 'Clúster ' + str(prediccion_cluster))}`")

    # Muestra de probabilidades asociadas a los vecinos más cercanos
    st.markdown("#### 📊 Distribución de Probabilidades por Vecindad (K-NN):")
    df_probs = pd.DataFrame({
        'Clúster': ['Clúster 0', 'Clúster 1', 'Clúster 2'],
        'Probabilidad': probabilidades
    }).set_index('Clúster')

    st.bar_chart(df_probs)

    st.info("Nota técnica: La inferencia utiliza un modelo K-NN con métrica euclidiana optimizada sobre las fronteras de Voronoi aprendidas de K-Means.")
