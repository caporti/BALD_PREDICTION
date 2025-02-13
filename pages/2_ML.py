import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Cargar el modelo y el escalador
with open("modelo_calvicie.pkl", "rb") as f:
    data = pickle.load(f)

gb = data["modelo"]
sc = data["escalador"]

# Columnas esperadas por el modelo
columnas_modelo = sc.feature_names_in_

# Mapas de conversión español-inglés
map_stress = {"Bajo": 0, "Moderado": 1, "Alto": 2}  # Convertimos a números
map_yes_no = {"Sí": 1, "No": 0}

map_medical_conditions = {
    "Eccema": "Eczema", "Psoriasis": "Psoriasis", "Dermatitis": "Dermatitis",
    "Dermatitis Seborreica": "Seborrheic Dermatitis", "Tiña": "Ringworm",
    "Infección del cuero cabelludo": "Scalp Infection", "Alopecia Areata": "Alopecia Areata",
    "Alopecia Androgenética": "Androgenetic Alopecia", "Problemas de tiroides": "Thyroid Problems",
    "Dermatosis": "Dermatosis", "Sin datos": "No Data"
}

map_nutritional_deficiencies = {
    "Deficiencia de Vitamina A": "Vitamin A Deficiency", "Deficiencia de Vitamina D": "Vitamin D Deficiency",
    "Deficiencia de Biotina": "Biotin Deficiency", "Deficiencia de Vitamina E": "Vitamin E deficiency",
    "Deficiencia de Magnesio": "Magnesium deficiency", "Deficiencia de Selenio": "Selenium deficiency",
    "Deficiencia de Zinc": "Zinc Deficiency", "Deficiencia de Proteínas": "Protein deficiency",
    "Deficiencia de Ácidos Grasos Omega-3": "Omega-3 fatty acids", "Deficiencia de Hierro": "Iron deficiency",
    "Sin datos": "No Data"
}

map_medications = {
    "Antibióticos": "Antibiotics", "Crema antifúngica": "Antifungal Cream",
    "Medicación para la presión arterial": "Blood Pressure Medication",
    "Medicación para el corazón": "Heart Medication", "Inmunomoduladores": "Immunomodulators",
    "Esteroides": "Steroids", "Antidepresivos": "Antidepressants", "Rogaine": "Rogaine",
    "Accutane": "Accutane", "Quimioterapia": "Chemotherapy", "Sin datos": "No Data"
}

# Función para preprocesar los datos
def preprocesar_datos(datos):
    # Convertir valores categóricos a numéricos
    datos["Stress"] = datos["Stress"].map(map_stress)
    
    columnas_yes_no = ["Genetics", "Hormonal Changes", "Poor Hair Care Habits ", "Environmental Factors", "Smoking", "Weight Loss "]
    for col in columnas_yes_no:
        datos[col] = datos[col].map(map_yes_no)

    datos["Medical Conditions"] = datos["Medical Conditions"].map(map_medical_conditions)
    datos["Nutritional Deficiencies "] = datos["Nutritional Deficiencies "].map(map_nutritional_deficiencies)
    datos["Medications & Treatments"] = datos["Medications & Treatments"].map(map_medications)

    # Llenar NaN con "No Data" para evitar valores faltantes
    datos.fillna("No Data", inplace=True)

    # Agrupar edad
    bins = [0, 25, 35, 45, 60]
    labels = [0, 1, 2, 3]
    datos['Age_group'] = pd.cut(datos['Age'], bins=bins, labels=labels, right=False)
    datos.drop("Age", axis=1, inplace=True)

    # One-Hot Encoding asegurando las mismas categorías
    columnas_categoricas = ["Medical Conditions", "Nutritional Deficiencies ", "Medications & Treatments"]
    datos = pd.get_dummies(datos, columns=columnas_categoricas)

    # Asegurar columnas correctas y llenar valores faltantes con 0
    datos = datos.reindex(columns=columnas_modelo, fill_value=0)

    # **Convertir TODO a tipo float**
    datos = datos.astype(float)

    # **Aplicar transformación con el escalador**
    datos = sc.transform(datos)

    return datos

# Interfaz de usuario
st.title("Predicción de Pérdida de Cabello")
st.header("Ingresa los datos para predecir")

# Añadir imágenes en las esquinas
col1, col2 = st.columns(2)

with col1:
    st.image("media/calvo.jpg", width=300)  

with col2:
    st.image("media/pelo.jpg", width=300)

stress = st.selectbox("🧠 Nivel de estrés", ["Bajo", "Moderado", "Alto"])
age = st.number_input("🎂 Edad", min_value=0, max_value=100, value=30)

medical_conditions = st.selectbox(
    "⚕️ Condiciones médicas",
    list(map_medical_conditions.keys())
)

nutritional_deficiencies = st.selectbox(
    "🍎 Deficiencias nutricionales",
    list(map_nutritional_deficiencies.keys())
)

medications = st.selectbox(
    "💊 Medicamentos y tratamientos",
    list(map_medications.keys())
)

genetics = st.selectbox("🧬 Genética", ["Sí", "No"])
hormonal_changes = st.selectbox("🔄 Cambios hormonales", ["Sí", "No"])
poor_hair_care = st.selectbox("🛁 Malos hábitos de cuidado del cabello", ["Sí", "No"])
environmental_factors = st.selectbox("🌍 Factores ambientales", ["Sí", "No"])
smoking = st.selectbox("🚬 Fumar", ["Sí", "No"])
weight_loss = st.selectbox("⚖️ Pérdida de peso", ["Sí", "No"])

# Predecir
if st.button("Predecir"):
    datos_usuario = pd.DataFrame({
        "Stress": [stress], "Age": [age], 
        "Medical Conditions": [medical_conditions],
        "Nutritional Deficiencies ": [nutritional_deficiencies], 
        "Medications & Treatments": [medications],
        "Genetics": [genetics], "Hormonal Changes": [hormonal_changes], 
        "Poor Hair Care Habits ": [poor_hair_care],
        "Environmental Factors": [environmental_factors], 
        "Smoking": [smoking], "Weight Loss ": [weight_loss]
    })

    # Preprocesar datos
    try:
        datos_procesados = preprocesar_datos(datos_usuario)

        # Verificar si hay valores NaN antes de predecir
        if np.isnan(datos_procesados).any():
            st.error("Se encontraron valores faltantes en los datos procesados. Verifique las opciones seleccionadas.")
        else:
            prediccion = gb.predict(datos_procesados)
            st.success(f"La predicción es: {'Pérdida de cabello' if prediccion[0] == 1 else 'No pérdida de cabello'}")

            probabilidad = gb.predict_proba(datos_procesados)
            st.write(f"Probabilidad de pérdida de cabello: {probabilidad[0][1]:.2f}")
    
    except Exception as e:
        st.error(f"Error en el procesamiento: {e}")
