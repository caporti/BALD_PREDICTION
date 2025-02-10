import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Cargar el modelo y el escalador
with open("modelo_calvicie.pkl", "rb") as f:
    data = pickle.load(f)

gb = data["modelo"]
sc = data["escalador"]

# Nombres originales de las columnas en el entrenamiento
columnas_modelo = [
    'Genetics', 'Hormonal Changes', 'Stress',
    'Poor Hair Care Habits ', 'Environmental Factors', 'Smoking',
    'Weight Loss ', 'Age_group',
    'Enfermedades inflamatorias de la piel', 'Infecciones', 'No Data',
    'Problemas sistémicos', 'Trastornos del cabello', 'Término genérico',
    'Macronutrientes', 'Minerales', 'Sin deficiencia', 'Vitaminas',
    'Ácidos grasos', 'Antibióticos/Antifúngicos', 'Enfermedades crónicas',
    'Inmunológicos', 'No Data', 'Quimioterapia', 'Salud mental',
    'Tratamientos para caída de cabello'
]

# Función para preprocesar los datos
def preprocesar_datos(datos):
    map_stres = {"Low": 0, "Moderate": 1, "High": 2}
    datos["Stress"] = datos["Stress"].map(map_stres)
    
    map_yes_no = {"Yes": 1, "No": 0}
    columnas_yes_no = ["Genetics", "Hormonal Changes", "Poor Hair Care Habits ", "Environmental Factors", "Smoking", "Weight Loss "]
    for col in columnas_yes_no:
        datos[col] = datos[col].map(map_yes_no)
    
    def asignar_grupo(valor, diccionario):
        for grupo, valores in diccionario.items():
            if valor in valores:
                return grupo
        return "Otros"

    grupos = {
        "Medical Conditions": {
            "Enfermedades inflamatorias de la piel": ["Eczema", "Psoriasis", "Dermatitis", "Seborrheic Dermatitis"],
            "Infecciones": ["Ringworm", "Scalp Infection"],
            "Trastornos del cabello": ["Alopecia Areata", "Androgenetic Alopecia"],
            "Problemas sistémicos": ["Thyroid Problems"],
            "Término genérico": ["Dermatosis"],
            "No Data": ["No Data"]
        },
        "Nutritional Deficiencies ": {
            "Vitaminas": ["Vitamin A Deficiency", "Vitamin D Deficiency", "Biotin Deficiency", "Vitamin E deficiency"],
            "Minerales": ["Magnesium deficiency", "Selenium deficiency", "Zinc Deficiency"],
            "Macronutrientes": ["Protein deficiency"],
            "Ácidos grasos": ["Omega-3 fatty acids"],
            "Sin deficiencia": ["No Data", "Iron deficiency"]
        },
        "Medications & Treatments": {
            "Antibióticos/Antifúngicos": ["Antibiotics", "Antifungal Cream"],
            "Enfermedades crónicas": ["Blood Pressure Medication", "Heart Medication"],
            "Inmunológicos": ["Immunomodulators", "Steroids"],
            "Salud mental": ["Antidepressants"],
            "Tratamientos para caída de cabello": ["Rogaine", "Accutane"],
            "Quimioterapia": ["Chemotherapy"],
            "No Data": ["No Data"]
        }
    }
    
    for col, dic in grupos.items():
        datos[col] = datos[col].apply(lambda x: asignar_grupo(x, dic))
    
    # Agrupar edad
    bins = [0, 25, 35, 45, 60]
    labels = [0, 1, 2, 3]
    datos['Age_group'] = pd.cut(datos['Age'], bins=bins, labels=labels, right=False)
    datos.drop("Age", axis=1, inplace=True)
    
    # Ya ha parado de agrupar
    
    # One-Hot Encoding asegurando las mismas categorías
    columnas_categoricas = list(grupos.keys())
    datos = pd.get_dummies(datos, columns=columnas_categoricas)
    
    # Asegurar columnas correctas
    datos = datos.reindex(columns=columnas_modelo, fill_value=0)
    
    datos = sc.transform(datos)
    return datos

st.title("Predicción de Pérdida de Cabello")
st.header("Ingresa los datos para predecir")

# Metemos los datos en un DataFrame

stress = st.selectbox("Nivel de estrés", ["Low", "Moderate", "High"])
age = st.number_input("Edad", min_value=0, max_value=100, value=30)
medical_conditions = st.selectbox("Condiciones médicas", ["Eczema", "Psoriasis", "Thyroid Problems", "No Data"])
nutritional_deficiencies = st.selectbox("Deficiencias nutricionales", ["Vitamin D Deficiency", "Biotin Deficiency", "Iron deficiency", "No Data"]) #Hay que meter todos los nombres que faltan aquí
medications = st.selectbox("Medicamentos y tratamientos", ["Rogaine", "Antibiotics", "No Data"])
genetics = st.selectbox("Genética", ["Yes", "No"])
hormonal_changes = st.selectbox("Cambios hormonales", ["Yes", "No"])
poor_hair_care = st.selectbox("Malos hábitos de cuidado del cabello", ["Yes", "No"])
environmental_factors = st.selectbox("Factores ambientales", ["Yes", "No"])
smoking = st.selectbox("Fumar", ["Yes", "No"])
weight_loss = st.selectbox("Pérdida de peso", ["Yes", "No"])

# Predecir

if st.button("Predecir"):
    datos_usuario = pd.DataFrame({
        "Stress": [stress], "Age": [age], "Medical Conditions": [medical_conditions],
        "Nutritional Deficiencies ": [nutritional_deficiencies], "Medications & Treatments": [medications],
        "Genetics": [genetics], "Hormonal Changes": [hormonal_changes], "Poor Hair Care Habits ": [poor_hair_care],
        "Environmental Factors": [environmental_factors], "Smoking": [smoking], "Weight Loss ": [weight_loss]
    })
    datos_procesados = preprocesar_datos(datos_usuario)
    prediccion = gb.predict(datos_procesados)
    st.success(f"La predicción es: {'Pérdida de cabello' if prediccion[0] == 1 else 'No pérdida de cabello'}")
    
    # Mostrar probabilidad, podemos quitarlo perfectamente pero me parece interesante porque tenemos 50-50
    
    probabilidad = gb.predict_proba(datos_procesados)
    st.write(f"Probabilidad de pérdida de cabello: {probabilidad[0][1]:.2f}")
    