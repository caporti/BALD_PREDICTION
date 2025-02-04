import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Cargar el modelo y el escalador (si los tienes guardados)
# gb = joblib.load('modelo_gb.pkl')
# sc = joblib.load('escalador.pkl')

# Función para preprocesar los datos de entrada
def preprocesar_datos(datos):
    # Mapear valores categóricos
    map_stres = {"Low": 0, "Moderate": 1, "High": 2}
    datos["Stress"] = datos["Stress"].map(map_stres)

    map_yes_no = {"Yes": 1, "No": 0}
    columnas_yes_no = ["Genetics", "Hormonal Changes", "Poor Hair Care Habits ", "Environmental Factors", "Smoking", "Weight Loss "]
    for col in columnas_yes_no:
        datos[col] = datos[col].map(map_yes_no)

    # Definir funciones de asignación
    def asignar_grupo(valor, diccionario):
        for grupo, valores in diccionario.items():
            if valor in valores:
                return grupo
        return "Otros"

    grupos_condiciones = {
        "Enfermedades inflamatorias de la piel": ["Eczema", "Psoriasis", "Dermatitis", "Seborrheic Dermatitis"],
        "Infecciones": ["Ringworm", "Scalp Infection"],
        "Trastornos del cabello": ["Alopecia Areata", "Androgenetic Alopecia"],
        "Problemas sistémicos": ["Thyroid Problems"],
        "Término genérico": ["Dermatosis"],
        "No Data": ["No Data"]
    }
    datos["Grupo_Condiciones"] = datos["Medical Conditions"].apply(lambda x: asignar_grupo(x, grupos_condiciones))
    datos.drop("Medical Conditions", axis=1, inplace=True)

    grupos_deficiencias = {
        "Vitaminas": ["Vitamin A Deficiency", "Vitamin D Deficiency", "Biotin Deficiency", "Vitamin E deficiency"],
        "Minerales": ["Magnesium deficiency", "Selenium deficiency", "Zinc Deficiency"],
        "Macronutrientes": ["Protein deficiency"],
        "Ácidos grasos": ["Omega-3 fatty acids"],
        "Sin deficiencia": ["No Data", "Iron deficiency"]
    }
    datos["Grupo Deficiencias"] = datos["Nutritional Deficiencies "].apply(lambda x: asignar_grupo(x, grupos_deficiencias))
    datos.drop("Nutritional Deficiencies ", axis=1, inplace=True)

    grupos_medicamentos = {
        "Antibióticos/Antifúngicos": ["Antibiotics", "Antifungal Cream"],
        "Enfermedades crónicas": ["Blood Pressure Medication", "Heart Medication"],
        "Inmunológicos": ["Immunomodulators", "Steroids"],
        "Salud mental": ["Antidepressants"],
        "Tratamientos para caída de cabello": ["Rogaine", "Accutane"],
        "Quimioterapia": ["Chemotherapy"],
        "No Data": ["No Data"]
    }
    datos["Grupo Medicamentos"] = datos["Medications & Treatments"].apply(lambda x: asignar_grupo(x, grupos_medicamentos))
    datos.drop("Medications & Treatments", axis=1, inplace=True)

    # Agrupar edad
    bins = [0, 25, 35, 45, 60]
    labels = [0, 1, 2, 3]
    datos['Age_group'] = pd.cut(datos['Age'], bins=bins, labels=labels, right=False)
    datos.drop("Age", axis=1, inplace=True)

    # Aplicar One-Hot Encoding
    columnas_categoricas = ["Grupo_Condiciones", "Grupo Deficiencias", "Grupo Medicamentos"]
    datos = pd.get_dummies(datos, columns=columnas_categoricas)

    # Asegurar que las columnas coincidan con el conjunto de entrenamiento
    columnas_modelo = [
        'Genetics', 'Hormonal Changes', 'Stress', 'Poor Hair Care Habits ', 
        'Environmental Factors', 'Smoking', 'Weight Loss ', 'Age_group',
        'Grupo_Condiciones_Enfermedades inflamatorias de la piel', 'Grupo_Condiciones_Infecciones',
        'Grupo_Condiciones_No Data', 'Grupo_Condiciones_Problemas sistémicos',
        'Grupo_Condiciones_Trastornos del cabello', 'Grupo_Condiciones_Término genérico',
        'Grupo Deficiencias_Macronutrientes', 'Grupo Deficiencias_Minerales',
        'Grupo Deficiencias_Sin deficiencia', 'Grupo Deficiencias_Vitaminas',
        'Grupo Deficiencias_Ácidos grasos', 'Grupo Medicamentos_Antibióticos/Antifúngicos',
        'Grupo Medicamentos_Enfermedades crónicas', 'Grupo Medicamentos_Inmunológicos',
        'Grupo Medicamentos_No Data', 'Grupo Medicamentos_Quimioterapia',
        'Grupo Medicamentos_Salud mental', 'Grupo Medicamentos_Tratamientos para caída de cabello'
    ]
    datos = datos.reindex(columns=columnas_modelo, fill_value=0)

    # Escalar los datos
    datos = sc.transform(datos)
    return datos

# Interfaz de Streamlit
st.title("Predicción de Pérdida de Cabello")

# Crear un formulario para que el usuario ingrese datos
st.header("Ingresa los datos para predecir")

# Campos de entrada
stress = st.selectbox("Nivel de estrés", ["Low", "Moderate", "High"])
age = st.number_input("Edad", min_value=0, max_value=100, value=30)
medical_conditions = st.selectbox("Condiciones médicas", ["Eczema", "Psoriasis", "Thyroid Problems", "No Data"])
nutritional_deficiencies = st.selectbox("Deficiencias nutricionales", ["Vitamin D Deficiency", "Iron deficiency", "No Data"])
medications = st.selectbox("Medicamentos y tratamientos", ["Rogaine", "Antibiotics", "No Data"])
genetics = st.selectbox("Genética", ["Yes", "No"])
hormonal_changes = st.selectbox("Cambios hormonales", ["Yes", "No"])
poor_hair_care = st.selectbox("Malos hábitos de cuidado del cabello", ["Yes", "No"])
environmental_factors = st.selectbox("Factores ambientales", ["Yes", "No"])
smoking = st.selectbox("Fumar", ["Yes", "No"])
weight_loss = st.selectbox("Pérdida de peso", ["Yes", "No"])

# Botón para realizar la predicción
if st.button("Predecir"):
    # Crear un DataFrame con los datos ingresados
    datos_usuario = pd.DataFrame({
        "Stress": [stress],
        "Age": [age],
        "Medical Conditions": [medical_conditions],
        "Nutritional Deficiencies ": [nutritional_deficiencies],
        "Medications & Treatments": [medications],
        "Genetics": [genetics],
        "Hormonal Changes": [hormonal_changes],
        "Poor Hair Care Habits ": [poor_hair_care],
        "Environmental Factors": [environmental_factors],
        "Smoking": [smoking],
        "Weight Loss ": [weight_loss]
    })

    # Preprocesar los datos
    datos_procesados = preprocesar_datos(datos_usuario)

    # Realizar la predicción
    prediccion = gb.predict(datos_procesados)

    # Mostrar el resultado
    st.success(f"La predicción es: {'Pérdida de cabello' if prediccion[0] == 1 else 'No pérdida de cabello'}")

