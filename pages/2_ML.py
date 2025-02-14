import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Configuración de página
st.set_page_config(
    page_title="BaldAI - Predictor Capilar",
    page_icon="🧑🦲",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    :root {
        --primary: #2ecc71;
        --secondary: #3498db;
        --accent: #e74c3c;
    }
    
    .header {
        padding: 2rem 0;
        border-bottom: 3px solid var(--primary);
        margin-bottom: 2rem;
    }
    
    .input-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .risk-high {
        background: #f8d7da;
        border: 2px solid #721c24;
    }
    
    .risk-low {
        background: #d4edda;
        border: 2px solid #155724;
    }
</style>
""", unsafe_allow_html=True)

# Cargar modelo y escalador
@st.cache_data
def load_model():
    with open("modelo_calvicie.pkl", "rb") as f:
        data = pickle.load(f)
    return data["modelo"], data["escalador"]

gb, sc = load_model()

# Mapeos de conversión
map_stress = {"Bajo": 0, "Moderado": 1, "Alto": 2}
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
# Función de preprocesamiento
def preprocesar_datos(datos):
    # Convertir valores categóricos
    datos["Stress"] = datos["Stress"].map(map_stress)
    
    columnas_yes_no = ["Genetics", "Hormonal Changes", "Poor Hair Care Habits ", 
                      "Environmental Factors", "Smoking", "Weight Loss "]
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

    # One-Hot Encoding
    columnas_categoricas = ["Medical Conditions", "Nutritional Deficiencies ", "Medications & Treatments"]
    datos = pd.get_dummies(datos, columns=columnas_categoricas)

    # Asegurar columnas correctas
    datos = datos.reindex(columns=sc.feature_names_in_, fill_value=0)
    datos = datos.astype(float)

    # Escalar datos
    return sc.transform(datos)

# Interfaz de usuario
st.markdown('<div class="header">', unsafe_allow_html=True)
st.title("🧑🦲 BaldAI - Predictor de Salud Capilar")
st.markdown("**Sistema de predicción de pérdida capilar mediante Machine Learning**")
st.markdown('</div>', unsafe_allow_html=True)

# Sección de entrada de datos
with st.container():
    with st.form("prediccion_form"):
        st.markdown("### 📋 Datos del Paciente")
        
        # Organizar inputs en columnas
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("🎂 Edad", 18, 59, 30)
            stress = st.select_slider("🧠 Nivel de estrés", options=["Bajo", "Moderado", "Alto"])
            
        with c2:
            genetics = st.radio("🧬 Historial genético", ["Sí", "No"], horizontal=True)
            smoking = st.toggle("🚬 Fumador actual")
        
        st.markdown("---")
        st.markdown("### 🏥 Historial Clínico")
        
        # Tres columnas para sección médica
        m1, m2, m3 = st.columns(3)
        with m1:
            medical_conditions = st.selectbox(
                "⚕️ Condiciones médicas",
                list(map_medical_conditions.keys()),
                index=len(map_medical_conditions)-1
            )
            
        with m2:
            nutritional_deficiencies = st.selectbox(
                "🍎 Deficiencias nutricionales",
                list(map_nutritional_deficiencies.keys()),
                index=len(map_nutritional_deficiencies)-1
            )
            
        with m3:
            medications = st.selectbox(
                "💊 Tratamientos médicos",
                list(map_medications.keys()),
                index=len(map_medications)-1
            )
        
        # Botón de envío
        st.markdown("---")
        submitted = st.form_submit_button("🔮 Obtener Predicción", use_container_width=True)

# Procesamiento y resultados
if submitted:
    try:
        # Crear DataFrame con los datos
        datos_usuario = pd.DataFrame({
            "Stress": [stress], "Age": [age], 
            "Medical Conditions": [medical_conditions],
            "Nutritional Deficiencies ": [nutritional_deficiencies], 
            "Medications & Treatments": [medications],
            "Genetics": [genetics], "Hormonal Changes": ["No"], 
            "Poor Hair Care Habits ": ["No"],
            "Environmental Factors": ["No"], 
            "Smoking": ["Sí" if smoking else "No"], "Weight Loss ": ["No"]
        })

        # Preprocesar y predecir
        datos_procesados = preprocesar_datos(datos_usuario)
        prediccion = gb.predict(datos_procesados)[0]
        probabilidad = gb.predict_proba(datos_procesados)[0][1]
        
        # Mostrar resultados
        risk_class = "risk-high" if prediccion == 1 else "risk-low"
        emoji = "⚠️" if prediccion == 1 else "✅"
        
        st.markdown(f"""
        <div class="prediction-card {risk_class}">
            <h2>{emoji} Resultado de la Predicción</h2>
            <p style="font-size: 1.5rem; margin: 1rem 0;">
                Probabilidad de pérdida capilar: 
                <strong>{probabilidad*100:.1f}%</strong>
            </p>
            <div style="background: {'#f5b7b1' if prediccion == 1 else '#abebc6'}; 
                     height: 20px; border-radius: 10px; margin: 1rem 0;">
                <div style="width: {probabilidad*100}%; 
                          background: {'#e74c3c' if prediccion == 1 else '#2ecc71'}; 
                          height: 100%; border-radius: 10px;"></div>
            </div>
            <h3>🔍 Factores Clave:</h3>
            <ul>
                <li>Edad: {age} años</li>
                <li>Historial genético: {genetics}</li>
                <li>Nivel de estrés: {stress}</li>
                <li>Condición médica principal: {medical_conditions}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"🚨 Error en el procesamiento: {str(e)}")
        st.info("ℹ️ Verifique que todos los campos estén correctamente completados")