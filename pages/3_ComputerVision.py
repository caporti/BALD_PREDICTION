import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import streamlit as st
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO

# Configuración de la página
st.set_page_config(
    page_title="BaldAI - Visión por Computador",
    page_icon="📷",
    layout="wide"
)

# CSS personalizado (mantener el mismo estilo)
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
    
    .prediction-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 2rem 0;
    }
    
    .progress-bar {
        height: 20px;
        border-radius: 10px;
        background: #f0f0f0;
        margin: 1rem 0;
    }
    
    .progress-bar-fill {
        height: 100%;
        border-radius: 10px;
        background: var(--primary);
    }
</style>
""", unsafe_allow_html=True)

# Configuración del dispositivo
device = torch.device("cpu")

# Cargar el modelo
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Bald vs Not Bald
    model.load_state_dict(torch.load("modelo_calvicie2.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# Transformaciones de la imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Función para obtener datos de LinkedIn
def fetch_profile_data(url):
    api_endpoint = 'https://nubela.co/proxycurl/api/v2/linkedin'
    api_key = '1bnj5bMMVRtmJXAJTxYaoQ'
    headers = {'Authorization': 'Bearer ' + api_key}
    response = requests.get(api_endpoint, params={'url': url, 'skills': 'include'}, headers=headers)
    return response.json() if response.status_code == 200 else None

# Interfaz de usuario
st.markdown('<div class="header">', unsafe_allow_html=True)
st.title("📷 BaldAI - Detección de Calvicie por Imagen")
st.markdown("Sistema de clasificación capilar mediante visión por computador")
st.markdown('</div>', unsafe_allow_html=True)

# Sección de carga de imagen
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("### 📸 Opciones de Entrada")
    input_method = st.radio("Selecciona el método de entrada:", 
                          ["Subir imagen", "Tomar foto", "URL de LinkedIn"])

    uploaded_file = None
    linkedin_url = None

    if input_method == "Subir imagen":
        uploaded_file = st.file_uploader("Sube una imagen facial", 
                                       type=["jpg", "jpeg", "png"],
                                       help="La imagen debe mostrar claramente el cuero cabelludo")
    elif input_method == "Tomar foto":
        uploaded_file = st.camera_input("Toma una foto de tu cuero cabelludo",
                                       help="Asegúrate de tener buena iluminación y enfoque")
    else:
        linkedin_url = st.text_input("Pega una URL de perfil de LinkedIn")

# Sección de visualización de ejemplo
with col2:
    st.markdown("### Ejemplo de Imágenes")
    example_img = Image.open("media/calvo.jpg")
    st.image(example_img, use_container_width=True)

# Procesamiento de la imagen
def process_and_predict(image):
    try:
        # Validación de tamaño de imagen
        if image.size[0] < 224 or image.size[1] < 224:
            st.warning("⚠️ La imagen es demasiado pequeña. Por favor, sube una imagen de mayor resolución.")
            return

        st.markdown("---")
        st.markdown("### 🖼️ Imagen Analizada")
        st.image(image, caption="Imagen cargada", use_container_width=True)
        
        # Preprocesar la imagen
        with st.spinner("Analizando características capilares..."):
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Hacer la predicción
            with torch.no_grad():
                outputs = model(image_tensor)
                _, predicted = torch.max(outputs, 1)
                probabilities = F.softmax(outputs, dim=1)
                
                # Mapear el índice a la clase
                class_names = ["Calvo", "No Calvo"]
                result = class_names[predicted.item()]
                prob_bald = probabilities[0][0].item()
                prob_not_bald = probabilities[0][1].item()
        
        # Mostrar resultados
        st.markdown("---")
        st.markdown("### 📊 Resultados del Análisis")

        st.markdown(f"""
        <div class="prediction-card" style="background: none; box-shadow: none;">
            <h2>Predicción: {result}</h2>
            <div style="background: #f0f0f0; height: 20px; border-radius: 10px; margin: 1rem 0;">
                <div style="width: {prob_bald*100:.1f}%; 
                        height: 100%; 
                        border-radius: 10px; 
                        background: {'#e74c3c' if predicted.item() == 0 else '#2ecc71'};"></div>
            </div>
            <p>Probabilidad de calvicie: <strong>{prob_bald*100:.1f}%</strong></p>
            <p>Probabilidad de no calvicie: <strong>{prob_not_bald*100:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

        # Recomendaciones
        st.markdown("---")
        st.markdown("### 💡 Recomendaciones")
        if predicted.item() == 0:
            st.markdown("""
            - Consulta con un especialista en salud capilar
            - Considera tratamientos preventivos
            - Monitorea cambios en tu cuero cabelludo
            """)
        else:
            st.markdown("""
            - Mantén hábitos saludables de cuidado capilar
            - Realiza chequeos periódicos
            - Usa productos adecuados para tu tipo de cabello
            """)
            
    except Exception as e:
        st.error(f"🚨 Error en el procesamiento: {str(e)}")
        st.info("ℹ️ Asegúrate de subir una imagen válida y nítida del cuero cabelludo")

# Lógica principal de procesamiento
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    process_and_predict(image)
elif linkedin_url:
    try:
        with st.spinner("Obteniendo datos de LinkedIn..."):
            profile_data = fetch_profile_data(linkedin_url)
            if profile_data and "profile_pic_url" in profile_data:
                response = requests.get(profile_data["profile_pic_url"])
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content))
                    process_and_predict(image)
                else:
                    st.error("Error al descargar la imagen del perfil")
            else:
                st.error("Perfil de LinkedIn no encontrado o sin imagen")
    except Exception as e:
        st.error(f"Error al procesar perfil de LinkedIn: {str(e)}")