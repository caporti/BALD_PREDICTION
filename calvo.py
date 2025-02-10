import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import streamlit as st
import torch.nn as nn
import torch.nn.functional as F

# Configuración del dispositivo
device = torch.device("cpu")  # Usar CPU explícitamente

# Cargar el modelo entrenado
model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # Bald vs Not Bald
model.load_state_dict(torch.load("modelo_calvicie2.pth", map_location=device))
model = model.to(device)
model.eval()

# Transformaciones de la imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Interfaz de usuario con Streamlit
st.title("Detección de Calvicie con IA")
st.write("Sube una imagen para analizar si la persona es calva o no.")

uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)
    
    # Preprocesar la imagen
    image = transform(image).unsqueeze(0).to(device)
    
    # Hacer la predicción
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        
        # Mapear el índice a la clase
        class_names = ["Bald", "Not Bald"]
        result = class_names[predicted.item()]
        probabilities = F.softmax(outputs, dim=1)
        
        # Mostrar el resultado
        st.write(f"### Predicción: {result}")
        st.write(f"Probabilidades: {probabilities}")