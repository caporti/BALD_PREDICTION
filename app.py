import streamlit as st
from PIL import Image
import os
# from pages import CV, eda, ML
from multiapp import MultiApp

# Configurar la página
st.set_page_config(page_title="Proyecto Calvos", page_icon="🧑‍🦲", layout="wide")

# Título de la app
st.title("🔍 Proyecto de Detección de Calvicie con IA")
st.write("Bienvenido a nuestra aplicación de detección de calvicie usando análisis de datos, visión por computadora y machine learning.")

# Cargar imágenes
media_folder = "media"
pelo_img = Image.open(os.path.join(media_folder, "pelo.jpg"))
calvo_img = Image.open(os.path.join(media_folder, "calvo.jpg"))

# Mostrar imágenes en columnas
col1, col2 = st.columns(2)
with col1:
    st.image(pelo_img, caption="Cabello Abundante", use_container_width=True)
with col2:
    st.image(calvo_img, caption="Calvicie", use_container_width=True)