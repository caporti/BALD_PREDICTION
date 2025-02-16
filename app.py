import streamlit as st
from PIL import Image
import os

# Configurar la página
st.set_page_config(
    page_title="BaldAI - Predicción de Calvicie",
    page_icon="🧑🦲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado
st.markdown("""
<style>
    .header {padding: 0 0 3rem 0;}
    .team {color: #2ecc71; font-weight: 700;}
    .section {padding: 2rem 1rem; border-radius: 15px; background: #f8f9fa; margin: 1rem 0;}
    .card {transition: transform .2s; border-radius: 15px; padding: 1.5rem; margin: 1rem 0;}
    .card:hover {transform: scale(1.02); cursor: pointer;}
    img {border-radius: 15px;}
</style>
""", unsafe_allow_html=True)

# Título y descripción
st.markdown('<div class="header">', unsafe_allow_html=True)
st.title("🧑🦲 BaldAI - Sistema de Detección de Calvicie")
st.markdown("""
**¡Bienvenido a nuestra solución integral para la detección temprana de alopecia!**  
*Desarrollado por Nacho Martínez, Jorge Moltó y Carlos Portilla para el Máster en Inteligencia Artificial*
""")
st.markdown('</div>', unsafe_allow_html=True)

# Nueva sección de contexto sobre alopecia en España
st.markdown("""
<div>
    <h3>🔍 ¿Por qué es crucial abordar la alopecia en España?</h3>
    <p>La alopecia es un problema creciente en España, con un impacto significativo en la calidad de vida de quienes la padecen. Aquí algunos datos clave:</p>
    <ul>
        <li>🇪🇸 <strong>España lidera los rankings mundiales de calvicie</strong>, con un 44.5% de hombres afectados por alopecia androgénica.</li>
        <li>👨‍🦲 El 30% de los varones de 30 años ya presenta pérdida capilar significativa.</li>
        <li>🧠 La OMS reconoce su impacto en la salud mental: <strong>38% mayor riesgo de depresión</strong> en pacientes con alopecia.</li>
        <li>💡 El estrés pandémico aumentó un 40% los casos de efluvio telógeno (caída temporal del cabello).</li>
    </ul>
    <p>Nuestra solución combina <strong>IA avanzada</strong> y <strong>dermatología digital</strong> para:</p>
    <ol>
        <li>🔬 Detectar patrones tempranos mediante análisis exploratorio de datos (EDA).</li>
        <li>🤖 Predecir riesgos con modelos de Machine Learning.</li>
        <li>📸 Diagnosticar mediante Redes Neuronales Convolucionales (CNN).</li>
    </ol>
    <p>Únete a la revolución capilar.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)


# Tarjetas de navegación
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Análisis Exploratorio"):
        st.session_state["page"] = "pages/1_eda.py"
        st.switch_page("pages/1_eda.py")
    st.image(Image.open("media/datospelo.jpg"), use_container_width=True)

with col2:
    if st.button("🤖 Predicción ML"):
        st.session_state["page"] = "pages/2_ML.py"
        st.switch_page("pages/2_ML.py")
    st.image(Image.open("media/prediccion.jpg"), use_container_width=True)

with col3:
    if st.button("📸 Diagnóstico por Imagen"):
        st.session_state["page"] = "pages/3_ComputerVision.py"
        st.switch_page("pages/3_ComputerVision.py")
    st.image(Image.open("media/calvoespejo.jpg"), use_container_width=True)

# Sección del equipo
st.markdown("""
<div class="section">
    <h3>👥 Nuestro Equipo</h3>
    <div class="team">Carlos Portilla · Nacho Martínez· Jorge Moltó</div>
    <p>Estudiantes del Máster en Inteligencia Artificial especializados en salud capilar</p>
    <img src="https://images.unsplash.com/photo-1552581234-26160f608093" style="width:100%; height:300px; object-fit:cover;">
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#666;">
    <p>© 2024 BaldAI · Todos los derechos reservados</p>
</div>
""", unsafe_allow_html=True)