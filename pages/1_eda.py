import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Configuración de página
st.set_page_config(
    page_title="BaldAI - EDA Capilar",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado
st.markdown("""
<style>
    /* Añade estos estilos */
    .stDataFrame {
        background: white !important;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .dataframe td, .dataframe th {
        color: #2c3e50 !important;
        border-color: #f1f1f1 !important;
    }
    
    .metric-value {
        font-size: 1.2rem !important;
        color: var(--primary) !important;
    }
    
    /* Mejora el contraste general */
    body {
        color: #2c3e50;
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Cargar datos directamente
@st.cache_data
def load_data():
    return pd.read_csv('data/Predict Hair Fall.csv')

df = load_data()
df.drop('Id', axis=1, inplace=True)

# ---------------------- Header ----------------------
st.markdown('<div class="header">', unsafe_allow_html=True)
st.title("🧑🦲 Análisis Exploratorio de Datos Capilares")
st.markdown("**Dataset:** `Predict Hair Fall.csv` | **Registros:** `{}` | **Variables:** `{}`".format(df.shape[0], df.shape[1]))
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- Métricas rápidas ----------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-box">'
                '<h3>👥 Población</h3>'
                '<p style="font-size: 2rem; color: var(--primary); margin: 0;">{}</p>'
                '</div>'.format(df.shape[0]), unsafe_allow_html=True)

with col2:
    balance = df['Hair Loss'].value_counts(normalize=True)[1]
    st.markdown('<div class="metric-box">'
                '<h3>⚖️ Balance</h3>'
                '<p style="font-size: 2rem; color: var(--secondary); margin: 0;">{:.1%}</p>'
                '</div>'.format(balance), unsafe_allow_html=True)

with col3:
    avg_age = df['Age'].mean()
    st.markdown('<div class="metric-box">'
                '<h3>📅 Edad Media</h3>'
                '<p style="font-size: 2rem; color: var(--accent); margin: 0;">{:.1f}</p>'
                '</div>'.format(avg_age), unsafe_allow_html=True)

with col4:
    missing = df.isna().sum().sum()
    st.markdown('<div class="metric-box">'
                '<h3>⚠️ Faltantes</h3>'
                '<p style="font-size: 2rem; color: #e67e22; margin: 0;">{}</p>'
                '</div>'.format(missing), unsafe_allow_html=True)

# ---------------------- Análisis Univariado ----------------------
with st.container():
    st.markdown("## 🔍 Distribución de Variables")
    
    tab1, tab2 = st.tabs(["Categóricas", "Numéricas"])
    
    with tab1:
        col1, col2 = st.columns([3, 2])
        with col1:
            selected_cat = st.selectbox("Selecciona variable categórica:", 
                                      df.select_dtypes(include='object').columns,
                                      key='cat_select')
            
            fig = px.pie(df, names=selected_cat, hole=0.3,
                        color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(title=f"Distribución de {selected_cat}", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Frecuencias")
            counts = df[selected_cat].value_counts()
            for value, count in counts.items():
                st.markdown(f"- **{value}**: `{count}` registros ({count/len(df):.1%})")
    
    with tab2:
        col1, col2 = st.columns([3, 2])
        with col1:
            selected_num = st.selectbox("Selecciona variable numérica:", 
                                      df.select_dtypes(include=np.number).columns,
                                      key='num_select')
            
            fig = px.histogram(df, x=selected_num, marginal="box", 
                             nbins=30, color_discrete_sequence=["#2ecc71"])
            fig.update_layout(title=f"Distribución de {selected_num}", 
                            template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Estadísticas descriptivas
            st.markdown("### 📊 Estadísticas Descriptivas")
            stats = df[selected_num].describe().to_frame().T  # Convertir a DataFrame
            st.dataframe(stats.style.format("{:.2f}"), use_container_width=True)

# ---------------------- Análisis Multivariado ----------------------
with st.container():
    st.markdown("## 🔗 Relaciones entre Variables")
    
    col1, col2 = st.columns([2, 3])
    with col1:
        x_var = st.selectbox("Variable X:", df.select_dtypes(include=np.number).columns)
        y_var = st.selectbox("Variable Y:", df.select_dtypes(include=np.number).columns)
        hue_var = st.selectbox("Variable de color:", df.select_dtypes(include='object').columns)
    
    with col2:
        fig = px.scatter(df, x=x_var, y=y_var, color=hue_var,
                       trendline="lowess", 
                       color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(title=f"{x_var} vs {y_var} | Color: {hue_var}",
                        height=500)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------- Feature Importance ----------------------
with st.expander("📈 Importancia de Variables (Machine Learning)"):
    st.markdown("### 🎯 Importancia de Características")
    
    # Preprocesamiento
    train_df = df.iloc[:, :12].copy()

    # Codificar TODAS las columnas categóricas excepto la target
    categoricals = [col for col in df.columns if df[col].dtype == 'object' and col != 'Hair Loss']

    le = LabelEncoder()
    for col in categoricals:  # Eliminamos [:-1]
        train_df[col] = le.fit_transform(train_df[col])

    # Asegurar que la target está en la última posición
    train_df = pd.concat([train_df.drop('Hair Loss', axis=1), train_df['Hair Loss']], axis=1)

    X = train_df.iloc[:, :-1].values
    y = train_df.iloc[:, -1].values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Modelo
    forest = RandomForestClassifier(random_state=42)
    forest.fit(X_train, y_train)
    
    # Importancias
    importances = forest.feature_importances_
    features = train_df.columns[:-1]
    
    fig = px.bar(x=features, y=importances, 
               color=importances, color_continuous_scale='Bluered')
    fig.update_layout(title="Importancia de Variables (Random Forest)",
                   xaxis_title="Variables",
                   yaxis_title="Importancia",
                   coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------- PCA ----------------------
with st.expander("🔬 Análisis de Componentes Principales"):
    st.markdown("### 📉 Análisis PCA")
    
    scaler_pca = StandardScaler()
    scaled_df = scaler_pca.fit_transform(X_train)
    
    pca = PCA()
    pca.fit(scaled_df)
    
    variance_ratio = pca.explained_variance_ratio_
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(x=range(1, len(variance_ratio)+1), 
                   y=variance_ratio,
                   labels={'x': 'Componente', 'y': 'Varianza Explicada'},
                   color_discrete_sequence=["#3498db"])
        fig.update_layout(title="Varianza Explicada por Componente")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cumulative = np.cumsum(variance_ratio)
        fig = px.line(x=range(1, len(cumulative)+1), y=cumulative,
                    markers=True, color_discrete_sequence=["#e74c3c"])
        fig.update_layout(title="Varianza Acumulada",
                        yaxis_title="Varianza Acumulada")
        fig.add_hline(y=0.95, line_dash="dot", line_color="grey")
        st.plotly_chart(fig, use_container_width=True)