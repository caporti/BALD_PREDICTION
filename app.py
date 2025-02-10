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
	

# Display
st.write("Hello World EDEM students from Conchita")
st.write("Welcome to our streamlit sessions")
st.write(123455154125151513)
st.write('Inglés o español?')
st.markdown("### eueuee")
st.balloons()

# Cargar datos
# df = pd.read_csv('./Predict Hair Fall.csv') # Si queremos que solo se haga con ese csv
# st.dataframe(df.head(6))

uploaded_file = st.file_uploader("Carga un archivo CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Archivo cargado correctamente.")
    st.dataframe(df.head(5))



# ---------------------- Gráfico de columna categórica ----------------------
# Visualización de la distribución de una columna categórica con colores y etiquetas
# Visualización de la distribución de una columna categórica con gráficos
st.title("Visualización de Distribución de una Columna")

# Seleccionar columna categórica
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
selected_column_1 = st.selectbox("Selecciona una columna categórica:", categorical_columns, key="categorical_select_1")

# Graficar la distribución de la columna seleccionada como un gráfico de barras
if selected_column_1:
    # Establecer el estilo para el gráfico de barras
    sns.set_style("darkgrid")
    
    # Crear gráfico de barras
    fig, ax = plt.subplots(figsize=(8, 5))
    ax = sns.countplot(x=df[selected_column_1], ax=ax, palette='Set2')  # Usar una paleta de colores, por ejemplo 'Set2'
    
    # Añadir etiquetas a las barras
    for bars in ax.containers:
        ax.bar_label(bars, fontsize=9, fontweight='bold', color='black')

    # Rotar etiquetas si es necesario
    plt.xticks(rotation=45)
    
    # Mostrar el gráfico de barras
    st.pyplot(fig)

    # Graficar la distribución de la columna seleccionada como un gráfico de pastel
    label_counts = df[selected_column_1].value_counts()

    fig, ax = plt.subplots()
    ax.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, 
           colors=sns.color_palette("Set2", len(label_counts)))
    ax.set_title(f'Distribución de la columna: {selected_column_1}')

    # Mostrar el gráfico de pastel
    st.pyplot(fig)

    # Mostrar los conteos de cada etiqueta
    st.write(f"Conteos de la columna {selected_column_1}:")
    st.write(label_counts)



# ---------------------- KDE + Histograma con Seaborn ----------------------
st.title("Ver distribución de edades")

numerical_columns = df.select_dtypes(include=['int', 'float']).columns.tolist()
selected_numerical = st.selectbox("Selecciona una columna numérica:", numerical_columns, 
                                  index=numerical_columns.index('Age') if 'Age' in numerical_columns else 0, 
                                  key="numerical_select")

if st.button("Mostrar distribución"):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    sns.kdeplot(df, x=selected_numerical, fill=True, color='green', ax=ax)
    sns.histplot(df, x=selected_numerical, stat='density', fill=False, color='green', ax=ax)
    plt.title(f"{selected_numerical}", color='black')
    st.pyplot(fig)

# ---------------------- Histograma interactivo con Plotly ----------------------
st.title("Ver distribución de edades (Interactivo)")

selected_numerical_plotly = st.selectbox("Selecciona una columna numérica:", numerical_columns, 
                                         index=numerical_columns.index('Age') if 'Age' in numerical_columns else 0, 
                                         key="numerical_select_plotly")

if st.button("Mostrar distribución interactiva"):
    fig = px.histogram(df, x=selected_numerical_plotly, marginal="rug", nbins=30, opacity=0.7, color_discrete_sequence=['green'])
    
    fig.update_layout(
        title=f"{selected_numerical_plotly}",
        xaxis_title=selected_numerical_plotly,
        yaxis_title="Density"
    )

    st.plotly_chart(fig)

# ---------------------- Gráfico de importancias ----------------------
st.title("Importancias de características")

# ---------------------- Preprocesamiento de datos ----------------------

# Seleccionar las primeras 12 columnas y hacer una copia para entrenamiento
train_df = df.iloc[:, :12].copy()

# Definir las columnas categóricas excluyendo 'Age' y 'Hair Loss'
categoricals = df.columns[(df.columns != "Age") & (df.columns != "Hair Loss")]

# Codificar las columnas categóricas con LabelEncoder
le = LabelEncoder()
for i in categoricals[:-1]:  # Excluir la última columna categórica
    train_df[i] = le.fit_transform(train_df[i])

# Separar las características (X) y la etiqueta (y)
X = train_df.iloc[:, :-1].values
y = train_df.iloc[:, -1].values

# Escalar las características numéricas
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# ---------------------- Modelo de RandomForest para importancias ----------------------

# Crear y entrenar el modelo RandomForestClassifier
forest = RandomForestClassifier(random_state=42)
forest.fit(X_train, y_train)

# Obtener las importancias de las características
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

# Crear un DataFrame con las importancias y su desviación estándar
forest_importances = pd.Series(importances, index=train_df.columns[:-1])

# ---------------------- Visualización de Importancias ----------------------

# Graficar las importancias
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax, color='green')
ax.set_title("Feature Importances using MDI")
ax.set_ylabel("Mean Decrease in Impurity")
fig.tight_layout()

# Mostrar el gráfico en Streamlit
st.pyplot(fig)

# ----------------------- PCA --------------------------------------------

# Establecer el estándar de escalado para PCA
scaler_pca = StandardScaler()
scaled_df = scaler_pca.fit_transform(X_train)

# Aplicar PCA
pca = PCA()
pca.fit(scaled_df)

# Explicar la varianza para cada componente
variance_ratio = pca.explained_variance_ratio_
st.write("Explained variance ratio:", variance_ratio) # Varianza explicada

# Graficar la varianza explicada por cada componente
plt.figure(figsize=(10, 6))
plt.bar(range(len(variance_ratio)), variance_ratio)
plt.xticks(range(len(variance_ratio)))
plt.ylabel('Variance')
plt.xlabel('PCA Feature')
plt.title('Variance Explained by Each PCA Component')
st.pyplot(plt)

# Cálculo de la varianza acumulada para decidir el número de componentes
cumulative_variance_ratio = np.cumsum(variance_ratio)
st.write("Cumulative explained variance:", cumulative_variance_ratio) # Varianza acumulada

# Graficar la varianza acumulada explicada
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--')  # Línea umbral para 95%
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
st.pyplot(plt)

# Determinar el número de componentes para explicar el 95% de la varianza
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
st.write(f'Número de componentes que explican al menos el 95% de la varianza: {n_components_95}')

# Aplicar PCA con el número de componentes seleccionado
pca = PCA(n_components=n_components_95)
X_train_pca = pca.fit_transform(scaled_df)

# Comprobar la forma de los datos transformados
st.write("Shape of data after PCA:", X_train_pca.shape)