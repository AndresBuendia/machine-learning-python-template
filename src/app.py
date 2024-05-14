import streamlit as st
import pandas as pd
import pickle

# Cargar el modelo entrenado
model_path = 'models/wine_quality_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Título de la aplicación
st.title("Predicción de la Calidad del Vino")

# Descripción de la aplicación
st.write("""
Esta aplicación predice la calidad del vino basada en varias características químicas.
Por favor, ingresa los valores de las características para obtener la predicción de calidad.
""")

# Función para hacer predicciones
def predict_quality(features):
    prediction = model.predict(features)
    return prediction

# Crear un formulario para ingresar las características del vino
with st.form(key='wine_form'):
    fixed_acidity = st.number_input('Acidez fija', min_value=0.0, step=0.1)
    volatile_acidity = st.number_input('Acidez volátil', min_value=0.0, step=0.01)
    citric_acid = st.number_input('Ácido cítrico', min_value=0.0, step=0.01)
    residual_sugar = st.number_input('Azúcar residual', min_value=0.0, step=0.1)
    chlorides = st.number_input('Cloruros', min_value=0.0, step=0.001)
    free_sulfur_dioxide = st.number_input('Dióxido de azufre libre', min_value=0, step=1)
    total_sulfur_dioxide = st.number_input('Dióxido de azufre total', min_value=0, step=1)
    density = st.number_input('Densidad', min_value=0.0, step=0.0001)
    pH = st.number_input('pH', min_value=0.0, step=0.01)
    sulphates = st.number_input('Sulfatos', min_value=0.0, step=0.01)
    alcohol = st.number_input('Alcohol', min_value=0.0, step=0.1)
    
    submit_button = st.form_submit_button(label='Predecir Calidad')

# Procesar la entrada y hacer la predicción
if submit_button:
    input_data = pd.DataFrame({
        'fixed_acidity': [fixed_acidity],
        'volatile_acidity': [volatile_acidity],
        'citric_acid': [citric_acid],
        'residual_sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free_sulfur_dioxide': [free_sulfur_dioxide],
        'total_sulfur_dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]
    })
    
    prediction = predict_quality(input_data)
    st.write(f"La calidad predicha del vino es: {prediction[0]}")


