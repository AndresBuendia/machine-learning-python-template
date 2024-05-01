from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('../models/wine_quality_model.pkl')

# Mapeo inverso de etiquetas definido en tu exploración de datos
inverse_label_mapping = {0: 4, 1: 5, 2: 6, 3: 7, 4: 8}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    
    # Aplicar el mapeo inverso para convertir la predicción a la escala original
    output = inverse_label_mapping[prediction[0]]
    
    return render_template('index.html', prediction_text='Predicted Wine Quality: {}'.format(output))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)