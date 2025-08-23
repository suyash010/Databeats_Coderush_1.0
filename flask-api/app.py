from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load your trained .keras model
model = load_model('schizophrenia_model.keras')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract feature inputs from form
        features = [float(request.form[f'feature{i}']) for i in range(1, 11)]  # adjust number
        input_data = np.array([features])  # shape: (1, N)
        
        prediction = model.predict(input_data)
        predicted_class = np.argmax(prediction, axis=1)[0]

        return render_template('index.html', result=f"Predicted Class: {predicted_class}")

    except Exception as e:
        return render_template('index.html', result=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
