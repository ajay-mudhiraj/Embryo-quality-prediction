# -*- coding: utf-8 -*-
"""
Created on Wed May  7 13:34:03 2025

@author: Ajay
"""

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# === Flask App Initialization ===
app = Flask(__name__)

# === Load Trained Model ===
MODEL_PATH = 'embryo_model_final.h5'  # Make sure this path is correct
model = load_model(MODEL_PATH)

# === Class labels used during training ===
class_labels = [
    '8_cells_Grade_A', '8_cells_Grade_B', '8_cells_Grade_C',
    'morula_Grade_A', 'morula_Grade_B', 'morula_Grade_C',
    'blastocyst_Grade_A', 'blastocyst_Grade_B', 'blastocyst_Grade_C',
    'abnormal'
]

# === Image input configuration ===
IMG_HEIGHT = 224
IMG_WIDTH = 224

# === Prediction endpoint ===
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    # Preprocess the image
    img = load_img(file, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_index]
    confidence = float(np.max(predictions))

    # Separate main class and subclass
    if predicted_label == 'abnormal':
        main_class = 'abnormal'
        subclass = None
    else:
        parts = predicted_label.split('_Grade_')
        main_class = parts[0]
        subclass = f"Grade_{parts[1]}"

    return jsonify({
        'predicted_class': predicted_label,
        'main_class': main_class,
        'subclass': subclass,
        'confidence': round(confidence, 4)
    })

# === Start the Flask app ===
if __name__ == '__main__':
    app.run(debug=True, port=5000)
