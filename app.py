from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the model
MODEL_PATH = 'model/best_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Class names (replace with your actual class names in correct order)
class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

# Predict function
def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds)
    return pred_class, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    file_url = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join('static', file.filename)
            file.save(filepath)
            prediction, confidence = model_predict(filepath)
            file_url = filepath
            return render_template('index.html', prediction=prediction, confidence=confidence, image_path=file_url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
