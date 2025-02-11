from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load your model
model = load_model('NEW_INCEPTION_model.keras')

# Define the class names
class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']


def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final


def preprocess_image(image):
    image = cv2.resize(image, (256, 256))
    image = apply_clahe(image)
    image = img_to_array(image)
    image = image / 255.0
    image = tf.expand_dims(image, axis=0)
    return image


def predict(image):
    image_np = np.array(image)
    clahe_image = apply_clahe(image_np)
    processed_image = preprocess_image(clahe_image)
    prediction = model.predict(processed_image)
    class_idx = tf.argmax(prediction, axis=1).numpy()[0]
    confidence = tf.reduce_max(prediction).numpy()
    return image_np, clahe_image, class_names[class_idx], f"{confidence:.2f}"


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        actual_class = request.form['actual_class']
        img = Image.open(file.stream)

        original_image, clahe_image, predicted_class, confidence = predict(img)

        # Convert original image to base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
        original_image_b64 = base64.b64encode(buffer).decode('utf-8')

        # Convert CLAHE image to base64
        _, buffer = cv2.imencode('.png', cv2.cvtColor(clahe_image, cv2.COLOR_RGB2BGR))
        clahe_image_b64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'original_image': original_image_b64,
            'clahe_image': clahe_image_b64,
            'actual_class': actual_class,
            'predicted_class': predicted_class,
            'confidence': confidence
        })

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
