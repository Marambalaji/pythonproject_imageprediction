from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)

model = hub.load("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4")


LABELS_URL = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
labels = requests.get(LABELS_URL).text.splitlines()


def preprocess_image(image_path):
    image = Image.open(image_path).resize((224, 224))
    image = np.array(image) / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image


def predict(image_path):
    image = preprocess_image(image_path)
    predictions = model(image)
    predicted_index = np.argmax(predictions)
    predicted_label = labels[predicted_index]
    return predicted_label

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image_path = "uploaded_image.jpg"
            file.save(image_path)
            predicted_object = predict(image_path)
            return render_template('result.html', object_name=predicted_object)
    return render_template('index.html')  # Display the upload form

@app.route('/result')
def result():

    return render_template('result.html', object_name="No prediction yet")

if __name__ == '__main__':
    app.run(debug=True)
