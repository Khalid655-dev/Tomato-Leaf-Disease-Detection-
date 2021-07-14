
import sys
import os
import glob
import re
import numpy as np


##os.environ['KERAS_BACKEND'] = 'theano'

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# from gevent.pywsgi import WSGIServer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH =  os.path.join(BASE_DIR, 'model_inception.h5')

# Load your trained model
import tensorflow as tf
model = tf.keras.models.load_model(MODEL_PATH)


def model_predict(img_path, model):

    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)

    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    label = np.argmax(preds,axis=1)
    preds = label[0]
    if preds == 0:
        preds = "Tomato___Bacterial_spot"
    if preds == 1:
        preds = "Tomato___Early_blight"
    if preds == 2:
        preds = "Tomato___Late_blight"
    if preds == 3:
        preds = "Tomato___Leaf_Mold"
    if preds == 4:
        preds = "Tomato___Septoria_leaf_spot"
    if preds == 5:
        preds = "Tomato___Spider_mites Two-spotted_spider_mite"
    if preds == 6:
        preds = "Tomato___Target_Spot"
    if preds == 7:
        preds = "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
    if preds == 8:
        preds = "Tomato___mosaic_virus"
    elif preds == 9:
        preds = "Tomato__healthy"
    print(preds)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        f = request.files['file']

        basepath = os.path.dirname(__file__)
        if not os.path.exists('uploads'):
            os.mkdir('uploads')
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Prediction
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


if __name__ == '__main__':
    app.run()
