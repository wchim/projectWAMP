from __future__ import division, print_function
import os
import numpy as np
from PIL import Image, ImageOps
import cv2 as cv
import base64
from io import BytesIO
import tensorflow as tf
import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
from keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# BEGIN CONSTANTS
detection_model_path = 'models/detection_model.pb'
classification_model_path = 'models/classification_model.h5'

graph_def = tf.GraphDef()

# with open('detection_labels.txt', 'r') as file:
#     detection_labels = file.read().split('\n')

with open('classification_labels.txt', 'r') as file:
    classification_labels = file.read().split('\n')

outputs = ('num_detections','detection_classes','detection_scores','detection_boxes')
# END CONSTANTS

# BEGIN INITIALIZATIONS
app = Flask(__name__)

tf_config = os.environ.get('TF_CONFIG')
sess = tf.Session(config=tf_config)

# The one g to rule them all.
g = tf.get_default_graph()

# Load graph from a gfile    
def load_detection_model():
    with tf.gfile.GFile(detection_model_path, "rb") as f:
        graph_def.ParseFromString(f.read())
    return None

with g.as_default():
    set_session(sess)
    detection_model = load_detection_model()

# Loads the model onto g.
with g.as_default():
    set_session(sess)
    classification_model = load_model(classification_model_path)
    classification_model._make_predict_function()

print('Model loaded. Check http://127.0.0.1:5000/')
# END INITIALIZATIONS

# BEGIN MODELING FUNCTION DEFINITIONS
# def encode_image(image_path):
#     input_image_binary = open(image_path, 'rb')

#     base64_encoding = base64.b64encode(input_image_binary.read())
#     base64_encoding_str = str(base64_encoding)
#     base64_encoding_str = base64_encoding_str.replace("'", "")
#     base64_encoding_str = base64_encoding_str.lstrip("b")

#     return base64_encoding_str

# def load_encoding(image_path, base64_encoding_str):
#     base64_encoding_str = encode_image(image_path)
#     input_image = Image.open(BytesIO(base64.b64decode(base64_encoding_str)))

#     return input_image

def preprocess_image(image_path):
    # input_image = load_encoding(image_path, base64_encoding_str)
    input_image = Image.open(image_path)
    input_image = image.img_to_array(input_image)
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)

    frame_dim = input_image.shape
    max_dimension = max(frame_dim)

    width_padding = max_dimension - frame_dim[1]
    height_padding = max_dimension - frame_dim[0]

    left_padding = width_padding // 2
    right_padding = width_padding - left_padding
    top_padding = height_padding // 2
    bottom_padding = height_padding - top_padding

    padding = (left_padding, top_padding, right_padding, bottom_padding)

    input_image = image.array_to_img(input_image)
    padded_image = ImageOps.expand(input_image, padding, (255,255,255,255))
    resized_image = padded_image.resize((224,224), Image.ANTIALIAS)

    return resized_image

def preprocess_subimage(subimage):
    frame_dim = subimage.shape
    max_dimension = max(frame_dim)

    width_padding = max_dimension - frame_dim[1]
    height_padding = max_dimension - frame_dim[0]

    left_padding = width_padding // 2
    right_padding = width_padding - left_padding
    top_padding = height_padding // 2
    bottom_padding = height_padding - top_padding

    padding = (left_padding, top_padding, right_padding, bottom_padding)

    subimage = image.array_to_img(subimage)
    padded_subimage = ImageOps.expand(subimage, padding, (255,255,255,255))
    resized_subimage = padded_subimage.resize((224,224), Image.ANTIALIAS)

    return resized_subimage

def detect_subimages(image_path, graph): 
    input_for_detection = []
    resized_image = preprocess_image(image_path)
    resized_image = image.img_to_array(resized_image)
    input_for_detection.append(resized_image)

    with g.as_default():
        set_session(sess)

        tf.import_graph_def(graph_def, name="")
    
        detections = sess.run([sess.graph.get_tensor_by_name(f'{op}:0') for op in outputs],
                              feed_dict={ 'image_tensor:0': input_for_detection })

    return detections

@app.route('/predict', methods=['GET', 'POST']) 
def crop_and_classify_subimages():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

    with g.as_default():
        set_session(sess)
        detections = detect_subimages(file_path, detection_model)

    num_detections = detections[0]
    detection_classes = detections[1]
    detection_scores = detections[2]
    detection_boxes = detections[3]

    cropped_image_list = []
    cropped_image_count = 0

    resized_image = preprocess_image(file_path)
    resized_image = image.img_to_array(resized_image)
    resized_height, resized_width = 224, 224

    for i in range(0,int(num_detections[0])):
        left = int(resized_width*detection_boxes[0][i][1])
        top = int(resized_height*detection_boxes[0][i][0])
        right = int(resized_width*detection_boxes[0][i][3])
        bottom = int(resized_height*detection_boxes[0][i][2])
    
        if detection_scores[0][i] >= 0.5:
            subimage = resized_image[top:bottom, left:right]
            cropped_image_list.append(subimage)
            cropped_image_count += 1

    if cropped_image_count == 0:
        return ("There are no detections, try another image")
    else:
        print("Number of confident detections: {}".format(cropped_image_count))

        predictions = []

        for cropped_image in cropped_image_list:
            resized_subimage = preprocess_subimage(cropped_image)
            resized_subimage = image.img_to_array(resized_subimage) 
            with g.as_default():
                set_session(sess)
                prediction = classification_model.predict(np.array([resized_subimage]))
                prediction_class = np.argmax(prediction)
                # predictions.append(prediction)
                predictions.append(classification_labels[prediction_class])
        return (str(predictions))
    return None
# END MODELING FUNCTION DEFINITIONS

# BEGIN HANDLER FUNCTIONS
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/background')
def background():
    return render_template('background.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/blue')
def blue():
    return render_template('blue.html')

@app.route('/green')
def green():
    return render_template('green.html')

@app.route('/orange')
def orange():
    return render_template('orange.html')

@app.route('/purple')
def purple():
    return render_template('purple.html')

@app.route('/red')
def red():
    return render_template('red.html')
# END HANDLER FUNCTIONS

# MAIN METHOD
if __name__ == '__main__':
    app.run(debug=True)
