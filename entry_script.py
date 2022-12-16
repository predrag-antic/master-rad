import os, requests
import numpy as np
import tensorflow as tf
import joblib
import requests
from azureml.contrib.services.aml_request import rawhttp
from azureml.contrib.services.aml_response import AMLResponse

def init():
    global model, inputs_dc, prediction_dc

    model_root = os.getenv("AZUREML_MODEL_DIR")
    model_folder = "keras-model.pkl"

    if (model_root == None):
        model_root = ".."
    
    model_path = os.path.join(model_root, model_folder)
    model = joblib.load(model_path)
    print("Model is loaded")

@rawhttp
def run(request):
    classes = ["NORMAL", "PNEUMONIA"]

    if request.method == 'POST':
        image = request.form['name']
        image = process_image(image)
        image_batch = tf.data.Dataset.from_tensor_slices([tf.constant(image)]).batch(32)
        prediction = model.predict(image_batch)
        preds = []
        for i in range(len(prediction)):
            preds.append(get_pred_label(prediction[i], classes))
            
        print('Input:' + request.form['name'] + '; Prediction:' + preds[0])
        return preds[0]
    else:
        return AMLResponse('Not a POST method, try again.', 500)


def process_image(image_path, img_size=128):
    image = tf.image.decode_jpeg(requests.get(image_path).content, channels=1, name="jpeg_reader")
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[img_size, img_size])
    return image


def get_pred_label(prediction_probabilities, classes):
  return classes[np.argmax(prediction_probabilities)]