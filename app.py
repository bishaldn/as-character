from flask import Flask, render_template, request
import keras
import keras.utils
import tensorflow as tf
from keras.models import load_model

import cv2
import numpy as np


model= load_model('assamese-ch.h5')

app = Flask(__name__)
@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')
@app.route('/',methods=['POST'])







def predict():
    imagefile=  request.files['imagefile']
    image_path= "./image/"+ imagefile.filename
    imagefile.save(image_path)
    test_img = cv2.imread(image_path)
    test_img=cv2.resize(test_img,(40,40))
    
    test_img_arr = tf.keras.utils.img_to_array(test_img)
    test_img_arr = np.expand_dims(test_img_arr, axis = 0)
    prediction = model.predict(test_img_arr)
    res = [np.argmax(pred) for pred in prediction]

    return render_template('index.html', prediction=res )



if __name__ == '__main__':
    app.run(debug=True)