#!/usr/bin/env python
from flask import Flask, render_template, request
from markupsafe import Markup
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from disease import disease_dic
app = Flask(__name__)

# Class labels
disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']


# Load models
mobilenet = load_model('MobileNetV2_plants.h5')
inception = load_model('InceptionV3_plants.h5')

def predict_label(img_path):
    test_image = image.load_img(img_path, target_size=(224,224))
    test_image = image.img_to_array(test_image)/255.0
    test_image = test_image.reshape(1, 224, 224, 3)

    predict_x = inception.predict(test_image) 
    classes_x = np.argmax(predict_x, axis=1)
    return disease_classes[classes_x[0]]

def predict_labels(img_path):
    test_image = image.load_img(img_path, target_size=(224,224))
    test_image = image.img_to_array(test_image)/255.0
    test_image = test_image.reshape(1, 224, 224, 3)

    predict_x = mobilenet.predict(test_image) 
    classes_x = np.argmax(predict_x, axis=1)
    return disease_classes[classes_x[0]]

@app.route("/")
@app.route("/first")
def first():
    return render_template('first.html')
    
@app.route("/login")
def login():
    return render_template('login.html')    

@app.route("/graph")
def graph():
    return render_template('graph.html')


@app.route("/chart")
def chart():
    return render_template('chart.html')

@app.route("/performance")
def performance():
    return render_template('performance.html')

@app.route("/index", methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files.get('my_image')
        model = request.form.get('model')
        
        # Ensure both image and model were provided
        if img and model:
            img_path = "static/tests/" + img.filename	
            img.save(img_path)

            if model == 'InceptionV3':
                predict_result = predict_label(img_path)
            elif model == 'MobileNetV2':
                predict_result = predict_labels(img_path)
            else:
                predict_result = "Unknown model selected"
            if predict_result not in disease_dic:
                return f"Error: Key '{predict_result}' not found in disease_dic", 500

            predictions = Markup(str(disease_dic[predict_result]))
            print(predictions)
            return render_template("result.html", prediction=predictions, img_path=img_path, model=model)
        
        # Handle case if image or model is missing
        return "Image or model selection is missing. Please try again.", 400  # 400: Bad Request
    
    # Fallback if the request method is not POST
    return "Invalid request method. Please submit the form correctly.", 405  # 405: Method Not Allowed

if __name__ == '__main__':
    app.run(debug=True)
