from flask import Flask, render_template
app = Flask(__name__)
from flask import Flask, render_template, request, redirect, url_for
from model1 import create_model, preprocess_image
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np




# Image rec model is model1
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def init():
    global graph
    graph = tf.get_default_graph()

# Predict route for the image rec model 
@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = preprocess_image(file_path)
            model1 = create_model()
            class_prediction = model1.predict(img)
            predicted_class_index = np.argmax(class_prediction)

    return render_template('home.html', prediction=predicted_class_index)

@app.route('/', methods=['GET', 'POST'])
def home():
    
    return render_template('home.html')

@app.route('/audio1', methods=['GET', 'POST'])
def audio1():
    
    return render_template('audio_upload1.html')

@app.route('/audio2', methods=['GET', 'POST'])
def audio2():
    
    return render_template('audio_upload2.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    
    return render_template('result.html')

if __name__ == "__main__":
    app.run(debug=True)
