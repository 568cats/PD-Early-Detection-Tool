from flask import Flask, render_template
app = Flask(__name__)
from flask import Flask, render_template, request, redirect, url_for
from PDimage1model import create_model, preprocess_image
from PDimage2model import create_model2
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
@app.route("/predictPDImage1", methods = ['GET','POST'])
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

    return render_template('PDimage_upload1.html', prediction=predicted_class_index)
#2nd image route
@app.route("/predictPDImage2", methods = ['GET','POST'])
def predict2():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = preprocess_image(file_path)


            model2 = create_model2()
            class_prediction = model2.predict(img)
            predicted_class_index = np.argmax(class_prediction)

    return render_template('PDimage_upload2.html', prediction=predicted_class_index)
@app.route('/', methods=['GET', 'POST'])
def home():
    
    return render_template('home.html')
#AD
@app.route('/ADimage1', methods=['GET', 'POST'])
def ADimage1():
    
    return render_template('ADimage_upload1.html')


@app.route('/ADimage2', methods=['GET', 'POST'])
def ADimage2():
    
    return render_template('ADimage_upload2.html')


@app.route('/ADaudio1', methods=['GET', 'POST'])
def ADaudio1():
    
    return render_template('ADaudio_upload1.html')


@app.route('/ADaudio2', methods=['GET', 'POST'])
def ADaudio2():
    
    return render_template('ADaudio_upload2.html')

#PD
@app.route('/PDimage1', methods=['GET', 'POST'])
def PDimage1():
    
    return render_template('PDimage_upload1.html')



@app.route('/PDimage2', methods=['GET', 'POST'])
def PDimage2():
    
    return render_template('PDimage_upload2.html')

@app.route('/PDaudio1', methods=['GET', 'POST'])
def PDaudio1():
    
    return render_template('PDaudio_upload1.html')

@app.route('/PDaudio2', methods=['GET', 'POST'])
def PDaudio2():
    
    return render_template('PDaudio_upload2.html')

@app.route('/result', methods=['GET', 'POST'])
def result():
    
    return render_template('result.html')

if __name__ == "__main__":
    app.run(debug=True)
