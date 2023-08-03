from flask import Flask, render_template
app = Flask(__name__)
from flask import Flask, render_template, request, redirect, url_for
from PDimage1model import create_model, preprocess_image
from PDimage2model import create_model2
import os
import numpy as np
import librosa
from PDaudio1Model import preprocess_audio, create_audio_model
from ADimage1model import create_AD_model_1
from flask import request
from PIL import Image
import io
import base64
import re
import numpy as np
import os
from flask import session

app = Flask(__name__)
app.secret_key = 'secret'
# Image rec model is model1
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
def init():
    global graph
    graph = tf.get_default_graph()

# Predict route for the image rec model 
@app.route('/predict_AD_image_1', methods = ['GET', 'POST'])
def predict_AD_image_1():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = preprocess_image(file_path)
            model1 = create_AD_model_1()
            class_prediction = model1.predict(img)
            predicted_class_index = np.argmax(class_prediction)
    return render_template('ADimage_upload1.html', prediction=predicted_class_index)

@app.route('/predictPDImage1', methods=['GET', 'POST'])
def predictPDImage1():
    if request.method == 'POST':
        dataUrlPattern = re.compile('data:image/(png|jpeg);base64,(.*)$')
        ImageData = request.form.get('drawing')

        ImageData = dataUrlPattern.match(ImageData).group(2)
        if ImageData is None or len(ImageData) == 0:  # If no image data was provided
            return render_template('home.html', prediction="No image data provided.")

        # We have image data, process it
        data = base64.b64decode(ImageData)
        image = Image.open(io.BytesIO(data))
        filename = "some_unique_filename.png"  # You may want to use a unique filename in production
        file_path = os.path.join('static/images', filename)
        image.save(file_path)
        img = preprocess_image(file_path)
        model1 = create_AD_model_1()
        class_prediction = model1.predict(img)
        session['PDprediction1'] = class_prediction.tolist()
        predicted_class_index = np.argmax(class_prediction)
        return render_template('PDimage_upload1.html', prediction=predicted_class_index)
    
    # If not POST method
    return render_template('PDimage_upload1.html')

@app.route("/predictPDImage11", methods = ['GET','POST'])
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
            session['PDprediction11'] = class_prediction.tolist()
            predicted_class_index = np.argmax(class_prediction)

    return render_template('PDimage_upload1.html', prediction=predicted_class_index)



@app.route('/predictPDImage2', methods=['GET', 'POST'])
def predictPDImage2():
    if request.method == 'POST':
        dataUrlPattern = re.compile('data:image/(png|jpeg);base64,(.*)$')
        ImageData = request.form.get('drawing')

        ImageData = dataUrlPattern.match(ImageData).group(2)
        if ImageData is None or len(ImageData) == 0:  # If no image data was provided
            return render_template('home.html', prediction="No image data provided.")

        # We have image data, process it
        data = base64.b64decode(ImageData)
        image = Image.open(io.BytesIO(data))
        filename = "some_unique_filename.png"  # You may want to use a unique filename in production
        file_path = os.path.join('static/images', filename)
        image.save(file_path)
        img = preprocess_image(file_path)
        model1 = create_AD_model_1()
        class_prediction = model1.predict(img)
        session['PDprediction2'] = class_prediction.tolist()
        predicted_class_index = np.argmax(class_prediction)
        return render_template('PDimage_upload1.html', prediction=predicted_class_index)
    
    # If not POST method
    return render_template('PDimage_upload1.html')
@app.route("/predictPDImage22", methods = ['GET','POST'])
def predictPDImage22():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = preprocess_image(file_path)


            model2 = create_model2()
            class_prediction = model2.predict(img)
            session['PDprediction22'] = class_prediction.tolist()
            predicted_class_index = np.argmax(class_prediction)

    return render_template('PDimage_upload2.html', prediction=predicted_class_index)
@app.route('/', methods=['GET', 'POST'])
def home():
    
    return render_template('home.html')
#AD
@app.route('/ADimage1', methods=['GET', 'POST'])
def ADimage1():
    
    return render_template('ADimage_upload1.html')




@app.route('/ADaudio1', methods=['GET', 'POST'])
def ADaudio1():
    
    return render_template('ADaudio_upload1.html')



#PD
@app.route('/PDimage1', methods=['GET', 'POST'])
def PDimage1():
    
    return render_template('PDimage_upload1.html')



@app.route('/PDimage2', methods=['GET', 'POST'])
def PDimage2():
    
    return render_template('PDimage_upload2.html')

# Handles Parkinson's Disease Audio Classifiction. Accepts either a .wav or mp3 file
ALLOWED_EXTENSIONS_AUDIO = ['wav', 'mp3']
# Check if file is valid
def allowed_audio_file(filename): 
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_AUDIO

@app.route('/PDaudio1', methods=['GET', 'POST'])
def PDaudio1():
    predicted_class_index = "Submit a file and click 'Upload' to see your results."  # initialize to default value
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_audio_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/audio', filename)
            file.save(file_path)
            
            # Load the audio file
            audio, sample_rate = librosa.load(file_path, sr=None)

            # Preprocess the audio file by performing 
            audio_preprocessed = preprocess_audio(audio)

            # Pass the preprocessed audio file to the model
            model_audio = create_audio_model()
            class_prediction = model_audio.predict(audio_preprocessed)
            predicted_class_index = np.argmax(class_prediction)

    return render_template('PDaudio_upload1.html', prediction=predicted_class_index)

    


@app.route('/result', methods=['GET'])
def result():
    # Fetch the predictions from the session
    PDprediction1 = np.array(session.get('PDprediction1'))
    PDprediction11 = np.array(session.get('PDprediction11'))
    PDprediction2 = np.array(session.get('PDprediction2'))
    PDprediction22 = np.array(session.get('PDprediction22'))
    # Pass the predictions into the template
    return render_template('result.html', PDprediction1 = PDprediction1, PDprediction11 = PDprediction11, PDprediction2 = PDprediction2, PDprediction22 = PDprediction22)

if __name__ == "__main__":
    app.run(debug=True)
