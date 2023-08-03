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
from ADaudio1model import preprocess_ad_audio, ad_audio_model
from flask import request, jsonify
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
@app.route('/clear', methods=['POST'])
def clear_session():

    session.clear()

    return redirect(url_for('home'))

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
            session['ADprediction1'] = class_prediction.flatten().tolist()
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
        session['PDprediction1'] = class_prediction.flatten().tolist()
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
            session['PDprediction1'] = class_prediction.flatten().tolist()
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
        session['PDprediction2'] = class_prediction.flatten().tolist()
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
            session['PDprediction2'] = class_prediction.flatten().tolist()
            predicted_class_index = np.argmax(class_prediction)

    return render_template('PDimage_upload2.html', prediction=predicted_class_index)
@app.route('/', methods=['GET', 'POST'])
def home():
    
    return render_template('home.html')
#AD
@app.route('/ADimage1', methods=['GET', 'POST'])
def ADimage1():
    
    return render_template('ADimage_upload1.html')




ALLOWED_EXTENSIONS_AD_AUDIO = ['wav', 'mp3']
def allowed_ad_audio_file(filename): 
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_AD_AUDIO

@app.route('/ADaudio1', methods=['GET', 'POST'])
def ADaudio1():
    predicted_class_index = "Submit a file and click 'Upload' to see your results."  # initialize to default value
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_ad_audio_file(file.filename):
            filename = file.filename
            file_path = os.path.join('static/audio', filename)
            file.save(file_path)
            
            # Load the audio file
            audio, sample_rate = librosa.load(file_path, sr=None)

            # Preprocess the audio file by performing 
            audio_preprocessed = preprocess_ad_audio(audio)

            # Pass the preprocessed audio file to the model
            model_audio = ad_audio_model()
            class_prediction = model_audio.predict_proba(audio_preprocessed.reshape(1, -1))
            session['ADprediction2'] = class_prediction.tolist()
            predicted_class_index = np.argmax(class_prediction)
            #os.remove(file_path)
    return render_template('ADaudio_upload1.html', prediction=predicted_class_index)



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

            
            #just a quick test if the model works
            exampleFeature = np.array([-1.53323887, -1.20521085, -1.11614902, -1.20484865, -1.76998794,
        0.31389406, -0.72827947, -0.38706982, -1.32216396])
            
            model_audio = create_audio_model()
            class_prediction = model_audio.predict(exampleFeature.reshape(1, -1))

            #session['PDpredictionAudio'] = class_prediction.tolist()

            model_audio = create_audio_model()
            #class_prediction = model_audio.predict(audio_preprocessed)
            session['PDpredictionAudio'] = class_prediction.tolist()
            predicted_class_index = np.argmax(class_prediction)

    return render_template('PDaudio_upload1.html', prediction=predicted_class_index)

    


@app.route('/PDresult', methods=['GET'])
def PDresult():
    # Fetch the predictions from the session
    PDprediction1 = np.array(session.get('PDprediction1')).tolist()
    PDprediction11 = np.array(session.get('PDprediction11')).tolist()
    PDprediction2 = np.array(session.get('PDprediction2')).tolist()
    PDprediction22 = np.array(session.get('PDprediction22')).tolist()
    PDpredictionAudio = np.array(session.get('PDpredictionAudio')).tolist()

    # Pass the predictions into the template
    return render_template('PDresult.html', PDprediction1 = PDprediction1, PDprediction2 = PDprediction2, PDpredictionAudio= PDpredictionAudio[0])


@app.route('/ADresult', methods=['GET'])
def ADresult():
    # Fetch the predictions from the session
    ADprediction1 = np.array(session.get('ADprediction1')).tolist()
    ADprediction2 = np.array(session.get('ADprediction2')).tolist()
    

    # Pass the predictions into the template
    return render_template('ADresult.html', ADprediction1 = ADprediction1, ADprediction2 = ADprediction2[0])
if __name__ == "__main__":
    app.run(debug=True)
