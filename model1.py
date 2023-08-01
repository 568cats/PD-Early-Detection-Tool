import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
#Definings the model for the image rec
def create_model():
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    resnet_base.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer1 = tf.keras.layers.Dense(64, activation='relu')
    dropout_layer1 = tf.keras.layers.Dropout(0.6)
    prediction_layer_2 = tf.keras.layers.Dense(16, activation="relu")
    prediction_layer_3 = tf.keras.layers.Dense(2, activation='softmax')

    model = tf.keras.Sequential([
        resnet_base,
        global_average_layer,
        prediction_layer1,
        dropout_layer1,
        prediction_layer_2,
        prediction_layer_3
    ])

    model.load_weights('model_weights1.h5')

    return model

#Pre proccessing for the image recognition
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


