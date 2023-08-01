import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
#Definings the model for the image rec
def create_model2():
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    resnet_base.trainable = False

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer1 = tf.keras.layers.Dense(128, activation='relu')
    dropout_layer1 = tf.keras.layers.Dropout(0.5)
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

    model.load_weights('model2_weights.h5')

    return model