import tensorflow as tf
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input

def create_AD_model_1():
    resnet_base = ResNet152(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    resnet_base.trainable = False

    # Add a global spatial average pooling layer
    ## YOUR CODE HERE
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    # Add a fully-connected layer
    ## YOUR CODE HERE
    prediction_layer1 = tf.keras.layers.Dense(128, activation='relu')
    # Add the final classification layer
    dropout_layer1 = tf.keras.layers.Dropout(0.5)
    prediction_layer_2 = tf.keras.layers.Dense(16, activation="relu")
    prediction_layer_3 = tf.keras.layers.Dense(2, activation='softmax')

    # Build the model you will train
    model = tf.keras.Sequential([
        resnet_base,
        global_average_layer,
        prediction_layer1,
        dropout_layer1,
        prediction_layer_2,
        prediction_layer_3
    ])
    model.load_weights('AD_model_image_1.h5')
    return model

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array