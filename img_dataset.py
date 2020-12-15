# from __future__ import print_function

# import tensorflow as tf
# import os

# # Dataset Parameters - CHANGE HERE
# MODE = 'folder' # or 'file', if you choose a plain text file (see above).
# DATASET_PATH = '/Users/gisturiz/Desktop/images' # the dataset file or root folder path.

# # Image Parameters
# N_CLASSES = 4 # CHANGE HERE, total number of classes
# IMG_HEIGHT = 224 # CHANGE HERE, the image height to be resized to
# IMG_WIDTH = 224 # CHANGE HERE, the image width to be resized to
# CHANNELS = 3 # The 3 color channels, change to 1 if grayscale

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 4

model = keras.Sequential([
    layers.Input((IMG_HEIGHT, IMG_WIDTH, 1)),
    layers.Conv2D(16, 3, padding='same'),
    layers.Conv2D(32, 3, padding='same'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(10),
])

# Method 1

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'Users/gisturiz/Desktop/images',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset='training'
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    'Users/gisturiz/Desktop/images',
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True,
    seed=123,
    validation_split=0.1,
    subset='validation'
)