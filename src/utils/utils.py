import keras
import tensorflow as tf


def preprocess(images, classes):
    processed_image = keras.applications.resnet.preprocess_input(images)
    return processed_image, classes