import tensorflow as tf


class Preprocessor:
    def __init__(self, dataset:tf.data.Dataset):
        self.dataset = dataset
        
    def _resnet_preprocess(self, images, labels):
        processed_image = tf.keras.applications.resnet.preprocess_input(images)
        return processed_image, labels
    
    def preprocess(self)-> tf.data.Dataset:
        return self.dataset.map(self._resnet_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
         