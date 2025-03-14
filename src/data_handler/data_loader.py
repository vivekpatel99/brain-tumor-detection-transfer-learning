import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class DataLoader:
    def __init__(self, img_list:list[str], cls_id_list:list[list], bbx_list:list[list], img_size=240, num_classes=3):
        self.img_list = img_list
        self.cls_id_list = cls_id_list
        self.image_size = img_size
        self.bbx_list = bbx_list
        self.num_classes = num_classes
        self.multi_hot_class_ids = None
        self.data_augmentation = tf.keras.Sequential([
            layers.RandomBrightness(0.3, seed=42), # seed can be added for reproducibility
            layers.RandomContrast(0.3, seed=42),
            layers.RandomSaturation(0.3, seed=42),
            layers.RandomHue(0.3, seed=42),
        ])

    
    def load_image(self, image_path) -> tf.Tensor:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            image = tf.image.resize_with_pad(image, self.image_size, self.image_size) # background is always black so padding is fine
        image = tf.cast(image, tf.float32) 
        return image

    def load_dataset(self, image, class_ids, bbox):
        tf_image = self.load_image(image)
        # multi_hot = tf.reduce_max(tf.one_hot(tf.cast(class_ids, tf.int32), self.num_classes), axis=0 )  # Shape: (NUM_CLASSES,)
        return  tf_image, bbox #(class_ids , bbox)
        # return  tf_image,  {'classes': multi_hot, 'boxes': tf.cast(bbox, tf.float32)}
    
    def _common_loader(self)->tf.data.Dataset:
        padded_bbx = self.pad_bbx()
        self.multi_hot_class_ids = self.create_multi_hot()
        datasets = tf.data.Dataset.from_tensor_slices((self.img_list, self.multi_hot_class_ids, padded_bbx))
        ds = datasets.map(self.load_dataset, num_parallel_calls=tf.data.AUTOTUNE) 
        return ds
    
    def load_train_dataset(self)->tf.data.Dataset:
        ds = self._common_loader()
        ds =  ds.map(lambda x, y: (self.data_augmentation(x),y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.shuffle(buffer_size=ds.cardinality().numpy())

    def load_val_dataset(self) ->tf.data.Dataset:
        ds = self._common_loader()
        return ds

    def pad_bbx(self):
        # padded_class_ids = keras.preprocessing.sequence.pad_sequences(self.cls_id_list, padding='post', dtype='int32')
        return keras.preprocessing.sequence.pad_sequences(self.bbx_list, padding='post', dtype='float32')
    
    def multi_hot_encode_single_id(self, class_ids):
        encoded = np.zeros(self.num_classes, dtype=np.float32)
        encoded[class_ids] = 1.0
        return encoded
    
    def create_multi_hot(self) -> np.ndarray:
        """
        Creates a multi-hot encoded vector for the given class IDs.

        Args:
            class_ids: A list of class IDs (e.g., [0, 1, 2], [1], [0, 1]).

        Returns:
            A TensorFlow tensor representing the multi-hot encoded vector.
        """
        return np.array([self.multi_hot_encode_single_id(sample) for sample in self.cls_id_list])