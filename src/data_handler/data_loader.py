import keras
import tensorflow as tf
from tensorflow.keras import layers


class DataLoader:
    def __init__(self, img_list:list[str], cls_id_list:list[list], bbx_list:list[list], num_classes=3):
        self.img_list = img_list
        self.cls_id_list = cls_id_list
        self.bbx_list = bbx_list
        self.num_classes = num_classes
        self.data_augmentation = tf.keras.Sequential([
            layers.RandomBrightness(0.1),
            layers.RandomContrast(0.1),
            layers.RandomSaturation(0.1),
            layers.RandomHue(0.1)
        ])

    
    def load_image(self, image_path) -> tf.Tensor:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32) 
        return image

    def load_dataset(self, image, class_ids, bbox):
        tf_image = self.load_image(image)
        multi_hot = tf.reduce_max(tf.one_hot(tf.cast(class_ids, tf.int32), self.num_classes), axis=0 )  # Shape: (NUM_CLASSES,)
        return  tf_image, multi_hot  #(multi_hot, bbox)
        # return  tf_image,  {'classes': multi_hot, 'boxes': tf.cast(bbox, tf.float32)}
    
    def _common_loader(self)->tf.data.Dataset:
        padded_class_ids, padded_bbx = self.pad_cls_id_bbx()
        datasets = tf.data.Dataset.from_tensor_slices((self.img_list, padded_class_ids, padded_bbx))
        ds = datasets.map(self.load_dataset, num_parallel_calls=tf.data.AUTOTUNE) 
        return ds
    
    def load_train_dataset(self)->tf.data.Dataset:
        ds = self._common_loader()
        ds =  ds.map(lambda x, y: (self.data_augmentation(x),y), num_parallel_calls=tf.data.AUTOTUNE)
        return ds.shuffle(buffer_size=ds.cardinality().numpy())

    def load_val_dataset(self) ->tf.data.Dataset:
        ds = self._common_loader()
        return ds

    def pad_cls_id_bbx(self):
        """
        Pads class id and bounding box lists to the length of the longest in the batch.
        
        Args:
            class_id_list (list): List of class ids.
            bbox_list (list): List of bounding boxes.
        
        Returns:
            tuple: Padded class id list and padded bounding box list.
        """
        
        padded_class_ids = keras.preprocessing.sequence.pad_sequences(self.cls_id_list, padding='post', dtype='int32')
        padded_bbx = keras.preprocessing.sequence.pad_sequences(self.bbx_list, padding='post', dtype='float32')
        
        return padded_class_ids, padded_bbx