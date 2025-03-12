import tensorflow as tf


def pad_cls_id_bbx(class_id_list, bbox_list):
    padded_class_ids = tf.keras.preprocessing.sequence.pad_sequences(
    class_id_list, padding='post', dtype='int32')
    padded_bbx = tf.keras.preprocessing.sequence.pad_sequences(
        bbox_list, padding='post', dtype='float32')
    
    return padded_class_ids, padded_bbx

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    return image

def load_dataset(image, class_ids, bbox, NUM_CLASSES=3):
    tf_image = load_image(image)
    multi_hot = tf.reduce_max(
    tf.one_hot(tf.cast(class_ids, tf.int32), NUM_CLASSES), 
    axis=0
)  # Shape: (NUM_CLASSES,)
    return  tf_image, (multi_hot, bbox)

def preprocess(images, classes):
    processed_image = tf.keras.applications.resnet.preprocess_input(images)
    return processed_image, classes