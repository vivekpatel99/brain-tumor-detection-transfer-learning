import math
import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

from src.models import resnet50
from utils.utils import pad_cls_id_bbx

load_dotenv()

import logging

import tensorflow as tf
from hydra import compose, initialize
from tensorflow.keras import layers

from utils import utils
from utils.logs import get_logger
from utils.prepare_dataset import AnnotationProcessor

tf.random.set_seed(42)
with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config")
    
DATASET_DIRS = Path(cfg.DATASET.DATASET_DIR)
TRAIN_DIR = Path(cfg.DATASET_DIRS.TRAIN_DIR)
VALIDATION_DIR = Path(cfg.DATASET_DIRS.VALIDATION_DIR)
TEST_DIR = Path(cfg.DATASET_DIRS.TEST_DIR)
CHECK_POINT_DIR = Path(cfg.OUTPUTS.CHECKPOINT_PATH)

IMG_SIZE = cfg.TRAIN.IMG_SIZE
BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
LOG_DIR = cfg.OUTPUTS.LOG_DIR
NUM_EPOCHS = cfg.TRAIN.NUM_EPOCHS
LEARNING_RATE = cfg.TRAIN.LEARNING_RATE

CLASS_NAME = [
    'label0',
    'label1',
    'label2'
]
NUM_CLASSES = len(CLASS_NAME)
class_map = {k: v for k, v in enumerate(CLASS_NAME)}

data_augmentation = tf.keras.Sequential([
    layers.RandomBrightness(0.1),
    layers.RandomContrast(0.1),
    layers.RandomSaturation(0.1),
    layers.RandomHue(0.1)
])

def _prepare_ds(img_list, cls_id_list, bbx_list, is_train=False):
    padded_class_ids, padded_bbx = pad_cls_id_bbx(cls_id_list, bbx_list)
    datasets = tf.data.Dataset.from_tensor_slices((img_list, padded_class_ids, padded_bbx))
    ds = datasets.map(utils.load_dataset, num_parallel_calls=tf.data.AUTOTUNE) 
    ds = ds.map(utils.preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        ds = ds.map(lambda x,y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(buffer_size=ds.cardinality().numpy(), reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds



def main():
    log = get_logger(__name__, log_level=logging.INFO)
    found_gpu = tf.config.list_physical_devices('GPU')
    if not found_gpu:
        log.error("No GPU found")
        raise Exception("No GPU found")
    
    mlflow.set_experiment("/brain-tumor-resnet50")
    mlflow.tensorflow.autolog(log_models=True, log_datasets=False)

    prepare_train_dataset = AnnotationProcessor(annotation_file= str(TRAIN_DIR/'_annotations.csv'))
    _class_map = {v: k for k, v in enumerate(CLASS_NAME)}
    train_images, train_class_ids, train_bboxes  = prepare_train_dataset.process_annotations(image_dir=TRAIN_DIR, class_id_map=_class_map)
    train_ds = _prepare_ds( train_images, train_class_ids, train_bboxes, is_train=True)

    prepare_valid_dataset = AnnotationProcessor(annotation_file= str(VALIDATION_DIR/'_annotations.csv'))
    valid_images, valid_class_ids, valid_bboxes  = prepare_valid_dataset.process_annotations(image_dir=VALIDATION_DIR, class_id_map=_class_map)
    valid_ds = _prepare_ds(valid_images, valid_class_ids, valid_bboxes, is_train=False)

    METRICS = [
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.AUC(name='AUC', multi_label=True), 
        tf.keras.metrics.F1Score(name='f1_score',average='weighted'),
    ]

    to_monitor = 'val_loss'
    mode = 'min'
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, 
                                                patience=2, 
                                                monitor=to_monitor,
                                                mode=mode,
                                                min_lr=1e-6,
                                                verbose=1),

        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(str(CHECK_POINT_DIR), "ckpt_{epoch}.keras") ,
                                            save_weights_only=False,
                                            save_best_only=True,
                                            monitor=to_monitor,
                                            mode=mode,
                                            verbose=1),
                                            
        tf.keras.callbacks.EarlyStopping(monitor=to_monitor, 
                                        patience=10,
                                        mode=mode, 
                                        restore_best_weights=True),

    ]
    model = resnet50.final_model(input_shape=(IMG_SIZE, IMG_SIZE,3), num_classes=NUM_CLASSES)

    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
    loss={'classification': tf.keras.losses.BinaryCrossentropy(from_logits=False), 'bounding_box': tf.keras.losses.MeanSquaredError()},
    metrics={'classification': METRICS, 'bounding_box': 'mse'})  # Use IoU metric
   
    # Get the length of the training set
    length_of_training_dataset = len(train_images)
    # Get the length of the validation set
    length_of_validation_dataset = len(valid_images)
    # Get the steps per epoch 
    steps_per_epoch = math.ceil(length_of_training_dataset/BATCH_SIZE)
    # get the validation steps (per epoch) 
    validation_steps = math.ceil(length_of_validation_dataset/BATCH_SIZE)

    model.fit(
        train_ds,
        # steps_per_epoch=steps_per_epoch,
        epochs=NUM_EPOCHS,
        validation_data=valid_ds,
        # validation_steps=validation_steps,
        batch_size=BATCH_SIZE,
        callbacks=[callbacks],
    )


    # Evaluation
    prepare_test_dataset = AnnotationProcessor(annotation_file= str(TEST_DIR/'_annotations.csv'))
    test_images, test_class_ids, test_bboxes  = prepare_test_dataset.process_annotations(image_dir=TEST_DIR, class_id_map=_class_map)
    test_ds = _prepare_ds(test_images, test_class_ids, test_bboxes, is_train=False)
    results = model.evaluate(test_ds, return_dict=True)
    mlflow.log_metrics(results)

if __name__ == '__main__':
    main()