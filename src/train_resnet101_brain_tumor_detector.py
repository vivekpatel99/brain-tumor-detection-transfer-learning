
from pathlib import Path
from re import I, L

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import tensorflow as tf
from flask.cli import F

tf.random.set_seed(42)

import os

import matplotlib.pyplot as plt
import mlflow
from hydra import compose, initialize
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec
from sklearn.metrics import classification_report

from data_handler.annotation_processor import AnnotationProcessor
from data_handler.data_loader import DataLoader
from data_handler.preprocessor import Preprocessor
from losses import binary_weighted_loss as _loss
from losses import iou_loss
from models.resnet101 import final_model
from utils.logs import get_logger
from utils.visualization_funcs import plot_auc_curve, plot_iou_histogram

with initialize(version_base=None, config_path="../conf"):
    cfg = compose(config_name="config")
    print(cfg.DATASET_DIRS.TRAIN_DIR)

DATASET_DIRS = Path(cfg.DATASET.DATASET_DIR)
TRAIN_DIR = Path(cfg.DATASET_DIRS.TRAIN_DIR)
VALIDATION_DIR = Path(cfg.DATASET_DIRS.VALIDATION_DIR)
TEST_DIR = Path(cfg.DATASET_DIRS.TEST_DIR)


IMG_SIZE = cfg.TRAIN.IMG_SIZE
BATCH_SIZE = cfg.TRAIN.BATCH_SIZE
LOG_DIR = cfg.OUTPUTS.LOG_DIR
CHECK_POINT_DIR = Path(cfg.OUTPUTS.CHECKPOINT_PATH)
CLASS_NAME = [
    'label0',
    'label1',
    'label2'
]
class_map = {k: v for k, v in enumerate(CLASS_NAME)}

NUM_EPOCHS = cfg.TRAIN.NUM_EPOCHS
LEARNING_RATE = cfg.TRAIN.LEARNING_RATE

NUM_CLASSES = len(CLASS_NAME)


def main() -> None:
    log = get_logger(__name__)
    # to_monitor = 'val_classification_AUC'
    # mode = 'max'
    to_monitor = 'val_bounding_box_iou_metric'
    mode = 'max'

    found_gpu = tf.config.list_physical_devices('GPU')
    if not found_gpu:
        log.error("No GPU found")
        raise Exception("No GPU found")
    
    if not TRAIN_DIR.exists():
        from roboflow import Roboflow
        rf = Roboflow(api_key="AAjLIN3PenSZ29LjbI3d")
        project = rf.workspace("yousef-ghanem-jzj4y").project("brain-tumor-detection-fpf1f")
        version = project.version(2)
        dataset = version.download("tensorflow") 

    # Load images from directory
    train_dl, train_ds, valid_ds = get_train_valid_ds()

    CLS_METRICS, REG_METRICS = get_metrics()

    callbacks = get_callbacks(to_monitor, mode)

    # ### Define Optimizer
    optimizer=tf.keras.optimizers.AdamW(learning_rate=LEARNING_RATE,global_clipnorm=1.0)

    padded_class_ids = train_dl.multi_hot_class_ids
    positive_weights, negative_weights = _loss.compute_class_weights(padded_class_ids)

    mlflow.set_experiment("/brain-tumor-resnet101-detector")
    with mlflow.start_run() :
        mlflow.tensorflow.autolog(log_models=True, 
                                log_datasets=False, 
                                log_input_examples=True,
                                log_model_signatures=True,
                                keras_model_kwargs={"save_format": "keras"},
                                checkpoint_monitor=to_monitor, 
                                checkpoint_mode=mode)
        # Model Building and Compilation
        model = final_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES)
        model.compile(
            optimizer=optimizer,
                # loss_weights={
                #     'classification': 0.5,  # Example: Reduce weight for classification
                #     'bounding_box': 0.5     # Example: Increase weight for regression
                # },
            loss={'classification': _loss.set_binary_crossentropy_weighted_loss(positive_weights, negative_weights),
                'bounding_box': iou_loss.iou_loss},
            metrics={'classification': CLS_METRICS, 
                    'bounding_box': REG_METRICS})

        #  Train and Validate the model
        model.fit(
            train_ds,
            epochs=NUM_EPOCHS,
            validation_data=valid_ds,
            batch_size=BATCH_SIZE,
            callbacks=[callbacks],
        )

        signature = get_mlflow_model_log_schema()
        mlflow.tensorflow.log_model(
        model,
        "my_model",
        signature=signature,
        code_paths=["src/losses"])

        test_ds, y_true_labels, y_true_bboxes = get_test_ds()

        evaluate_model(model=model, 
                        test_ds=test_ds, 
                        y_true_labels=y_true_labels, 
                        y_true_bboxes=y_true_bboxes,
                        output_dir=cfg.OUTPUTS.OUPUT_DIR,
                        class_name=CLASS_NAME) 

def get_train_valid_ds()-> tuple[DataLoader, tf.data.Dataset, tf.data.Dataset]:
    prepare_train_dataset = AnnotationProcessor(annotation_file= str(TRAIN_DIR/'_annotations.csv'))
    _class_map = {v: k for k, v in enumerate(CLASS_NAME)}
    train_images, train_class_ids, train_bboxes  = prepare_train_dataset.process_annotations(image_dir=TRAIN_DIR, class_id_map=_class_map)

    train_dl = DataLoader(train_images, train_class_ids, train_bboxes)
    train_ds = train_dl.load_train_dataset()
    train_ds = Preprocessor(train_ds).preprocess()
    train_ds = train_ds.batch(BATCH_SIZE)\
                    .prefetch(tf.data.AUTOTUNE)

    #Validation datasets setup
    prepare_valid_dataset = AnnotationProcessor(annotation_file= str(VALIDATION_DIR/'_annotations.csv'))
    valid_image_paths, valid_class_ids, valid_bboxes  = prepare_valid_dataset.process_annotations(image_dir=VALIDATION_DIR, class_id_map=_class_map)
    valid_dl = DataLoader(valid_image_paths, valid_class_ids, valid_bboxes).load_val_dataset()
    valid_ds = Preprocessor(valid_dl).preprocess()
    valid_ds = valid_ds.batch(BATCH_SIZE)\
                    .prefetch(tf.data.AUTOTUNE)
                    
    return train_dl,train_ds,valid_ds

def get_metrics() -> tuple[list, list]:
    CLS_METRICS = [
        tf.keras.metrics.AUC(name='AUC', multi_label=True), 
        tf.keras.metrics.F1Score(name='f1_score',average='weighted')]
    REG_METRICS = [
        iou_loss.iou_metric,
        tf.keras.metrics.MeanSquaredError(name='mse')]
        
    return CLS_METRICS,REG_METRICS

def get_callbacks(to_monitor, mode) -> list:
    # to_monitor = 'val_iou_metric'
    # mode = 'max'
    return [
        # Learning Rate Scheduler
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=to_monitor,
            factor=0.2,  # Less aggressive reduction
            patience=5,
            mode=mode,
            min_lr=1e-6,  # Higher minimum learning rate
            verbose=1
        ),
        
        # Model Checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECK_POINT_DIR, "detector_ckpt_{epoch}.keras"),
            save_weights_only=False,
            save_best_only=True,
            monitor=to_monitor,
            mode=mode,
            verbose=1
        ),
        
        # Early Stopping
        tf.keras.callbacks.EarlyStopping(
            monitor=to_monitor,
            patience=15,  # Allow 3 LR reductions before stopping
            mode=mode,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Additional Recommendation
        tf.keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1  # Track gradients/weights
        )]

    # return [tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, 
    #                                             patience=5, 
    #                                             monitor=to_monitor,
    #                                             mode=mode,
    #                                             min_lr=1e-7,
    #                                             verbose=1),
    #     tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(str(CHECK_POINT_DIR), "detector_ckpt_{epoch}.keras") ,
    #                                         save_weights_only=False,
    #                                         save_best_only=True,
    #                                         monitor=to_monitor,
    #                                         mode=mode,
    #                                         verbose=1),                                 
    #     tf.keras.callbacks.EarlyStopping(monitor=to_monitor, 
    #                                     patience=10,
    #                                     mode=mode, 
    #                                     restore_best_weights=True,
    #                                     verbose=1)]
                                        
def get_test_ds()-> tuple[tf.data.Dataset, None | np.ndarray, None | np.ndarray]:
    prepare_test_dataset = AnnotationProcessor(annotation_file= str(TEST_DIR/'_annotations.csv'))
    _class_map = {v: k for k, v in enumerate(CLASS_NAME)}
    test_image_paths, test_class_ids, test_bboxes = prepare_test_dataset.process_annotations(image_dir=TEST_DIR, class_id_map=_class_map)
    test_dl = DataLoader(test_image_paths, test_class_ids, test_bboxes, img_size=IMG_SIZE)
    test_ds = test_dl.load_val_dataset()
    y_true_labels = test_dl.multi_hot_class_ids
    y_true_bboxes = test_dl.padded_bbx
    test_ds = Preprocessor(test_ds).preprocess()
    test_ds = test_ds.batch(BATCH_SIZE)\
                        .prefetch(tf.data.AUTOTUNE)
                    
    return test_ds, y_true_labels,y_true_bboxes                     

def get_mlflow_model_log_schema():

    # 1. Input Schema
    # -----------------
    # Your input is a batch of images with shape (32, 240, 240, 3)
    # We use -1 to indicate that the batch size can vary.
    input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, IMG_SIZE, IMG_SIZE, 3), "image")])

    # 2. Output Schema - Multilabel binary classification head
    # ------------------
    # Your model outputs a list of two arrays. We need to define a schema for each.
    # Array 1: Shape (1, 3)
    output_schema_array1 = TensorSpec(np.dtype(np.float32), (-1, 3), "classification")

    # Array 2: Shape (1, 3, 4) - 3 Bounding boxes per classification 
    output_schema_array2 = TensorSpec(np.dtype(np.float32), (-1, 3, 4), "bounding_box")

    # Create a schema for the list of outputs
    output_schema = Schema([output_schema_array1, output_schema_array2])

    # 3. Model Signature
    # --------------------
    # Combine the input and output schemas into a ModelSignature
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    return signature

def evaluate_model(*,model:tf.keras.Model, 
                   test_ds:tf.data.Dataset, 
                   y_true_labels:np.ndarray, 
                   y_true_bboxes:np.ndarray,
                   output_dir:Path,
                   class_name:list[str]) -> None:
    """Evaluates the model.

    Args:
        model: The model to evaluate.
        test_ds: The test dataset.
        y_true_labels: The true labels.
        y_true_bboxes: The true bounding boxes.
        cfg: The configuration object.
        class_name: The list of class names.
    """
    log = get_logger(__name__)
    log.info("Evaluating model...")
    results = model.evaluate(test_ds, return_dict=True, steps=1)
    mlflow.log_dict(results, 'test_metrics.json')
    predictions = model.predict(test_ds)
    y_prob_pred = predictions[0]
    pred_bbx = predictions[1]

    y_pred = (y_prob_pred>0.5).astype(int)

    report = classification_report(y_true_labels,
                                    y_pred, 
                                    labels=[0,1,2], 
                                    target_names=class_name,
                                    output_dict=True)
    mlflow.log_dict(report, 'classification_report.json')

    auc_fig = plot_auc_curve(output_dir, class_name, y_true_labels, y_prob_pred)
    mlflow.log_figure(auc_fig, 'ROC-Curve.png')

    hist = plot_iou_histogram(output_dir, y_true_bboxes, pred_bbx, is_image_show=False)
    mlflow.log_figure(hist, 'iou_histogram.png')
    log.info("Model evaluated.")


if __name__ == '__main__':
    main()


















