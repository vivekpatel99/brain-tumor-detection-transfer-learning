import keras
import tensorflow as tf


@keras.saving.register_keras_serializable()
def iou_metric(y_true, y_pred) -> float:  # No negation for metric
    epsilon = 1e-7  # Small value for numerical stability
    y_true = tf.cast(y_true, dtype=tf.float32) # Cast to float32
    y_pred = tf.cast(y_pred, dtype=tf.float32) # Cast to float32
    
    # Create a mask for valid bounding boxes
    mask = tf.reduce_any(tf.not_equal(y_true, 0), axis=-1)
    mask = tf.cast(mask, dtype=tf.float32)

    x_min_true = y_true[..., 0]
    y_min_true = y_true[..., 1]
    x_max_true = y_true[..., 2]
    y_max_true = y_true[..., 3]

    x_min_pred = y_pred[..., 0]
    y_min_pred = y_pred[..., 1]
    x_max_pred = y_pred[..., 2]
    y_max_pred = y_pred[..., 3]

    area_true = (x_max_true - x_min_true) * (y_max_true - y_min_true)
    area_pred = (x_max_pred - x_min_pred) * (y_max_pred - y_min_pred)

    x_intersect = tf.maximum(x_min_true, x_min_pred)
    y_intersect = tf.maximum(y_min_true, y_min_pred)
    x_max_intersect = tf.minimum(x_max_true, x_max_pred)
    y_max_intersect = tf.minimum(y_max_true, y_max_pred)

    area_intersect = tf.maximum(0.0, x_max_intersect - x_intersect) * tf.maximum(0.0, y_max_intersect - y_intersect) # avoid negative values
    iou = area_intersect / (area_true + area_pred - area_intersect + epsilon)  # Add small epsilon for numerical stability
    iou = tf.boolean_mask(iou, mask, axis=0)
    return iou  # Return IoU directly for metric


@keras.saving.register_keras_serializable()
def iou_loss(y_true, y_pred) -> float:  # Assuming y_true and y_pred are (batch_size, 4)
    """x_min, y_min, x_max, y_max
    """
    return 1- iou_metric(y_true, y_pred) 



def giou_metric(b1: tf.Tensor, b2: tf.Tensor, mode: str = "giou") -> tf.Tensor:
    """
    #https://github.com/tensorflow/addons/blob/v0.20.0/tensorflow_addons/losses/giou_loss.py#L26-L61
    Args:
        b1: bounding box. The coordinates of the each bounding box in boxes are
            encoded as [y_min, x_min, y_max, x_max].
        b2: the other bounding box. The coordinates of the each bounding box
            in boxes are encoded as [y_min, x_min, y_max, x_max].
        mode: one of ['giou', 'iou'], decided to calculate GIoU or IoU loss.

    Returns:
        GIoU loss float `Tensor`.
    """
    zero = tf.convert_to_tensor(0.0, b1.dtype)
    epsilon = 1e-7  # Small value for numerical stability
    # x_min, y_min, x_max, y_max
    b1_xmin, b1_ymin,  b1_xmax, b1_ymax  = tf.unstack(b1, 4, axis=-1)
    # b1_ymin, b1_xmin, b1_ymax, b1_xmax = tf.unstack(b1, 4, axis=-1)

    b2_xmin, b2_ymin,  b2_xmax, b2_ymax = tf.unstack(b2, 4, axis=-1)
    # b2_ymin, b2_xmin, b2_ymax, b2_xmax = tf.unstack(b2, 4, axis=-1)
    b1_width = tf.maximum(zero, b1_xmax - b1_xmin)
    b1_height = tf.maximum(zero, b1_ymax - b1_ymin)
    b2_width = tf.maximum(zero, b2_xmax - b2_xmin)
    b2_height = tf.maximum(zero, b2_ymax - b2_ymin)
    b1_area = b1_width * b1_height
    b2_area = b2_width * b2_height

    intersect_ymin = tf.maximum(b1_ymin, b2_ymin)
    intersect_xmin = tf.maximum(b1_xmin, b2_xmin)
    intersect_ymax = tf.minimum(b1_ymax, b2_ymax)
    intersect_xmax = tf.minimum(b1_xmax, b2_xmax)
    intersect_width = tf.maximum(zero, intersect_xmax - intersect_xmin)
    intersect_height = tf.maximum(zero, intersect_ymax - intersect_ymin)
    intersect_area = intersect_width * intersect_height

    union_area = b1_area + b2_area - intersect_area
    iou = tf.math.divide_no_nan(intersect_area, union_area+ epsilon)
    if mode == "iou":
        return iou

    enclose_ymin = tf.minimum(b1_ymin, b2_ymin)
    enclose_xmin = tf.minimum(b1_xmin, b2_xmin)
    enclose_ymax = tf.maximum(b1_ymax, b2_ymax)
    enclose_xmax = tf.maximum(b1_xmax, b2_xmax)
    enclose_width = tf.maximum(zero, enclose_xmax - enclose_xmin)
    enclose_height = tf.maximum(zero, enclose_ymax - enclose_ymin)
    enclose_area = enclose_width * enclose_height
    giou = iou - tf.math.divide_no_nan((enclose_area - union_area), enclose_area+ epsilon)
    # giou = tf.squeeze(giou)
    return giou

def giou_loss(b1: tf.Tensor, b2: tf.Tensor, mode: str = "giou") -> tf.Tensor:
    """
    #https://github.com/tensorflow/addons/blob/v0.20.0/tensorflow_addons/losses/giou_loss.py#L26-L61
    Args:
        b1: bounding box. The coordinates of the each bounding box in boxes are
            encoded as [y_min, x_min, y_max, x_max].
        b2: the other bounding box. The coordinates of the each bounding box
            in boxes are encoded as [y_min, x_min, y_max, x_max].
        mode: one of ['giou', 'iou'], decided to calculate GIoU or IoU loss.

    Returns:
        GIoU loss float `Tensor`.
    """
    return 1 - giou_metric(b1=b1, b2=b2, mode=mode)

