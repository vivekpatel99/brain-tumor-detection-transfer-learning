import keras
import tensorflow as tf


@keras.saving.register_keras_serializable()
def iou_loss(y_true, y_pred) -> float:  # Assuming y_true and y_pred are (batch_size, 4)
    """x_min, y_min, x_max, y_max
    """
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
    iou = area_intersect / (area_true + area_pred - area_intersect + 1e-7)  # Add small epsilon for numerical stability
    loss = tf.boolean_mask(iou, mask)
    return 1- loss 


@keras.saving.register_keras_serializable()
def iou_metric(y_true, y_pred) -> float:  # No negation for metric
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
    iou = area_intersect / (area_true + area_pred - area_intersect + 1e-7)  # Add small epsilon for numerical stability
    iou = tf.boolean_mask(iou, mask, axis=0)
    return iou  # Return IoU directly for metric




    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Create a mask for valid bounding boxes
    mask = tf.reduce_any(tf.not_equal(y_true, 0), axis=-1)
    mask = tf.cast(mask, dtype=tf.float32)
    
    # Calculate IoU
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
    union_area = (area_true + area_pred - area_intersect + 1e-7) 
    iou = area_intersect / union_area # Add small epsilon for numerical stability

    # Calculate GIoU
    enclose_mins = tf.minimum(y_true[..., :2], y_pred[..., :2])
    enclose_maxes = tf.maximum(y_true[..., 2:], y_pred[..., 2:])
    enclose_wh = tf.maximum(enclose_maxes - enclose_mins, 0.)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    
    giou = iou - ((enclose_area - union_area) / (enclose_area + 1e-7))
    giou = tf.clip_by_value(giou, -1.0, 1.0)

    giou_loss = tf.boolean_mask(giou, mask, axis=0)
    return 1-tf.math.reduce_mean (giou_loss) 