import tensorflow as tf


def iou_loss(y_true, y_pred):  # Assuming y_true and y_pred are (batch_size, 4)
    # y_true = y_true[0]
    # y_pred = tf.reshape(y_pred, ( 3, 4))
    y_true = tf.cast(y_true, dtype=tf.float32) # Cast to float32
    y_pred = tf.cast(y_pred, dtype=tf.float32) # Cast to float32

    x_true = y_true[..., 0]
    y_true_ = y_true[..., 1]
    x_max_true = y_true[..., 2]
    y_max_true = y_true[..., 3]

    x_pred = y_pred[..., 0]
    y_pred_ = y_pred[..., 1]
    x_max_pred = y_pred[..., 2]
    y_max_pred = y_pred[..., 3]

    area_true = (x_max_true - x_true) * (y_max_true - y_true_)
    area_pred = (x_max_pred - x_pred) * (y_max_pred - y_pred_)

    x_intersect = tf.maximum(x_true, x_pred)
    y_intersect = tf.maximum(y_true_, y_pred_)
    x_max_intersect = tf.minimum(x_max_true, x_max_pred)
    y_max_intersect = tf.minimum(y_max_true, y_max_pred)

    area_intersect = tf.maximum(0.0, x_max_intersect - x_intersect) * tf.maximum(0.0, y_max_intersect - y_intersect) # avoid negative values
    iou = area_intersect / (area_true + area_pred - area_intersect + 1e-7)  # Add small epsilon for numerical stability
    return 1.0 - iou  # We want to *minimize* the loss

def iou_metric(y_true, y_pred):  # No negation for metric
    # y_true = y_true[0]
    # y_pred = tf.reshape(y_pred, (3, 4))
    y_true = tf.cast(y_true, dtype=tf.float32) # Cast to float32
    y_pred = tf.cast(y_pred, dtype=tf.float32) # Cast to float32

    x_true = y_true[..., 0]
    y_true_ = y_true[..., 1]
    x_max_true = y_true[..., 2]
    y_max_true = y_true[..., 3]

    x_pred = y_pred[..., 0]
    y_pred_ = y_pred[..., 1]
    x_max_pred = y_pred[..., 2]
    y_max_pred = y_pred[..., 3]

    area_true = (x_max_true - x_true) * (y_max_true - y_true_)
    area_pred = (x_max_pred - x_pred) * (y_max_pred - y_pred_)

    x_intersect = tf.maximum(x_true, x_pred)
    y_intersect = tf.maximum(y_true_, y_pred_)
    x_max_intersect = tf.minimum(x_max_true, x_max_pred)
    y_max_intersect = tf.minimum(y_max_true, y_max_pred)

    area_intersect = tf.maximum(0.0, x_max_intersect - x_intersect) * tf.maximum(0.0, y_max_intersect - y_intersect) # avoid negative values
    iou = area_intersect / (area_true + area_pred - area_intersect + 1e-7)  # Add small epsilon for numerical stability
    return iou  # Return IoU directly for metric