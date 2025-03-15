import unittest

import numpy as np
import tensorflow as tf

from src.losses.iou_loss import iou_loss


class TestIoULoss(unittest.TestCase):
    def setUp(self):
        # Create a dummy DataLoader instance for testing
        pass

    def test_perfect_match(self):
        y_true = tf.constant([[0.1, 0.2, 0.3, 0.4]], dtype=tf.float32)
        y_pred = tf.constant([[0.1, 0.2, 0.3, 0.4]], dtype=tf.float32)
        loss = iou_loss(y_true, y_pred)
        self.assertAlmostEqual(loss.numpy().item(), 0.0, places=5)

    def test_no_overlap(self):
        y_true = tf.constant([[[0.1, 0.2, 0.3, 0.4]]], dtype=tf.float32)
        y_pred = tf.constant([[[0.5, 0.6, 0.7, 0.8]]], dtype=tf.float32)
        loss = iou_loss(y_true, y_pred)
        self.assertAlmostEqual(loss.numpy(), 1.0)

    def test_partial_overlap(self):
        y_true = tf.constant([[[0.1, 0.2, 0.5, 0.6]]], dtype=tf.float32)
        y_pred = tf.constant([[[0.3, 0.4, 0.7, 0.8]]], dtype=tf.float32)
        loss = iou_loss(y_true, y_pred)
        self.assertGreater(loss.numpy(), 0.0)
        self.assertLess(loss.numpy(), 1.0)

    def test_one_zero_box(self):
        y_true = tf.constant([[[0.0, 0.0, 0.0, 0.0]]], dtype=tf.float32)
        y_pred = tf.constant([[[0.1, 0.2, 0.3, 0.4]]], dtype=tf.float32)
        loss = iou_loss(y_true, y_pred)
        # When y_true is a zero box, iou_loss returns an empty tensor, so we expect 0.0
        if loss.shape.rank == 0:
            self.assertAlmostEqual(loss.numpy().item(), 0.0, places=5)
        else:
            self.assertEqual(loss.shape[0], 0)


    def test_both_zero_boxes(self):
        y_true = tf.constant([[[0.0, 0.0, 0.0, 0.0]]], dtype=tf.float32)
        y_pred = tf.constant([[[0.0, 0.0, 0.0, 0.0]]], dtype=tf.float32)
        loss = iou_loss(y_true, y_pred)
        # When y_true is a zero box, iou_loss returns an empty tensor, so we expect 0.0
        if loss.shape.rank == 0:
            self.assertAlmostEqual(loss.numpy().item(), 0.0, places=5)
        else:
            self.assertEqual(loss.shape[0], 0)

    def test_multiple_boxes_in_batch(self):
        y_true = tf.constant([
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
            [[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]]
        ], dtype=tf.float32)
        y_pred = tf.constant([
            [[0.1, 0.2, 0.3, 0.4], [0.6, 0.7, 0.8, 0.9]],
            [[0.2, 0.3, 0.4, 0.5], [0.5, 0.6, 0.7, 0.8]]
        ], dtype=tf.float32)
        loss = iou_loss(y_true, y_pred)
        # loss is a tensor of shape (2,) in this case, so we need to take the mean
        self.assertGreaterEqual(tf.reduce_mean(loss).numpy().item(), 0.0)
        self.assertLessEqual(tf.reduce_mean(loss).numpy().item(), 1.0)

    def test_multiple_boxes_in_batch_with_zero_box(self):
        y_true = tf.constant([
            [[0.1, 0.2, 0.3, 0.4], [0.0, 0.0, 0.0, 0.0]],
            [[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]]
        ], dtype=tf.float32)
        y_pred = tf.constant([
            [[0.1, 0.2, 0.3, 0.4], [0.6, 0.7, 0.8, 0.9]],
            [[0.2, 0.3, 0.4, 0.5], [0.5, 0.6, 0.7, 0.8]]
        ], dtype=tf.float32)
        loss = iou_loss(y_true, y_pred)
        # loss is a tensor of shape (2,) in this case, so we need to take the mean
        self.assertGreaterEqual(tf.reduce_mean(loss).numpy().item(), 0.0)
        self.assertLessEqual(tf.reduce_mean(loss).numpy().item(), 1.0)


if __name__ == '__main__':
    unittest.main()