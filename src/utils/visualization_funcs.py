import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

try:
    from src.losses.iou_loss import iou_metric
except ImportError:
    from losses.iou_loss import iou_metric



def plot_random_images_bbox(*, image_paths:np.ndarray, class_ids:np.ndarray, bboxes:np.ndarray, class_map:dict, NUM_IMAGES_DISPLAY:int=9) -> None:
    fig = plt.figure(figsize=(8, 8))
    random_samples = random.sample(range(len(image_paths)), NUM_IMAGES_DISPLAY)
    print(f"Random samples: {random_samples}")
    class_map_invert = {v: k for k, v in class_map.items()}

    # Define colors for each label
    label_colors = {
        "label0": (255, 0, 0),  # Red
        "label1": (0, 255, 0),  # Green
        "label2": (0, 0, 255),  # Blue
    }
  
    for i, idx in enumerate(random_samples):
        ax = fig.add_subplot(3, 3, i+1)
        image = image_paths[idx]
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Create title from class IDs
        # title_labels = [class_map_invert[int(cls_id)] for cls_id in class_ids[idx]]

        ax.imshow(image) #display image before bounding box

        # Draw bounding boxes with different colors
        title_labels = []
        for (xmin, ymin, xmax, ymax), cls_id in zip(bboxes[idx], class_ids[idx]):
            label = class_map_invert[int(cls_id)]
            color = label_colors.get(label, (0, 0, 0))  # Default to black if label not found
            title_labels.append(label)
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 1)

        ax.imshow(image) #display image with bounding box.
        title = ", ".join(title_labels)
        ax.set_title(title)
    plt.tight_layout() #prevents overlapping subplots
    plt.show()




def visualize_training_results(history):
    """
    Visualizes training and validation loss, and training and validation accuracy.

    Args:
        history: A dictionary or object containing training history data.
                 For example, a Keras History object or a dictionary with keys:
                 'loss', 'val_loss', 'accuracy', 'val_accuracy'.
    """

    if isinstance(history, dict):
        # Assumes history is a dictionary
        loss = history.get('loss')
        val_loss = history.get('val_loss')
        accuracy = history.get('accuracy')
        val_accuracy = history.get('val_accuracy')
    else:
        # Assumes history is a Keras History object or similar
        loss = history.history.get('loss')
        val_loss = history.history.get('val_loss')
        accuracy = history.history.get('accuracy')
        val_accuracy = history.history.get('val_accuracy')

    if loss and val_loss:
        epochs = range(1, len(loss) + 1)

        plt.figure(figsize=(12, 5))

        # Plot training & validation loss values
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

    if accuracy and val_accuracy:
        if not (loss and val_loss):
          plt.figure(figsize=(12, 5))
        else:
          plt.subplot(1, 2, 2)
        # Plot training & validation accuracy values
        plt.plot(epochs, accuracy, 'r', label='Training accuracy')
        plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
    plt.grid()
    plt.tight_layout() #prevents overlapping titles/labels
    plt.show()

def plot_auc_curve(output_dir, class_name_list, y_true, y_prob_pred):
    """Plots the ROC curve for each class and calculates the AUC."""
    n_classes = len(class_name_list)

    # Binarize the true labels (important for multi-label)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot ROC curve for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob_pred[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'ROC - {class_name_list[i]} (area = {roc_auc:0.2f})')

    # Plot the diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=2)

    # Set plot limits and labels
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")

    # Save the figure
    fig.savefig(output_dir / 'ROC-Curve.png')
    plt.close(fig)
    return fig
    


def plot_iou_histogram(output_dir, y_true_bbox, y_pred_bbox, class_ids):
    """
    Plots a histogram of Intersection over Union (IoU) scores.

    Args:
        y_true_bbox: Ground truth bounding boxes (list of lists or numpy array).
        y_pred_bbox: Predicted bounding boxes (list of lists or numpy array).
        class_ids: list of class ids.
    """
    fig, axs = plt.subplots(1)

    iou_scores = iou_metric(y_true_bbox, y_pred_bbox)

    # fig.figure(figsize=(10, 6))
    axs.hist(iou_scores, bins=20, range=(0, 1), edgecolor='black')
    axs.set_title('IoU Score Distribution')
    axs.set_xlabel('IoU Score')
    axs.set_ylabel('Frequency')
    axs.grid(True)
    plt.show()
    plt.savefig(f"{output_dir}/iou_histogram.png")
    return fig
# def plot_iou_histogram(y_true_bbox, y_pred_bbox, class_ids):
#     """
#     Plots a histogram of Intersection over Union (IoU) scores.

#     Args:
#         y_true_bbox: Ground truth bounding boxes (list of lists or numpy array).
#         y_pred_bbox: Predicted bounding boxes (list of lists or numpy array).
#         class_ids: list of class ids.
#     """
#     fig, axs = plt.subplots(1)

#     iou_scores = iou_metric(y_true_bbox, y_pred_bbox)

#     # fig.figure(figsize=(10, 6))
#     axs.hist(iou_scores, bins=20, range=(0, 1), edgecolor='black')
#     axs.set_title('IoU Score Distribution')
#     axs.set_xlabel('IoU Score')
#     axs.set_ylabel('Frequency')
#     axs.grid(True)
#     plt.show()
#     plt.savefig(f"{OUTPUT_DIR}/iou_histogram.png")
#     return fig