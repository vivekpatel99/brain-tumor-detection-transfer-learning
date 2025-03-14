import random

import cv2
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn import metrics

from utils.bounding_box_funcs import convert_coordinates_for_plot


def plot_random_images_bbox(*, image_paths:np.ndarray, class_ids:np.ndarray, bboxes:np.ndarray, class_map:dict, NUM_IMAGES_DISPLAY:int=9) -> None:
  fig = plt.figure(figsize=(8, 8))
  random_samples = random.sample(range(len(image_paths)), NUM_IMAGES_DISPLAY)
  print(f"Random samples: {random_samples}")
  class_map_invert = {v: k for k, v in class_map.items()}

  for i, idx in enumerate(random_samples):
    ax = fig.add_subplot(3, 3, i+1)
    image = image_paths[idx]
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Create title from class IDs
    title_labels = [class_map_invert[int(cls_id)] for cls_id in class_ids[idx]]
    title = ", ".join(title_labels)
    ax.set_title(title)
    ax.imshow(image) #display image before bounding box

    # Draw bounding boxes with different colors
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0,255,255), (255,0,255)] # Example colors
    for j, (xmin, ymin, xmax, ymax) in enumerate(bboxes[idx]):
        color = colors[j % len(colors)] # Cycle through colors
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 1)
    ax.imshow(image) #display image with bounding box.

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

    plt.tight_layout() #prevents overlapping titles/labels
    plt.show()

def plot_auc_curve(cfg, class_name_list, y_true, y_prob_pred):
    auc_roc_values = []
    fig, axs = plt.subplots(1)
    for i in range(len(class_name_list)):
        try:
            roc_score_per_label = metrics.roc_auc_score(y_true=y_true[:,i], y_score=y_prob_pred[:,i])
            auc_roc_values.append(roc_score_per_label)
            fpr, tpr, _ = metrics.roc_curve(y_true=y_true[:,i],  y_score=y_prob_pred[:,i])
        
            axs.plot([0,1], [0,1], 'k--')
            axs.plot(fpr, tpr, 
                label=f'{class_name_list[i]} - AUC = {round(roc_score_per_label, 3)}')

            axs.set_xlabel('False Positive Rate')
            axs.set_ylabel('True Positive Rate')
            axs.legend(loc='lower right')
        except:
            print(
            f"Error in generating ROC curve for {class_name_list[i]}. "
            f"Dataset lacks enough examples."
        )
    plt.savefig(f"{cfg.OUTPUTS.OUPUT_DIR}/ROC-Curve.png")
    mlflow.log_figure(fig, 'ROC-Curve.png')
