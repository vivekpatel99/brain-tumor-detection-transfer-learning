import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

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