
import numpy as np


def convert_coordinates_for_plot(image, bbox):
  x_center, y_center, width, height = bbox
  img_height, img_width = image.shape[:2]
  xmin = int(max(0, (x_center - width / 2) * img_width))  # Clip to 0
  ymin = int(max(0, (y_center - height / 2) * img_height)) # Clip to 0
  xmax = int(min(img_width, (x_center + width / 2) * img_width)) # Clip to image width
  ymax = int(min(img_height, (y_center + height / 2) * img_height))# Clip to image height


  return np.array([xmin, ymin, xmax, ymax], dtype=np.int32)#.reshape(1, 4)
