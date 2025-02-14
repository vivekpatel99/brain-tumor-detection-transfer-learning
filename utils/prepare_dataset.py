from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import tqdm
from tqdm.notebook import tqdm


class PrepareDataset:
    def __init__(self, image_dir: Path, label_dir: Path, dst_img_size:tuple[int, int]=(224,224)) -> None:
        """
        Args:
            image_dir (str): Path to the directory containing the images.
            label_dir (str): Path to the directory containing the labels.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.dst_img_size= dst_img_size
        self.images = []
        self.class_ids = []
        self.bboxes = []

    def seperate_class_with_datasets(self, class_id):
        idx = np.where(self.class_ids == class_id)[0]
        return (np.array(self.images)[idx], self.class_ids[idx], self.bboxes[idx])
    
    def rebalance_by_down_sampling_datasets(self):

        unique_class_ids, value_counts  = np.unique(self.class_ids, return_counts=True)
        print(f"[INFO] Unique class ids: {unique_class_ids}, value counts: {value_counts}")

        down_sampling_size = value_counts.min()
        _images = []
        _class_ids = []
        _bboxes = []
        for id in unique_class_ids:
            images, class_ids, bboxes = self.seperate_class_with_datasets(id)
            _images.extend(images[:down_sampling_size])
            _class_ids.extend(class_ids[:down_sampling_size])
            _bboxes.extend(bboxes[:down_sampling_size])
        
        return _images, _class_ids, _bboxes

    def get_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Loads and parses YOLOv8 labels.

        Args:
            image_dir: Path to the directory containing images.
            label_dir: Path to the directory containing labels.
            dst_img_size: Tuple (width, height) specifying the desired image size.

        Returns:

        images = []
        class_ids = []
        bboxes = []

        for file_name in tqdm(list(self.image_dir.iterdir())[:100]):  # Removed [:100]
            if file_name.suffix.lower() in (".jpg", ".png", ".jpeg"): # Added .jpeg and lower() for robustness
                image_path = file_name
                label_file_path = self.label_dir / f'{file_name.stem}.txt'

                if not label_file_path.exists():
                    print(f"Label file not found for image: {image_path}")
                    continue

                with label_file_path.open('r') as f:
                    lines = f.readlines()

                if not lines:
                    print(f"Label file is empty: {label_file_path}")
                    continue

                for line in lines:
                    try:
                        values = np.array([float(value) for value in line.split()]) # Explicit float conversion
                        class_id = int(values[0])  # Explicit int conversion for class ID
                        coords = values[1:5].astype(np.float32)  # Ensure float32 for coords
                        
                        images.append(str(image_path))
                        bboxes.append(coords)
                        class_ids.append(class_id)

                    except ValueError as e:  # Catch specific ValueError
                        print(f"[ERROR] - {e} in file {label_file_path} on line: {line}")
                        continue
                    except IndexError as e: # Catch potential IndexError
                        print(f"[ERROR] - {e} in file {label_file_path} on line: {line}")
                        continue

        return images, (np.array(class_ids, dtype=np.int8), np.array(bboxes, dtype=np.float32))
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - Image data (NumPy array, shape (num_images, height, width, channels)).
                - Class IDs (NumPy array, dtype=np.int32).
                - Bounding boxes (NumPy array, shape (num_images, max_objects, 4), dtype=np.float32).
        """

        for file_name in tqdm(list(self.image_dir.iterdir())):  # Removed [:100]
            if file_name.suffix.lower() in (".jpg", ".png", ".jpeg"): # Added .jpeg and lower() for robustness
                image_path = file_name
                label_file_path = self.label_dir / f'{file_name.stem}.txt'

                if not label_file_path.exists():
                    print(f"Label file not found for image: {image_path}")
                    continue

                with label_file_path.open('r') as f:
                    lines = f.readlines()

                if not lines:
                    print(f"Label file is empty: {label_file_path}")
                    continue

                for line in lines:
                    try:
                        values = np.array([float(value) for value in line.split()]) # Explicit float conversion
                        class_id = int(values[0])  # Explicit int conversion for class ID
                        coords = values[1:5].astype(np.float32)  # Ensure float32 for coords
                        
                        self.images.append(str(image_path))
                        self.bboxes.append(coords)
                        self.class_ids.append(class_id)

                    except ValueError as e:  # Catch specific ValueError
                        print(f"[ERROR] - {e} in file {label_file_path} on line: {line}")
                        continue
                    except IndexError as e: # Catch potential IndexError
                        print(f"[ERROR] - {e} in file {label_file_path} on line: {line}")
                        continue

        self.images = np.array(self.images)
        self.bboxes = np.array(self.bboxes, dtype=np.float32)
        self.class_ids = np.array(self.class_ids, dtype=np.int8)

        return self.images, self.class_ids, self.bboxes