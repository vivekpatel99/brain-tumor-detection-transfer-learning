
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import tqdm
from tqdm.notebook import tqdm

# class CustomDatasetLoader(tf.data.Dataset):
#     def __new__(cls, image_dir: Path, label_dir: Path, num_classes:int, dst_img_size:tuple[int, int]=(224,224), batch_size:int=32) -> tf.data.Dataset:
#         """
#         Args:
#             image_dir (str): Path to the directory containing the images.
#             label_dir (str): Path to the directory containing the labels.
#         """
#         images = [str(f) for f in image_dir.iterdir()]
#         labels = [str(f) for f in label_dir.iterdir()]
#         dataset = tf.data.Dataset.from_tensor_slices((images, labels))
#         dataset = dataset.map(lambda image_dir, label_dir: cls.load_data(image_dir, label_dir,num_classes, dst_img_size))
#         dataset = dataset.padded_batch(batch_size)
#         return dataset
    

#     @staticmethod
#     def load_data(image_path:str, label_path: Path, num_classes:int, dst_img_size:tuple[int, int]=(224,224)):
#         image = tf.io.read_file(image_path)
#         image = tf.image.decode_jpeg(image, channels=3)
#         image = tf.image.resize(image, dst_img_size)

#         labels = tf.io.read_file(label_path)
#         labels = tf.strings.split(labels, sep='\n')
#         labels = tf.strings.to_number(tf.strings.split(labels, sep=' '), out_type=tf.float32)

#         labels = tf.reshape(labels, [-1, 5]) # Assuming each line as 5 values
#         class_ids =labels[:, 0] # Class ID # Class ID
#         # one_hot_labels = tf.one_hot(tf.cast(class_ids, tf.int8), depth=num_classes)
#         bboxs = labels[:, 1:] # Bounding box coordinates
#         return image, (class_ids, bboxs) # image, label





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


    def get_dataset(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Loads and parses YOLOv8 labels.

        Args:
            None

        Returns:
            tuple[list[str], list[int], list[np.ndarray]]: A tuple containing:
                - A list of image paths.
                - A list of class ids.
                - A list of bounding boxes, where each bounding box is an array of 8 floats.
        """
        images = []
        class_ids = []
        bboxes = []
        for file_name in tqdm(list(self.image_dir.iterdir())[:100]):
            if file_name.suffix.endswith((".jpg", ".png")):
                image =  cv2.resize(cv2.imread(str(file_name)), self.dst_img_size)
                # preprocessed_img = self.prepare_images(image_path=imaga_file_path)
                label_file_path = self.label_dir / f'{file_name.stem}.txt'

                if not label_file_path.exists():
                    print(f"Label file not found for image: {label_file_path}")
                    continue

                with label_file_path.open('r') as f:
                    lines = f.readlines()

                if not lines:
                    continue

                # 1 0.4395833333333333 0.35625 0.12916666666666668 0.1125
                for line in lines:
                    try:
                        values = np.array([value for value in line.split()], dtype=np.float32)
                        class_id = values[0]
                        coords = values[1:5]  # Coords are already normalized YOLO format

                        bboxes.append(coords)
                        class_ids.append(class_id)
                        images.append(image)

                    except Exception as e:
                        print(f"[ERROR] - {e} in file {label_file_path} on line: {line}")
                        continue
                    
        return np.array(images), (np.array(class_ids,dtype=np.int8), np.array(bboxes, dtype=np.float32))
