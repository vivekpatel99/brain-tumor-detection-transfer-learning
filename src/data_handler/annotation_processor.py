
# https://github.com/aleju/imgaug/issues/859
# import imgaug.augmenters as iaa
import numpy as np
import pandas as pd


class AnnotationProcessor:
    def __init__(self, annotation_file):
        self.annotation_file = annotation_file
        self.df = pd.read_csv(str(self.annotation_file))  # Assumes CSV format
        self.images = []
        self.class_ids = []
        self.bboxes = []

    def process_annotations(self, image_dir, class_id_map, plot=False):
        """
        Processes annotations and draws bounding boxes on images.

        Args:
            image_dir: The directory containing the images.

        Returns:
            A list of tuples, where each tuple contains:
                - The image with bounding boxes drawn.
                - A list of normalized bounding box coordinates for each object in the image.
        """
        uni_list = list(self.df['filename'].unique())
        for image_name in uni_list:#[:100]:  # Iterate over unique images
            image_path = str(image_dir/ image_name)  # Construct full image path
            try:
                image_annotations = self.df[self.df['filename'] == image_name]  # Get annotations for this image
                labels = []
                cords = []
                for _, row in image_annotations.iterrows():
                    x_min = int(row['xmin'])
                    y_min = int(row['ymin'])
                    x_max = int(row['xmax'])
                    y_max = int(row['ymax'])
                    label = row['class']  

                    if not plot:
                        img_width = int(row['width'])
                        img_height = int(row['height'])
                        
                        #  Original Box → Resize Image → Adjust Box → Normalize
                        # Normalize bounding box coordinates
                        x_min = x_min / img_width
                        y_min = y_min / img_height
                        x_max = x_max / img_width
                        y_max = y_max / img_height

                    labels.append(class_id_map[label])
                    cords.append([x_min, y_min, x_max, y_max])

                self.images.append(image_path)
                self.class_ids.append(labels)
                self.bboxes.append(np.array(cords))

            except Exception as e:
                print(f"Error processing image {image_name}: {e}")

        return self.images, self.class_ids, self.bboxes