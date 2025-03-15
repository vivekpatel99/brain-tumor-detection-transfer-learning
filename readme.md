# 🧠 Brain Tumor mulit-label binary classfication with detection using ResNet101

![alt text](reaadme-assets/title-img.png)
## 📝 Project Overview



## 🚀 Key Features



## 🛠️ Technologies Used

- Python
- Tensorflow (with ResNet101)
- Jupyter Notebook

## 📊 Model Training
Exploratory data analisis is done in `01_exploratory_data_analyis_EDA.ipynb` [notebook](notebooks/01_exploratory_data_analyis_EDA.ipynb)
The model training process is detailed in `02_model_training.ipynb` [notebook](lnotebooks/04_model_building_training.ipynb). This notebook covers:

- Data preprocessing
- Model architecture
- Training pipeline
- Evaluation metrics


## 🔧 Setup and Installation

1. Clone the Repo
2. Install vs code with [docker](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker) and [devcontainer](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
3. press `Ctrl+Shift+P` and select `Dev Containers: Rebuild and Reopen in Container`
4. setup .env file with kaggle keys to download dataset directly (move it into datasets dir)
5. open conf/config.yaml for configuring parameters and path

### My Hardware Info

To run this project smoothly, consider the following hardware:

- **CPU**: AMD Ryzen 5900X
- **GPU**: NVIDIA GeForce RTX 3080 (with 10GB VRAM)
- **RAM**: 32 GB DDR4


## 🚀 Detection Results: A Visual Showcase

Dive into the performance of our object detection model with these compelling visualizations. Each image demonstrates the model's ability to locate and classify objects.
The minimum loss achieved on my hardware was approximately 0.0264, using the following hyperparameters:
```Python
batch_size = 32
EPOCHS = 50
learning_rate = 1e-4

```

**Key Visual Elements:**

* **<span style="color:green;">Green Bounding Boxes:</span>** Represent the model's predicted object locations.
* **<span style="color:red;">Red Bounding Boxes:</span>** Indicate the actual ground truth object locations.

* **Title Metrics:** Each image is labeled with:
    * **Score:** The model's confidence in its prediction (higher is better).
    * **IoU (Intersection over Union):** A measure of the overlap between predicted and ground truth boxes (closer to 1.0 is better).

**Visual Results:**

![Object Detection Performance]()

**Interactive Insights:**

To better understand the nuances of the model's performance, consider the following:

* **High Score, High IoU:** These images showcase the model's precision, indicating accurate object localization and high confidence.
* **High Score, Lower IoU:** These cases might reveal instances where the model confidently detects an object but with slight localization errors.
* **<span style="color:green;">Green Bounding Boxes:</span>** Represent the model's predicted object locations.
* **<span style="color:red;">Red Bounding Boxes:</span>** Indicate the actual ground truth object locations.

* **Title Metrics:** Each image is labeled with:
    * **Score:** The model's confidence in its prediction (higher is better).
    * **IoU (Intersection over Union):** A measure of the overlap between predicted and ground truth boxes (closer to 1.0 is better).nsider:

* Try to minimize loss further down to `>0.001` (current loss is `0.0264`) with setting up proper data processing pipeline (`tf.data.Dataset`) and Hyper parameter tuning.
* Analyzing the model's failure cases to identify potential areas for improvement.
