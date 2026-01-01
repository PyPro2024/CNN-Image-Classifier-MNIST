# CNN Image Classifier (MNIST)

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=flat&logo=keras)

## Project Overview
This project implements a **Convolutional Neural Network (CNN)** to classify handwritten digits using the **MNIST dataset**. The goal is to demonstrate the effectiveness of deep learning architectures in computer vision tasks compared to traditional dense neural networks.

The model is built using **TensorFlow** and **Keras**, featuring a robust architecture of convolutional and pooling layers to extract spatial hierarchies of features from input images.

## Model Architecture
The network follows a standard sequential CNN design:
1.  **Input Layer:** Accepts 28x28 grayscale images.
2.  **Convolutional Layers (Conv2D):** Extract features like edges and textures using learnable filters (kernels).
3.  **Pooling Layers (MaxPooling2D):** Downsample the feature maps to reduce computational cost and control overfitting.
4.  **Flatten Layer:** Converts the 2D matrix data into a 1D vector.
5.  **Dense Layers:** Fully connected layers to perform the final classification into 10 digit classes (0-9).

## Dataset
* **Source:** [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) (Loaded via Keras)
* **Training Set:** 60,000 images
* **Test Set:** 10,000 images
* **Classes:** 10 (Digits 0-9)

## Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Data Manipulation:** NumPy
* **Visualization:** Matplotlib
* **Environment:** Jupyter Notebook / Google Colab

## Key Results
* **Training Accuracy:** ~99% (Achieved over 10-15 epochs)
* **Test Accuracy:** High generalization capability on unseen data.
* **Evaluation:** The model successfully predicts handwritten digits with high confidence.

*(Note: Detailed accuracy plots and loss curves are available in the notebook.)*

## How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/PyPro2024/CNN-Image-Classifier-MNIST.git]
    ```
2.  **Install dependencies:**
    ```bash
    pip install tensorflow numpy matplotlib
    ```
3.  **Run the Notebook:**
    Open `CNN.ipynb` in Jupyter Notebook or Google Colab to execute the cells and view the training process.

---
*If you find this project helpful, feel free to ⭐ the repo!*
