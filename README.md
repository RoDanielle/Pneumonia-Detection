# Pneumonia-Detection

## Introduction

This project uses a deep convolutional neural network (CNN) to detect pneumonia in chest X-ray images.

### What is a Convolutional Neural Network?

A deep convolutional neural network (CNN) is a type of artificial neural network specifically designed for processing structured grid data, such as images. CNNs use convolutional layers to automatically and adaptively learn spatial hierarchies of features from input images. This allows the network to recognize patterns, edges, textures, and higher-level features, making them highly effective for image classification and object detection tasks.

### Dataset Distribution

- **Total Images**: 5863 JPEG Chest X-ray images.

- **Categories**: 
  - Bacterial Pneumonia
  - Viral Pneumonia
  - Normal

**Dataset Source**: The dataset is available on Kaggle. Get it [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

### How to Use

In the project's files, you can find:

1. **Google Colab Notebooks** (inside the `notebooks` folder):
   - **Pneumonia_Detection_Testing.ipynb**: Provides an image as input and gets a prediction output.
   - **Pneumonia_Detection_Training.ipynb**: Contains details about the training process.

2. **Python Scripts** (inside the `scripts` folder):
   - **pneumonia_detection_testing.py**: Python script version of the testing notebook.
   - **pneumonia_detection_training.py**: Python script version of the training notebook.

Both notebooks and scripts include a step-by-step guide on how to use them.

### Models

This project includes several models for pneumonia detection and classification:

1. Binary Classification Model: Classifies images as either Normal or Pneumonia.

2. Multi-Class Classification Model: Classifies images into one of three categories: Bacterial Pneumonia, Viral Pneumonia, or Normal.

3. KNN-based Classification:

    - Binary Model: Classifies new images based on the binary classification model using KNN.
    - Multi-Class Model: Classifies new images based on the multi-class classification model using KNN.

4. Anomaly Detection Model: Uses an autoencoder to detect anomalies in the images.

### Trained Models

The trained models used for this project are included as a link to a Google Drive folder in the test notebook. You can use these models directly without retraining.

### Results and preformance examples
 
 ![Confusion Matrix - Test Set A1 - binary class classification](Example_images/A1_testset_confusion_matrix.jpg)

 ![Confusion Matrix - Test Set A2 - multi class classification](Example_images/A2_testset_confusion_matrix.jpg)

 ![Binary t-SNE Visualization](Example_images/B_Binary_t-SNE.jpg)

 ![Multi-Class t-SNE Visualization](Example_images/B_Multi_Class_t-SNE.jpg)

 ![Original and Reconstructed images from anomaly detection](Example_images/D_Reconstructed_Detection.jpg)