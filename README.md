# Fingerprint-Generation-and-Similarity-Comparison-using-CNN
This project aims to generate fingerprints from images using a custom-trained convolutional neural network (CNN) model and compare the similarity
between these fingerprints. The CNN model is trained on a dataset comprising 2000 images, which include variations in contrast, noise, zoom, and rotation. 
The goal is to demonstrate the effectiveness of the custom-trained CNN in capturing image features for fingerprint generation and similarity assessment.

## Features
Train a CNN model from scratch on a dataset of images with diverse conditions.
Generate fingerprints for input images using the trained CNN model.
Compare the similarity between fingerprints using the Euclidean distance metric.
Visualize the results and analyze image similarities across diverse conditions.

## Requirements
Python 3.x
TensorFlow
Keras
NumPy
OpenCV (cv2)
Matplotlib (for visualization)

## Usage
Train the CNN model
Generate fingerprints for input images
Compare the similarity between fingerprints

## Instructions
Run the Python script Fingerprint.py

#### Train the CNN model
## Requirements

- Python 3.x
- TensorFlow 2.x
- scikit-learn
- pandas

## Dataset
The dataset consists of two folders: Class1 and Class2, 
each containing 1000 JPEG images. All images in Class1 belong to Class1, and all images in Class2 belong to Class2. 
The paths to the folders are:
Class1 folder: Your Path\Class1
Class2 folder: Your Path\Class2

## Instructions
Run the Python script CNN.py
After training, the model will be saved as my_cnn_model.h5 in the current directory.

## Customization
You can modify the script CNN.py to adjust hyperparameters, such as image size, batch size, number of epochs, etc.
If your dataset directory is different, make sure to update the paths accordingly in the script.
