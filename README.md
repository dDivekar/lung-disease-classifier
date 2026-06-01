# Lung Disease Classification using MobileNetV2

A deep learning-based medical imaging system that classifies chest X-ray images into three categories: **Fibrosis**, **Pneumonia**, and **Normal**. The project utilizes transfer learning with MobileNetV2 to improve classification performance while reducing training time.

---

## Overview

Lung diseases such as Pneumonia and Fibrosis can significantly impact respiratory health. Early detection through medical imaging can assist healthcare professionals in diagnosis and treatment planning.

This project uses a pre-trained MobileNetV2 model and transfer learning techniques to automatically classify chest images into multiple disease categories.

---

## Features

- Medical Image Classification
- Multi-Class Disease Detection
- Transfer Learning using MobileNetV2
- Image Preprocessing and Normalization
- Data Augmentation
- Chest X-ray Analysis
- Deep Learning-Based Prediction
- Visualization of Model Predictions

---

## Technologies Used

- Python
- NumPy
- OpenCV
- TensorFlow
- Keras
- MobileNetV2
- Matplotlib
- Scikit-learn

---

## Dataset

The dataset contains chest X-ray images belonging to three categories:

### Classes

- Fibrosis
- Pneumonia
- Normal

Images are resized and normalized before being used for training.

---

## Workflow

### 1. Data Loading

- Extract image dataset from ZIP file
- Load images using OpenCV
- Assign labels to each class

### 2. Data Preprocessing

- Resize images
- Normalize pixel values
- Shuffle dataset
- Split into training and testing sets

### 3. Data Augmentation

Image augmentation techniques used:

- Horizontal Flip
- Zoom
- Shear Transformation

```python
ImageDataGenerator()
```

---

### 4. Transfer Learning

The project uses MobileNetV2 pre-trained on ImageNet.

```python
MobileNetV2(
    weights='imagenet',
    include_top=False
)
```

Benefits:

- Faster training
- Better feature extraction
- Improved performance on limited datasets

---

### 5. Model Architecture

```text
Input Image
      ↓
MobileNetV2
      ↓
Global Average Pooling
      ↓
Dense Layer (ReLU)
      ↓
Dropout
      ↓
Softmax Output Layer
```

---

### 6. Training

The model is trained using:

- Adam Optimizer
- Sparse Categorical Crossentropy Loss
- Accuracy Metric

---

### 7. Prediction

The trained model predicts whether a chest image belongs to:

- Fibrosis
- Pneumonia
- Normal

---

## Deep Learning Concepts Used

- Computer Vision
- Convolutional Neural Networks (CNNs)
- Transfer Learning
- Image Classification
- Data Augmentation
- Multi-Class Classification

---

## Project Structure

```text
lung-disease-classification/
│
├── lung_disease_classification.ipynb
├── archive.zip
└── README.md
```

---

## Results

The model successfully classifies chest X-ray images into multiple disease categories using transfer learning.

Example output:

```text
Predicted: Pneumonia
Actual: Pneumonia

Predicted: Normal
Actual: Normal

Predicted: Fibrosis
Actual: Fibrosis
```

---

## Future Improvements

- Fine-Tune MobileNetV2 Layers
- Compare with ResNet50 and EfficientNet
- Add Grad-CAM Explainability
- Build a Streamlit Web Application
- Deploy as a Medical Imaging Assistant
- Improve Dataset Size and Diversity

---

## Learning Outcomes

Through this project, I learned:

- Medical Image Processing
- Transfer Learning
- CNN-Based Classification
- Data Augmentation Techniques
- Deep Learning Model Training
- Image Preprocessing using OpenCV
- Multi-Class Classification

---

## Author

Dhananjay Divekar
