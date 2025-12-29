# 🖼️ Image Classification with Convolutional Neural Network (CNN) using CIFAR-10

This project focuses on building a **deep learning–based image classification model** using the **CIFAR-10 dataset**.  
A **Convolutional Neural Network (CNN)** was developed using **TensorFlow and Keras** to classify images into one of ten object categories.

---

## 📌 Project Overview

The CIFAR-10 dataset consists of **60,000 color images (32×32 pixels)** belonging to **10 different classes**:

- Airplane  
- Automobile  
- Bird  
- Cat  
- Deer  
- Dog  
- Frog  
- Horse  
- Ship  
- Truck  

The dataset is split into:
- **50,000 training images**
- **10,000 test images**

---

## 🎯 Project Objectives

- Build a CNN model for multi-class image classification  
- Apply proper image preprocessing and normalization  
- Evaluate model performance on unseen test data  
- Analyze learning behavior and overfitting  

---

## 🧹 Data Preprocessing

- Pixel values normalized to the range **[0, 1]**
- Class labels converted using **one-hot encoding**
- Input shape standardized to **32×32×3 (RGB)**

---

## 🧠 Model Architecture

A **Sequential CNN model** was designed with the following layers:

- Input layer (32×32×3)
- Convolutional Layer (32 filters, 3×3, ReLU)
- Max Pooling Layer (2×2)
- Convolutional Layer (64 filters, 3×3, ReLU)
- Max Pooling Layer (2×2)
- Flatten layer
- Fully Connected Dense Layer (128 units, ReLU)
- Output Layer (10 units, Softmax)

---

## ⚙️ Model Training

- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Metric:** Accuracy  
- **Epochs:** 10  
- **Validation:** Test dataset used for validation  

---

## 📊 Model Performance

### Training Results
- Training accuracy reached approximately **85%**

### Test Results
- **Test Accuracy:** **~70%**

### Observations
- Training loss consistently decreased
- Validation loss stabilized after several epochs
- Performance gap indicates **mild overfitting**

---

## 📈 Performance Visualization

- Training vs Validation Accuracy
- Training vs Validation Loss

These plots highlight the learning progression and generalization behavior of the model.

---

## 💾 Model Saving

The trained model was saved in `.keras` format and can be reused for:
- Further training
- Model evaluation
- Deployment in image classification applications

---

## 🛠️ Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- OpenCV  
- Matplotlib  

---

## 🧾 Conclusion

- A CNN-based image classification model was successfully implemented on the CIFAR-10 dataset.
- The model demonstrated strong learning capability on training data.
- Slight overfitting was observed, indicating potential for improvement using:
  - Data augmentation
  - Regularization
  - Deeper architectures or transfer learning
- The project provides a solid foundation for advanced computer vision tasks.

---
