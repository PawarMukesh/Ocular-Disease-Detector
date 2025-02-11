# AI POWERED OCULAR DISEASE DETECTION USING TRANSFER LEARNING APPROACHES 

## Introduction
* Eye diseases are major causes of vision loss.
* Timely and accurate diagnosis is crucial for effective treatment
* Delayed or improper treatment can lead to vision loss.
* Common serious eye conditions include cataracts, diabetic retinopathy, and glaucoma
* Vision loss can significantly impact overall well-being

## problem statement 
This project aims to develop a transfer learning-based AI system to detect and classify a wide range of ocular diseases from medical imaging data, helping to improve diagnostic accuracy, reduce healthcare costs, and provide timely interventions. 

## The Need of ThIs TopIc
* Early Diagnosis
* Vision Protection
* Lower in Price
* Monitoring

## ObjectIves
1. Design and Develop Deep Learning Models: Create models to detect all types of ocular diseases.
2. Evaluate and Compare Transfer Learning Models: Analyze state-of-the-art models to improve accuracy and detection speed.
3. Integrate XAI Strategies: Use Explainable AI to build a precise and secure system for identifying ocular diseases from retinal images.
4. Develop an Intuitive Interface: Provide doctors with immediate and meaningful information about detected ocular diseases.

## Methodology
![image](https://github.com/user-attachments/assets/badc25ef-bea9-44c0-8597-d950c56fdf39)

## Data Collection
* Data collected from Roboflow, Total 6009 images divided into four classes.
* url = { https://universe.roboflow.com/projects-dmmza/ocular-diseases }
  * publisher = { Roboflow }
  * year = { 2024 }
  * month = { may }
 
## Data preprocessIng
**1. Clahe**
   * Contrast Limited Adaptive Histogram Equalization
   * Enhanced Contrast
   * Standardized Input
   * Improved Detection Accuracy
**2. Data augmentatIon**
   * Increases Data Diversity
   * Reduces Overfitting
   * Enhances Generalization
   * Makes Models Stronger

## Data splitting
* Use split-folder library to divide data into three parts with ratio (80%, 10%, 10%).
* Remove the duplicate samples from training set.
* Diversify the dataset by adding more number of samples.

## VIsualIse TraInIng Images
![image](https://github.com/user-attachments/assets/e813b4e7-c1f9-41ab-a4e6-00a5e04fdefd)

## TRAINING PHASE
* Experiment with different models: VGG19, VGG16, InceptionV3, Xception.
* Build a custom Deep Convolutional Neural Network (DCNN).
* Layers Used In Above Model:
  1. Convolutional Layer
  2. MaxPooling Layer
  3. Flatten Layer
  4. Dense Layer
  5. Dropout Layer
  6. Output Layer
 
### 1.Deep convolutIonal neural network
* Input Layer:
  - Size: 224x224 RGB images.
* Convolutional Blocks (4 Blocks):
  - Each Block:
    1. 1 Conv2D layer with 'relu' activation.
    2. 1 MaxPooling2D layer.
* Flatten Layer
* Two Dense Layers
* Output Layer with softmax activation.
* Model Summary:
  - Total Parameters: 19,427,520.






























