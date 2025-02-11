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
 
![image](https://github.com/user-attachments/assets/c06e76f8-4d24-4b35-9806-bc3f207e1b62)

#### **IntegratIon Of DCNN Model WIth XAI**
**1. LIME (Local Interpretable Model-agnostic Explanations)**
   - Identifies strong feature matches in the image.
   - Formula For Lime:
     * (𝒙)=𝑳(𝒇,𝒈,𝜫𝒙)+𝜴(𝒈)
   - The ultimate goal is to reduce the defectiveness L(f,g,∏x), and complexity Ω(g) to generate best interpretation and explanation. 
**2. Occlusion Sensitivity**
   - Shows how blocking parts of the image affects the model's output
   - Formula:
     * 𝐼(𝑖,𝑗)=𝑃(├ 𝑐┤|𝐼)−𝑃(𝑜𝑐𝑐𝑙𝑢𝑑𝑒𝑑__𝐼(𝐼,𝑗))
       - Where,
         *  P(c|I): Correct class c probability
         *  P(occluded__I(I,j)): specified image with occluded pixel(i,j) 

![image](https://github.com/user-attachments/assets/92d18a86-0527-489c-9e1c-43c054a24b06)

### 2. vgg19 model
* VGG: Stands for Visual Geometry Group.
* Total Layers: 19 (16 Conv2D, 3 Dense, MaxPooling2D).
* Model Loading:
  - Load VGG19 pretrained model (ImageNet weights).
  - nput Size: 256x256 pixels.
  - Exclude fully connected layers.
* Custom layers:
  - Flatten the output.
  - Use Dropout to prevent overfitting.
  - Final Dense layer with 'softmax' for multiclass classification.
* Model Summary:
  - Total Trainable Parameters: 20,155,460.
![image](https://github.com/user-attachments/assets/de061ba6-9261-4587-bf39-f18a18f9085d)

### 3. vgg16 model
* VGG16: Developed for large-scale image classification. 
* Total Layers: 16 (13 Conv2D, 3 Dense).
* Model Loading & Customization:
  - Load VGG16 pretrained model with (ImageNet weights).
  - Input Size: 256x256 pixels..
  - Add a custom Flatten layer.
  - Add an output layer with softmax activation function for classification.
 
 ![image](https://github.com/user-attachments/assets/a2613ff2-d7a6-417d-8a20-45a4300ea3ed)

 ### 3. Inceptionv3 model 
* InceptionV3 Overview:
  - Purpose: Extracts diverse features for accurate detection of retinal conditions.
  - pplications: Macular degeneration, diabetic retinopathy.
* Model Loading:
  - Load InceptionV3 pretrained model (ImageNet weights).
  - Input Size: 256x256 pixels.
  - Exclude fully connected layers. 
* Customization:
  - Add a GlobalAveragePooling2D layer.
  - Add a Dense layer: 1024 neurons, 'ReLU' activation.
  - Include a Dropout layer to prevent overfitting.
  - Add an Output layer: 4 neurons, 'softmax' activation for multiclass classification

### 4.XCEPTION
* Xception Overview:
  - Inspired by Inception architecture.
  - Features depthwise separable convolutions and residual connections. 
* Model Loading:
  - Load pretrained Xception model (ImageNet weights).
  - Input Size: 256x256 pixels.
  - Exclude fully connected layers.
* Customization:
  - Add a Flatten layer.
- Add an Output layer: 4 neurons, 'softmax' activation.

## Train and Test loss of different models
![image](https://github.com/user-attachments/assets/9a584245-7725-49d9-b682-5c5a3dbfc9f2)
* Lowest Training Loss: VGG19, Xception, and DCNN models.
* Good Performance: InceptionV3 had training losses below 0.15.
* Lowest Testing Error: Xception, DCNN, VGG19, and InceptionV3 models


## Trian &Test Accuracy as well as Loss of Different Models
![image](https://github.com/user-attachments/assets/c76aa504-4db2-494a-93e4-5e910c252997)


## F1 Score of all models
![image](https://github.com/user-attachments/assets/715a7f03-6dcd-4843-837b-6563e1bc33aa)

## PREDICTED IMAGES 
![image](https://github.com/user-attachments/assets/e035b1a9-7398-483c-9e09-76947bd3b2a7)
![image](https://github.com/user-attachments/assets/df3befc8-eaf4-4dcd-bb6b-25777f433a2f)


## Conclusion
In conclusion, both DCNN and InceptionV3 models performed excellently, with over 95% training accuracy and 90% testing accuracy. They effectively generalized across diverse eye disease images and had F1 scores above 90%, indicating minimal misclassifications. Using Explainable AI (XAI) techniques, such as LIME and Occlusion Sensitivity, we further validated the models by highlighting key image features and identifying influential areas. This project confirms that deep learning is highly effective for ocular disease detection.

## Refrences
1. Ahmed, M. R., Ahmed, S. R., Duru, A. D., Uçan, O. N., & Bayat, O. (2021). An expert system to predict eye disorder using deep convolutional neural network. Academic Platform-Journal of Engineering and Science, 9(1), 47-52.
2. Leonardo, M. M., Carvalho, T. J., Rezende, E., Zucchi, R., & Faria, F. A. (2018, October). Deep feature-based classifiers for fruit fly identification (Diptera: Tephritidae). In 2018 31st SIBGRAPI conference on graphics, patterns and images (SIBGRAPI) (pp. 41-47). IEEE.
3. Rothman Denis, "Hands-On Explainable AI (XAI) with Python: Interpret visualize explain and integrate reliable AI for fair secure and trustworthy AI apps", Birmingham: Packt Publishing Limited, 2020.
4. Bitto, A. K., & Mahmud, I. (2022). Multi categorical of common eye disease detect using convolutional neural network: a transfer learning approach. Bulletin of Electrical Engineering and Informatics, 11(4), 2378-2387.
5. Tammina, S. (2019). Transfer learning using vgg-16 with deep convolutional neural network for classifying images. International Journal of Scientific and Research Publications (IJSRP), 9(10), 143-150
6. Vujović, Ž. (2021). Classification model evaluation metrics. International Journal of Advanced Computer Science and Applications, 12(6), 599-606.






















































