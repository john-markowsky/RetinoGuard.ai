# ![image](https://user-images.githubusercontent.com/123923257/216790444-643e3e84-f2f8-42f2-b59b-3c0cf9c26ccf.png)
# Retino Guard AI: Decision Support System for Diabetic Retinopathy Prevention

## Introduction
Diabetic retinopathy (DR) is an extremely serious but all too common complication of diabetes mellitus which causes blindness. “Diabetic retinopathy is the most frequently occurring complication of diabetes mellitus and remains a leading cause of vision loss globally. Its etiology and pathology have been extensively studied for half a century, yet there are disappointingly few therapeutic options.”(Stitt et al., 2016). Early detection and proper management of diabetic retinopathy can reduce the possibility of vision loss, but with current methods, diabetic retinopathy is difficult to diagnose early. With the rate of diabetes climbing yearly, there is a pressing need for an intelligent system to support healthcare providers in detecting and managing diabetic retinopathy.

![aux_img](https://user-images.githubusercontent.com/123923257/234385052-b9b4bb72-a0db-49d2-94b7-8f51c7dcbdfc.png)

## Project Aims
The goal of this project is to develop a system which helps diagnose and predict the risk of diabetic retinopathy, allowing for earlier intervention and treatment to combat blindness. Our Retino Guard AI system will be trained on the APTOS 2019 Blindness Detection Data Set to classify the images and determine the severity of retinopathy. We aim to produce a device which accurately assists healthcare professionals detect and prevent diabetic retinopathy via early diagnosis.


## Methods
The AI methods that will be used in this will both be from the supervised learning category; Support Vector Machines and Artificial Neural Networks. "Neural network applications in computer-aided diagnosis represent the main stream of computational intelligence in medical imaging. Their penetration and involvement are almost comprehensive for all medical problems due to the fact that neural networks have the nature of adaptive learning from input information"(Jiang et al., 2010). Because of their specialized abilities in both classification and image recognition, these techniques will be used to train the Decision Support System (DSS) to classify by severity the images contained in the dataset. 

## Sources
The knowledge, documents, and data required for this project is found on the Kaggle’s APTOS 2019 Blindness Detection Competition Data Set. Kaggle is an online community platform for data scientists and machine learning enthusiasts. It hosts the dataset which includes over 115,000 images which will be used to train our system. 
[APTOS 2019 Blindness Detection Data Set (Kaggle)](https://www.kaggle.com/competitions/aptos2019-blindness-detection)

## Models
For this project, a DenseNet201 model was used for image classification. The model was trained on the APTOS 2019 dataset with a binary cross-entropy loss function, the Adam optimizer with a learning rate of 0.0001, and the evaluation metrics of accuracy and Cohen's kappa. The model achieved an accuracy of 0.825 on the validation set and 0.81 on the test set.

![__results___19_0](https://user-images.githubusercontent.com/123923257/234801053-970a8355-275d-4418-9a44-cc228346d590.png)

The DenseNet201 model is a convolutional neural network architecture that was introduced by Huang et al. in 2017. The network is characterized by dense connections between layers, which allow for efficient feature reuse and facilitate training of very deep neural networks.

The model includes an input layer that takes in 224x224 pixel RGB images, followed by a pre-trained DenseNet201 layer from the ImageNet dataset, which extracts features from the input image. The output from this layer is then passed through a global average pooling layer that computes the average value of each feature map, reducing the dimensions of the feature maps.

The resulting feature vector is then passed through a dropout layer to reduce overfitting and a fully connected layer with a sigmoid activation function that outputs a probability score for each of the 5 classes of the Aptos 2019 Blindness Detection Challenge.

![__results___23_0](https://user-images.githubusercontent.com/123923257/234801630-190f7d6b-5bd0-46c0-ac52-6a0e9d773cb5.png)

## Web Application
The Retino Guard AI web application was developed using FastAPI and Python to create a Decision Support System (DSS) for healthcare professionals, particularly ophthalmologists and optometrists, involved in the diagnosis and treatment of diabetic retinopathy patients. The system allows for the easy and efficient upload of retinal images by healthcare professionals and provides predictions for the severity of diabetic retinopathy.
![Submit](https://user-images.githubusercontent.com/123923257/234384778-efed0ff5-fd3f-4123-a4ce-929a5c571c6e.png)

Using advanced machine learning techniques, the system is capable of accurately predicting the severity of diabetic retinopathy based on the uploaded retinal image. Additionally, the system provides an explanation of the prediction to aid in the decision-making process of healthcare professionals.
![Prediction](https://user-images.githubusercontent.com/123923257/234384721-b268fdf1-1c38-44e3-84e4-dfad1ec29991.png)

## Clients
The clients we intend to serve are healthcare professionals, especially ophthalmologists and optometrists who are involved in the diagnosis and treatment of diabetic retinopathy patients. We imagine that clinics and hospitals which specialize in treating diabetic patients might also show interest as stakeholders in the Retino Guard AI project.

## Tools
The main programming language that will be used for this project is python and html. Several libraries that exist in python will serve as tools to help us train our system including TensorFlow and Keras.


## Data Dictionary
![image](https://user-images.githubusercontent.com/123923257/217098210-f64a661b-55a0-4a9e-a26f-d0c3d8cd9a48.png)

## Conclusion
The proposed Retino Guard AI project will contribute with detection and prevention of diabetes retinopathy, providing healthcare professionals with a Decision Support System trained on a large dataset of retinal images. Not only does the project sync with course objectives, with the rise of diabetes worldwide, we believe it will be a valuable tool for healthcare providers around the globe.
