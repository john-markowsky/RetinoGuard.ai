# ![image](https://user-images.githubusercontent.com/123923257/216790444-643e3e84-f2f8-42f2-b59b-3c0cf9c26ccf.png)
# RetinoGuard.ai: Decision Support System for Diabetic Retinopathy Prevention

## Introduction
Diabetic retinopathy (DR) is a serious complication of diabetes mellitus that can lead to blindness. Early detection and proper management can drastically reduce the risk of vision loss, but early diagnosis can be challenging. With the prevalence of diabetes increasing each year, there is a pressing need for intelligent systems to support healthcare providers in detecting and managing diabetic retinopathy. RetinoGuard.ai is our contribution to addressing this need.

![aux_img](https://user-images.githubusercontent.com/123923257/234385052-b9b4bb72-a0db-49d2-94b7-8f51c7dcbdfc.png)

## Project Aims
The goal of this project is to develop a system that helps diagnose and predict the risk of diabetic retinopathy, enabling earlier intervention and treatment to prevent blindness. Our RetinoGuard.ai system will be trained on the APTOS 2019 Blindness Detection Data Set to classify images and determine the severity of retinopathy.

## Methods
Our system employs a sophisticated ensemble learning approach that combines the predictions from multiple machine learning models, including InceptionV3, Xception, and a combined model. These models were chosen for their excellent performance in image classification tasks, and are trained using the APTOS 2019 dataset.

## Sources
The knowledge, documents, and data required for this project is obtained from the Kaggle’s APTOS 2019 Blindness Detection Competition Data Set, which includes over 115,000 images for training our system. [APTOS 2019 Blindness Detection Data Set (Kaggle)](https://www.kaggle.com/competitions/aptos2019-blindness-detection)

## Models
Our ensemble model combines the predictions from two pretrained models (InceptionV3 and Xception) to output a final prediction. This approach leverages the strengths of each individual model to improve overall prediction accuracy and robustness.

## Web Application
RetinoGuard.ai provides a user-friendly web interface for healthcare professionals to upload retinal images and receive predictions for the severity of diabetic retinopathy. The application is built with FastAPI and Python, and provides an easy-to-use platform for diagnosing and managing diabetic retinopathy.

In addition to providing a prediction, the system generates a Grad-CAM heatmap overlay on the original image to visualize the areas that the model found most relevant for its prediction. This provides a powerful tool for healthcare professionals to understand the basis of the AI's decision.

## Clients
RetinoGuard.ai is designed to serve healthcare professionals, particularly ophthalmologists and optometrists, involved in diagnosing and treating diabetic retinopathy. Clinics and hospitals specializing in treating diabetic patients may also find RetinoGuard.ai a valuable tool in their diagnostic processes.

## Tools
The main programming language used for this project is Python, with HTML for web interface design. We utilize several Python libraries, including TensorFlow and Keras, to train our models and build the application.

## Data Dictionary
![image](https://user-images.githubusercontent.com/123923257/217098210-f64a661b-55a0-4a9e-a26f-d0c3d8cd9a48.png)

## Conclusion
RetinoGuard.ai provides a valuable tool for healthcare providers, using advanced machine learning techniques to assist in the early detection and management of diabetic retinopathy. With the global rise in diabetes, such a tool can make a significant impact in preventing vision loss and improving patient outcomes.
