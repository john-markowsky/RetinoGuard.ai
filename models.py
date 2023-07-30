from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
import cv2
from PIL import Image
import io

# Load the models
model_inception = load_model('.models/inception.hdf5')
model_xception = load_model('.models/xception.hdf5')

# Define the image generator for preprocessing
datagen = ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.5, 1.5],
    rescale=1./255
)

def preprocess_image(image, target_size):
    # Resize the image to the target size and convert it to an array
    image = image.resize(target_size)
    image = img_to_array(image)
    
    # Apply the image generator transformations
    image = datagen.random_transform(image)
    image = datagen.standardize(image)
    
    # Add a dimension for batch size
    image = np.expand_dims(image, axis=0)
    
    return image

def predict(image_array):
    prediction_inception = model_inception.predict(image_array)[0]
    predicted_category_inception = np.argmax(prediction_inception)
    predicted_probability_inception = round(prediction_inception[predicted_category_inception]*100)

    prediction_xception = model_xception.predict(image_array)[0]
    predicted_category_xception = np.argmax(prediction_xception)
    predicted_probability_xception = round(prediction_xception[predicted_category_xception]*100)

    # Average the predictions
    prediction = (prediction_inception + prediction_xception) / 2.0
    predicted_category = np.argmax(prediction)
    predicted_probability = round(prediction[predicted_category]*100)

    return predicted_category_inception, predicted_probability_inception, predicted_category_xception, predicted_probability_xception, predicted_category, predicted_probability
