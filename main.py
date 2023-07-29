from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from PIL import Image, UnidentifiedImageError
import numpy as np
import io
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model
import tensorflow as tf
import cv2
import logging
import tempfile

# Define a function to preprocess the image
def preprocess_image(image, target_size):
    # Resize the image to target size and convert to array
    image = image.resize(target_size)
    image = img_to_array(image)
    
    # Apply the image generator transformations
    image = datagen.random_transform(image)
    image = datagen.standardize(image)
    
    # Add a dimension for batch size
    image = np.expand_dims(image, axis=0)
    
    return image

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

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

# Define the likert scale image file paths
image_paths = {
    0: "static/likert/0.png",
    1: "static/likert/1.png",
    2: "static/likert/2.png",
    3: "static/likert/3.png",
    4: "static/likert/4.png",
}

@app.get("/")
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about")
async def about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/submit")
async def submit(request: Request):
    return templates.TemplateResponse("submit.html", {"request": request})

@app.get("/error")
async def error(request: Request):
    return templates.TemplateResponse("error.html", {"request": request})

@app.get("/result")
async def result(request: Request, prediction_inception=None, probability_inception=None, prediction_xception=None, probability_xception=None, prediction=None, probability=None, explanation_text=None, likert_scale_path=None, category=None, category_inception=None, category_xception=None):
    return templates.TemplateResponse(
        "result.html", 
        {
        "request": request, 
        "prediction_inception": prediction_inception, 
        "probability_inception": probability_inception,
        "prediction_xception": prediction_xception, 
        "probability_xception": probability_xception,
        "prediction": prediction, 
        "probability": probability,
        "explanation_text": explanation_text, 
        "likert_scale_path": likert_scale_path,
        "category": category,
        "category_inception": category_inception, 
        "category_xception": category_xception,
        "superimposed_img_path": "static/superimposed_img.jpg"
        }
    )

@app.post("/predict")
async def predict(request: Request, image: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await image.read()
        image_stream = io.BytesIO(contents)
        image_file = Image.open(image_stream).convert('RGB')
        image_array = preprocess_image(image_file, target_size=(299, 299))

        # Make a prediction using the loaded models
        prediction_inception = model_inception.predict(image_array)[0]
        predicted_category_inception = np.argmax(prediction_inception)
        predicted_probability_inception = round(prediction_inception[predicted_category_inception]*100)
        category_inception = get_category(int(predicted_category_inception))
        logging.info(f"category_inception: {category_inception}")

        prediction_xception = model_xception.predict(image_array)[0]
        predicted_category_xception = np.argmax(prediction_xception)
        predicted_probability_xception = round(prediction_xception[predicted_category_xception]*100)
        category_xception = get_category(int(predicted_category_xception))
        logging.info(f"category_xception: {category_xception}")

        # Average the predictions
        prediction = (prediction_inception + prediction_xception) / 2.0
        predicted_category = np.argmax(prediction)
        predicted_probability = round(prediction[predicted_category]*100)
        category = get_category(int(predicted_category))

        # Define the explanation text based on the predicted category
        explanation_text = get_explanation_text(int(predicted_category))

        # Define the likert scale image path based on the predicted category
        likert_scale_path = image_paths[int(predicted_category)]

        # Create a separate model for Grad-CAM
        grad_model = Model([model_inception.inputs], 
                    [model_inception.get_layer('mixed10').output, 
                    model_inception.output])

        # Generate Grad-CAM heatmap
        with tf.GradientTape(persistent=True) as tape:
            image_tensor = tf.convert_to_tensor(image_array)  # Convert to TensorFlow tensor
            tape.watch(image_tensor)
            last_conv_layer_output, preds = grad_model(image_tensor)  # Pass the input through the grad_model
            top_pred_index = tf.argmax(preds[0])
            top_class_channel = preds[:, top_pred_index]

        # This is the gradient of the predicted class with regard to
        # the output feature map of 'mixed10'
        grads = tape.gradient(top_class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the predicted class
        last_conv_layer_output = last_conv_layer_output.numpy()[0]
        pooled_grads = pooled_grads.numpy()
        for i in range(pooled_grads.shape[-1]):
            last_conv_layer_output[:, :, i] *= pooled_grads[i]

        # The channel-wise mean of the resulting feature map
        # is our heatmap of class activation
        heatmap = np.mean(last_conv_layer_output, axis=-1)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

        # Create a temporary file in memory
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_image:
            image_file.save(temp_image.name)

            # Use cv2 to resize the heatmap to original image size
            img = cv2.imread(temp_image.name)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

        # Convert the heatmap to RGB
        heatmap = np.uint8(255 * heatmap)

        # We apply the heatmap to original image
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 0.4 here is a heatmap intensity factor
        superimposed_img = heatmap * 0.4 + img

        # Save the image to a file
        superimposed_img_path = 'static/superimposed_img.jpg'
        superimposed_img = cv2.imwrite(superimposed_img_path, superimposed_img)

        return RedirectResponse(
            url=f"/result?prediction_inception={predicted_category_inception}&probability_inception={predicted_probability_inception}&category_inception={category_inception}&prediction_xception={predicted_category_xception}&probability_xception={predicted_probability_xception}&category_xception={category_xception}&prediction={predicted_category}&explanation_text={explanation_text}&likert_scale_path={likert_scale_path}&category={category}&probability={predicted_probability}&superimposed_img_path={superimposed_img_path}",
            status_code=303
        )

    except UnidentifiedImageError:
        # Return an error message if the image format is not recognized
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": "The uploaded file is not recognized as an image file."}
        )


def get_category(predicted_category):
    if predicted_category == 0: 
        return "Normal"
    elif predicted_category == 1:
        return "Mild"
    elif predicted_category == 2:
        return "Moderate"
    elif predicted_category == 3:
        return "Severe"
    elif predicted_category == 4:
        return "Proliferative"

def get_explanation_text(predicted_category):
    if predicted_category == 0:
        return "No diabetic retinopathy (DR) was detected in this image."
    elif predicted_category == 1:
        return "This means that some small areas of the retina may have damaged blood vessels or swelling."
    elif predicted_category == 2:
        return "Moderate DR was detected in this image. This means that there is a more widespread area of the retina affected by damaged blood vessels or swelling."
    elif predicted_category == 3:
        return "Severe DR was detected in this image. This means that a large portion of the retina is affected by damaged blood vessels or swelling."
    elif predicted_category == 4:
        return "Proliferative diabetic retinopathy (PDR) was detected in this image. This means that there is a significant amount of new blood vessel growth on the retina, which can lead to serious vision problems or even blindness."