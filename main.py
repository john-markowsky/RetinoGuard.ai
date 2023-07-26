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

# Load the model
model = load_model('model.hdf5')

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
async def result(request: Request, prediction=None, explanation_text=None, likert_scale_path=None, category=None, probability=None):
    return templates.TemplateResponse(
        "result.html", 
        {"request": request, 
         "prediction": prediction, 
         "explanation_text": explanation_text, 
         "likert_scale_path": likert_scale_path,
         "category": category,
         "probability": probability}
    )

@app.post("/predict")
async def predict(request: Request, image: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await image.read()
        image_stream = io.BytesIO(contents)
        image_file = Image.open(image_stream).convert('RGB')
        image_array = preprocess_image(image_file, target_size=(299, 299))

        # Make a prediction using the loaded model
        prediction = model.predict(image_array)[0]

        # Get the index of the predicted category
        predicted_category = np.argmax(prediction)
        predicted_category = np.argmax(prediction)
        predicted_probability = round(prediction[predicted_category]*100)

        # Define the explanation text and category based on the predicted category
        explanation_text = ""
        category = ""
        if predicted_category == 0:
            explanation_text = "No diabetic retinopathy (DR) was detected in this image."
            category = "Healthy"
        elif predicted_category == 1:
            explanation_text = "This means that some small areas of the retina may have damaged blood vessels or swelling."
            category = "Mild"
        elif predicted_category == 2:
            explanation_text = "Moderate DR was detected in this image. This means that there is a more widespread area of the retina affected by damaged blood vessels or swelling."
            category = "Moderate"
        elif predicted_category == 3:
            explanation_text = "Severe DR was detected in this image. This means that a large portion of the retina is affected by damaged blood vessels or swelling."
            category = "Severe"
        elif predicted_category == 4:
            explanation_text = "Proliferative diabetic retinopathy (PDR) was detected in this image. This means that there is a significant amount of new blood vessel growth on the retina, which can lead to serious vision problems or even blindness."
            category = "Proliferative"

        # Define the likert scale image path based on the predicted category
        likert_scale_path = image_paths[predicted_category]

        # Return the prediction, explanation, explanation text, and likert scale image path to the webpage
        return RedirectResponse(
            url=f"/result?prediction={prediction}&explanation_text={explanation_text}&likert_scale_path={likert_scale_path}&category={category}&probability={predicted_probability}",
            status_code=303
        )

    except UnidentifiedImageError:
        # Return an error message if the image format is not recognized
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": "The uploaded file is not recognized as an image file."}
        )