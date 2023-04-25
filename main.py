from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import io
from PIL import Image, UnidentifiedImageError
import numpy as np
import tensorflow as tf
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from lime.lime_image import LimeImageExplainer
from fastapi.responses import RedirectResponse

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")  # add this line to mount the static files


# Load the model
model = tf.keras.models.load_model('model.h5')

# Define the explainer
explainer = LimeImageExplainer()

# Define the likert scale image file paths
image_paths = {
    0: "static/likert/0.png",
    1: "static/likert/1.png",
    2: "static/likert/2.png",
    3: "static/likert/3.png",
    4: "static/likert/4.png",
}

# Define the Index page.
@app.get("/")
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Define the About page.
@app.get("/about")
async def homepage(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

# Define the Submit a Photo page.
@app.get("/submit")
async def submit(request: Request):
    return templates.TemplateResponse("submit.html", {"request": request})

# Define the Error page.
@app.get("/error")
async def submit(request: Request):
    return templates.TemplateResponse("error.html", {"request": request})

# Define the results page
@app.get("/result")
async def homepage(request: Request, prediction=None, explanation_text=None, likert_scale_path=None):
    return templates.TemplateResponse(
        "result.html", 
        {"request": request, 
         "prediction": prediction, 
         "explanation_text": explanation_text, 
         "likert_scale_path": likert_scale_path}
    )


@app.post("/predict")
async def predict(request: Request, image: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await image.read()
        image_stream = io.BytesIO(contents)
        image_file = Image.open(image_stream).convert('RGB')

        # Preprocess the image
        image_resized = image_file.resize((224, 224))
        image_array = np.array(image_resized) / 255.0
        image_expanded = np.expand_dims(image_array, axis=0)

        # Make a prediction using the loaded model
        prediction = model.predict(image_expanded)[0]

        # Get the index of the predicted category
        predicted_category = np.argmax(prediction)

        # Generate an explanation
        explanation = explainer.explain_instance(image_array, model.predict, ...)

        # Define the explanation text based on the predicted category
        explanation_text = ""
        if predicted_category == 0:
            explanation_text = "No diabetic retinopathy (DR) was detected in this image."
        elif predicted_category == 1:
            explanation_text = "This means that some small areas of the retina may have damaged blood vessels or swelling."
        elif predicted_category == 2:
            explanation_text = "Moderate DR was detected in this image. This means that there is a more widespread area of the retina affected by damaged blood vessels or swelling."
        elif predicted_category == 3:
            explanation_text = "Severe DR was detected in this image. This means that a large portion of the retina is affected by damaged blood vessels or swelling."
        elif predicted_category == 4:
            explanation_text = "Proliferative diabetic retinopathy (PDR) was detected in this image. This means that there is a significant amount of new blood vessel growth on the retina, which can lead to serious vision problems or even blindness."

        # Define the likert scale image path based on the predicted category
        likert_scale_path = image_paths[predicted_category]

        # Return the prediction, explanation, explanation text, and likert scale image path to the webpage
        return RedirectResponse(
            url=f"/result?prediction={prediction}&explanation_text={explanation_text}&likert_scale_path={likert_scale_path}",
            status_code=303
        )

    except UnidentifiedImageError:
        # Return an error message if the image format is not recognized
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": "The uploaded file is not recognized as an image file."}
        )



