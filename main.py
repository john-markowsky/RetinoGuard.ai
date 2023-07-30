from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from PIL import Image, UnidentifiedImageError
import base64
import io
import logging

from models import preprocess_image, predict, model_inception
from categories import get_category, get_explanation_text
from image_processing import generate_gradcam_heatmap

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about")
async def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/submit")
async def submit_page(request: Request):
    return templates.TemplateResponse("submit.html", {"request": request})

@app.get("/error")
async def error_page(request: Request):
    return templates.TemplateResponse("error.html", {"request": request})

@app.get("/result")
async def result_page(request: Request, prediction_inception=None, probability_inception=None, prediction_xception=None, probability_xception=None, prediction=None, probability=None, explanation_text=None, category=None, category_inception=None, category_xception=None, superimposed_img_path=None, original_img_path=None):
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
        "category": category,
        "category_inception": category_inception, 
        "category_xception": category_xception,
        "superimposed_img_path": superimposed_img_path,
        "original_img_path": original_img_path
        }
    )

@app.post("/predict")
async def predict_page(request: Request, image: UploadFile = File(...)):
    try:
        contents = await image.read()
        image_stream = io.BytesIO(contents)
        image_file = Image.open(image_stream).convert('RGB')
        image_array = preprocess_image(image_file, target_size=(299, 299))

        predicted_category_inception, predicted_probability_inception, predicted_category_xception, predicted_probability_xception, predicted_category, predicted_probability = predict(image_array)
        category_inception = get_category(int(predicted_category_inception))
        logging.info(f"category_inception: {category_inception}")
        category_xception = get_category(int(predicted_category_xception))
        logging.info(f"category_xception: {category_xception}")
        category = get_category(int(predicted_category))
        explanation_text = get_explanation_text(int(predicted_category))
        original_img_path, superimposed_img_path = generate_gradcam_heatmap(image_array, image_file, model_inception)

        return RedirectResponse(
            url=f"/result?prediction_inception={predicted_category_inception}&probability_inception={predicted_probability_inception}&category_inception={category_inception}&prediction_xception={predicted_category_xception}&probability_xception={predicted_probability_xception}&category_xception={category_xception}&prediction={predicted_category}&explanation_text={explanation_text}&category={category}&probability={predicted_probability}&superimposed_img_path={superimposed_img_path}&original_img_path={original_img_path}",
            status_code=303
        )

    except UnidentifiedImageError:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": "The uploaded file is not recognized as an image file."}
        )