from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
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

from fastapi import FastAPI, Form
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Integer, ForeignKey, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

app = FastAPI()


# Initialize SQLAlchemy
DATABASE_URL = "sqlite:///./survey_data.db"
Base = declarative_base()
engine = create_engine(DATABASE_URL)

# Create an inspector object
inspector = inspect(engine)

# Get names of all tables in the database
existing_tables = inspector.get_table_names()

# Check and create tables only if they don't exist
if "users" not in existing_tables or "responses" not in existing_tables:
    Base.metadata.create_all(bind=engine)
# Create a table for user data
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    profession = Column(Integer)
    years_of_experience = Column(String)
    responses = relationship("Response", back_populates="user")

# Create a table for survey responses using enumerated types
class Response(Base):
    __tablename__ = 'responses'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    ease_of_navigation = Column(Integer)
    user_interface_intuitive = Column(Integer)
    model_accuracy = Column(Integer)
    model_alignment = Column(Integer)
    decision_confidence = Column(Integer)
    grad_cam_usefulness = Column(Integer)
    grad_cam_accuracy = Column(Integer)
    improvements_suggested = Column(String)
    recommend_retinoguard = Column(Integer)
    user = relationship("User", back_populates="responses")

# Create tables in the database
Base.metadata.create_all(bind=engine)

# Initialize session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Define the Pydantic model for the survey data with enums
class Survey(BaseModel):
    profession: int
    years_of_experience: str
    ease_of_navigation: int
    user_interface_intuitive: int
    model_accuracy: int
    model_alignment: int
    decision_confidence: int
    grad_cam_usefulness: int
    grad_cam_accuracy: int
    improvements_suggested: str
    recommend_retinoguard: int

# FastAPI instance
app = FastAPI()


templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output", StaticFiles(directory="output"), name="output")

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

@app.get("/consent")
async def consent(request: Request):
    return templates.TemplateResponse("consent.html", {"request": request})

@app.get("/annotate/{image_id}")
async def annotate(request: Request, image_id: str):
    image_data = {"image_id": image_id, "image_url": f"/static/misc/test_imgs/{image_id}.png"}
    return templates.TemplateResponse("annotate.html", {"request": request, "image_data": image_data})

@app.get("/survey")
async def survey_page(request: Request):
    return templates.TemplateResponse("survey.html", {"request": request})

@app.get("/complete")
async def complete(request: Request):
    return templates.TemplateResponse("complete.html", {"request": request})

@app.post("/submit_survey/")
async def submit_survey(
    profession: int = Form(...),
    years_of_experience: str = Form(None),
    ease_of_navigation: int = Form(...),
    user_interface_intuitive: int = Form(...),
    model_accuracy: int = Form(...),
    model_alignment: int = Form(...),
    decision_confidence: int = Form(...),
    grad_cam_usefulness: int = Form(...),
    grad_cam_accuracy: int = Form(...),
    improvements_suggested: str = Form(None),
    recommend_retinoguard: int = Form(...)
):
    db = SessionLocal()
    db_user = User(profession=profession, years_of_experience=years_of_experience)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    db_response = Response(
        user_id=db_user.id,
        ease_of_navigation=ease_of_navigation,
        user_interface_intuitive=user_interface_intuitive,
        model_accuracy=model_accuracy,
        model_alignment=model_alignment,
        decision_confidence=decision_confidence,
        grad_cam_usefulness=grad_cam_usefulness,
        grad_cam_accuracy=grad_cam_accuracy,
        improvements_suggested=improvements_suggested,
        recommend_retinoguard=recommend_retinoguard
    )
    db.add(db_response)
    db.commit()
    db.refresh(db_response)
    
    return RedirectResponse(url="/complete", status_code=303)


@app.get("/result")
async def result_page(request: Request, timestamp: str = None, prediction_inception=None, probability_inception=None, prediction_xception=None, probability_xception=None, prediction=None, probability=None, explanation_text=None, category=None, category_inception=None, category_xception=None):
    original_img_path = f"output/original_img_{timestamp}.jpg"
    superimposed_img_path = f"output/superimposed_img_{timestamp}.jpg"

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
        category_xception = get_category(int(predicted_category_xception))
        category = get_category(int(predicted_category))
        explanation_text = get_explanation_text(int(predicted_category))
        
        original_img_path, superimposed_img_path = generate_gradcam_heatmap(image_array, image_file, model_inception)
        timestamp = original_img_path.split('_')[-1].split('.')[0]

        return RedirectResponse(
            url=f"/result?timestamp={timestamp}&prediction_inception={predicted_category_inception}&probability_inception={predicted_probability_inception}&category_inception={category_inception}&prediction_xception={predicted_category_xception}&probability_xception={predicted_probability_xception}&category_xception={category_xception}&prediction={predicted_category}&probability={predicted_probability}&category={category}&explanation_text={explanation_text}",
            status_code=303
        )

    except UnidentifiedImageError:
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "message": "The uploaded file is not recognized as an image file."}
        )
