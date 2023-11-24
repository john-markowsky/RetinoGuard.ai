from fastapi import FastAPI, File, UploadFile, Request, Form, Depends
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session
import io
import json

from models import preprocess_image, predict, model_inception
from categories import get_category, get_explanation_text
from image_processing import generate_gradcam_heatmap
from database import SessionLocal, User, SurveyAnswers, GradingInterfaceAnswers


templates = Jinja2Templates(directory="templates")

grading_data = []

# Function to load grading data from the JSON file
def load_grading_data():
    global grading_data
    with open('data/aptos_data.json', 'r') as file:
        grading_data = json.load(file)

# Call the function to load the data when the app starts
load_grading_data()

# Dependency to get the DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def add_routes(app: FastAPI):
    
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
    
    @app.get("/instructions")
    async def instructions(request: Request):
        return templates.TemplateResponse("instructions.html", {"request": request})
    
    @app.get("/grading")
    async def grading(request: Request):
        # Create a new user record for each session
        db = SessionLocal()
        new_user = User()
        db.add(new_user)
        db.commit()

        # Pass the user ID and loaded grading data to the template
        return templates.TemplateResponse("grading_interface.html", {
            "request": request, 
            "user_id": new_user.id,
            "grading_data": grading_data  # Assuming grading_data is loaded as before
        })

    @app.post("/submit_grade")
    async def submit_grade(request: Request, user_id: int = Form(...), dr_rating: int = Form(...)):
        form_data = await request.form()
        image_id = form_data['image_id']
        user_DR_rating = dr_rating  # This is the rating provided by the user

        # Load the grading data to find the pre-classified DR_rating
        global grading_data
        grading_entry = next((entry for entry in grading_data if entry['image_id'] == image_id), None)

        db = SessionLocal()
        # Find the existing record or create a new one if it doesn't exist
        grading_response = db.query(GradingInterfaceAnswers).filter_by(user_id=user_id, image_id=image_id).first()
        if not grading_response:
            # Instantiate a new GradingInterfaceAnswer with user_DR_rating and pre-classified DR_rating
            grading_response = GradingInterfaceAnswers(
                user_id=user_id,
                image_id=image_id,
                DR_rating=grading_entry['aptos_dr_rating'] if grading_entry else None,  # Use the pre-classified DR_rating
                user_DR_rating=user_DR_rating,
                visual_accuracy=None,
                severity_accuracy=None
            )
            db.add(grading_response)
        else:
            # Update the existing record with the new user_DR_rating
            grading_response.user_DR_rating = user_DR_rating
            # Ensure the DR_rating is up to date
            grading_response.DR_rating = grading_entry['aptos_dr_rating'] if grading_entry else grading_response.DR_rating

        db.commit()
        db.close()

        return JSONResponse(content={"status": "success", "image_id": image_id})

    @app.post("/submit_accuracy")
    async def submit_accuracy(request: Request, user_id: int = Form(...)):
        form_data = await request.form()
        image_id = form_data['image_id']
        
        visual_accuracy = form_data.get('visual_accuracy')
        severity_accuracy = form_data.get('severity_accuracy')
        # ... capture any other fields from the form

        db = SessionLocal()
        # Find the existing record and update it
        grading_response = db.query(GradingInterfaceAnswers).filter_by(user_id=user_id, image_id=image_id).first()
        if grading_response:
            grading_response.visual_accuracy = visual_accuracy
            grading_response.severity_accuracy = severity_accuracy
            # ... update any other fields
            db.commit()

        db.close()
        return JSONResponse(content={"status": "success", "image_id": image_id})

    
    @app.get("/survey")
    async def survey_page(request: Request):
        user_id = request.query_params.get('user_id')
        return templates.TemplateResponse("survey.html", {"request": request, "user_id": user_id})

    question_id_map = {
        "profession": "profession",
        "years_of_experience": "years_of_experience",
        "model_accuracy": "model_accuracy",
        "model_alignment": "model_alignment",
        "decision_confidence": "decision_confidence",
        "grad_cam_usefulness": "grad_cam_usefulness",
        "grad_cam_accuracy": "grad_cam_accuracy",
        "improvements_suggested": "improvements_suggested",
        "recommend_retinoguard": "recommend_retinoguard"
    }


    @app.post("/submit_survey/")
    async def submit_survey(request: Request, user_id: int = Form(...), db: Session = Depends(get_db)):
        form_data = await request.form()

        for key, value in form_data.items():
            if key != "user_id":  # Assuming 'user_id' is not a question but metadata
                question_id = question_id_map.get(key, key)
                survey_response = SurveyAnswers(user_id=user_id, question_id=question_id, answer=value)
                db.add(survey_response)

        db.commit()
        return JSONResponse(content={"status": "success", "message": "Survey response submitted successfully"})

    @app.get("/complete")
    async def complete(request: Request):
        return templates.TemplateResponse("complete.html", {"request": request})

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
            
            superimposed_img = generate_gradcam_heatmap(image_array, model_inception)

            # Convert the superimposed image to a PIL image
            superimposed_img_pil = Image.fromarray(superimposed_img)

            # Save the superimposed image to a BytesIO object
            superimposed_img_bytesio = io.BytesIO()
            superimposed_img_pil.save(superimposed_img_bytesio, format='JPEG')
            superimposed_img_bytesio.seek(0)

            return JSONResponse(
                content={
                    "prediction_inception": predicted_category_inception,
                    "probability_inception": predicted_probability_inception,
                    "prediction_xception": predicted_category_xception,
                    "probability_xception": predicted_probability_xception,
                    "prediction": predicted_category,
                    "probability": predicted_probability,
                    "explanation_text": explanation_text,
                    "category": category
                },
                media_type="application/json",
            )

        except UnidentifiedImageError:
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "message": "The uploaded file is not recognized as an image file."}
            )
