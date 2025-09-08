import numpy as np
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware




# Get the folder where this script is located
BASE_DIR = os.path.dirname(__file__)

# Combine with your model filename
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.sav")

# Load the model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5500"] for specific frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema for API requests
class InputData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.get("/")
def home():
    return {"message": "Diabetes Prediction API is running"}

@app.post("/predict")
def predict(data: InputData):
    input_data = np.array([
        data.Pregnancies, data.Glucose, data.BloodPressure,
        data.SkinThickness, data.Insulin, data.BMI,
        data.DiabetesPedigreeFunction, data.Age
    ]).reshape(1, -1)

    prediction = model.predict(input_data)
    result = "Non Diabetic" if prediction[0] == 0 else "Diabetic"
    return {"prediction": result}
# To run the app, use the command:

# uvicorn temp2:app --reload

