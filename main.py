"""
This module has the code for the API
"""
import os
from fastapi import FastAPI
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from pandas import DataFrame
import numpy as np
import uvicorn
from pydantic import BaseModel
from src.utils import load_artifact, process_data, get_cat_features
from src.model import inference


# Creating application
app = FastAPI()

# POST Input Schema Literals  to ensure correct input to the model


class ModelInput(BaseModel):
    age: int
    workclass: Literal['State-gov',
                       'Self-emp-not-inc',
                       'Private',
                       'Federal-gov',
                       'Local-gov',
                       'Self-emp-inc',
                       'Without-pay']
    fnlgt: int
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
        'Some-college',
        'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
        '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    education_num: int
    marital_status: Literal["Never-married",
                            "Married-civ-spouse",
                            "Divorced",
                            "Married-spouse-absent",
                            "Separated",
                            "Married-AF-spouse",
                            "Widowed"]
    occupation: Literal["Tech-support",
                        "Craft-repair",
                        "Other-service",
                        "Sales",
                        "Exec-managerial",
                        "Prof-specialty",
                        "Handlers-cleaners",
                        "Machine-op-inspct",
                        "Adm-clerical",
                        "Farming-fishing",
                        "Transport-moving",
                        "Priv-house-serv",
                        "Protective-serv",
                        "Armed-Forces"]
    relationship: Literal["Wife", "Own-child", "Husband",
                          "Not-in-family", "Other-relative", "Unmarried"]
    race: Literal["White", "Asian-Pac-Islander",
                  "Amer-Indian-Eskimo", "Other", "Black"]
    sex: Literal["Female", "Male"]
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
        'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
        'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
        'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
        'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
        'Holand-Netherlands']

    class Config:
        schema_extra = {
            "example": {
                "age": 27,
                "workclass": 'State-gov',
                "fnlgt": 77516,
                "education": 'Bachelors',
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Tech-support",
                "relationship": "Unmarried",
                "race": "White",
                "sex": "Female",
                "capital_gain": 2000,
                "capital_loss": 0,
                "hours_per_week": 35,
                "native_country": 'United-States'
            }
        }


# Load artifacts [Load the model and label encoders]
model = load_artifact("model/model.pkl")
encoder = load_artifact("model/encoder.pkl")
lb = load_artifact("model/lb.pkl")


# Root path [Application Load Root]
@app.get("/")
async def root():
    return {
        "Hi": "This app predicts wether income exceeds $50K/yr based on census data."}

# Prediction path [a path that cuts to get a salary prediction this is to be used in URL of application]
@app.post("/predict-income")
async def predict(input: ModelInput):

    input_data = np.array([[
        input.age,
        input.workclass,
        input.fnlgt,
        input.education,
        input.education_num,
        input.marital_status,
        input.occupation,
        input.relationship,
        input.race,
        input.sex,
        input.capital_gain,
        input.capital_loss,
        input.hours_per_week,
        input.native_country]])

    original_cols = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours-per-week",
        "native-country"]

    input_df = DataFrame(data=input_data, columns=original_cols)
    cat_features = get_cat_features()

    X, _, _, _ = process_data(
        input_df, categorical_features=cat_features, encoder=encoder, lb=lb, training=False)
    y = inference(model, X)
    pred = lb.inverse_transform(y)[0]

    return {"Income prediction": pred}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8004, reload=True)
