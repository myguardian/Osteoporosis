import logging
import azure.functions as func
import fastapi
from pathlib import Path
from .schema import PatientInfo
from .http_asgi import AsgiMiddleware
from fastapi import File, UploadFile
import pandas as pd
import numpy as np
import pickle
import joblib
import sklearn

# general settings for FastAPI
app = fastapi.FastAPI(
    title="Testing",
    docs_url="/docs",
    redoc_url="/redoc",
)

BASE_DIR = Path(__file__).resolve(strict=True).parent

# feature columns for dataset
feature_columns = ['PatientId', 'PatientAge', "PatientGender", 'bmdtest_weight', 'bmdtest_height',
                   'parentbreak_1.0', 'alcohol_1.0', 'obreak_1.0', 'obreak_2.0', 'arthritis_1.0',
                   'diabetes_1.0','oralster_1.0', 'oralster_2.0', 'smoke_1.0']


# encode categorical data
def encode_cat_data(data):
    cat_features = ['parentbreak', 'alcohol',
                    'arthritis', 'diabetes',
                    'oralster', 'smoke', 'obreak',]
    dataset = data.copy()

    # one hot encoding process
    for feature in cat_features:
        cat_one_hot = pd.get_dummies(dataset[feature], prefix=f'{feature}', drop_first=False)
        dataset = dataset.drop(feature, axis=1)
        dataset = dataset.join(cat_one_hot)

    return dataset

# scale numerical data
def scale_data(data, scaler):
    # columns iwth numerical values
    cols_to_scale = ['PatientAge', 'bmdtest_weight', 'bmdtest_height']

    # no longer needed as we're using the scaler that was fitted from the remote data
    # # fits to the scaler that was handed over
    # scaler.fit(data[cols_to_scale])

    # transforms the columns into scale
    data[cols_to_scale] = scaler.transform(data[cols_to_scale])
    return data

# simple testing function
@app.get("/user/{user_id}")
async def get_user(user_id: int):
    return {
        "user_id": user_id,
    }


# Comment out if running on Windows
# @app.post("/risk_level/model/inference")
# def risk_level_prediction_inference(file: UploadFile = File(...)):

#     # loading model and scaler that were saved from remot data run
#     model = joblib.load(f"{BASE_DIR}/FRAX_random_forest_model.sav")
#     scaler = joblib.load(f"{BASE_DIR}/scaler_joblib.sav")

#     # read file
#     csv_data = pd.read_csv(file.file)

#     # encode categorical data
#     data = encode_cat_data(csv_data)

#     # keep only feature columns
#     for f in feature_columns:
#         if f not in data:
#             data[f] = 0
#     data = data[feature_columns]

#     # scale numerical data
#     data = scale_data(data, scaler)

#     # drop patient id
#     data.drop('PatientId', axis=1, inplace=True)

#     # calculate risk level
#     risk_level_prediction = model.predict(data)

#     # make dict of risk_level_prediction for payload
#     risk_level_dict = {}
#     for index, i in enumerate(risk_level_prediction):
#         risk_level_dict[index] = f"{i}"

#     # putting risk level in a dict format for the payload
#     payload = dict(risk=risk_level_dict)

#     return payload


def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return AsgiMiddleware(app).handle(req, context)

