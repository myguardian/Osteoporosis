import joblib
import pandas as pd
import numpy as np
from core import config
from fastapi import APIRouter, File, UploadFile
from models.schema import PatientInfo
import csv
import codecs
import pyarrow.parquet as pq

router = APIRouter()

# import paths of model and scaler which are set in backend/app/core/config.py
MODEL_PATH = config.RFC_PATH
SCALER_PATH = config.STANDARD_SC_PATH

# loading the model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# feature columns for dataset
feature_columns = ['PatientId', 'PatientAge', "PatientGender", 'bmdtest_weight', 'bmdtest_height',
                   'parentbreak_1.0', 'alcohol_1.0', 'obreak_1.0', 'obreak_2.0', 'arthritis_1.0',
                   'diabetes_1.0','oralster_1.0', 'oralster_2.0', 'smoke_1.0']


# encode categorical data
def encode_cat_data(data):
    cat_features = ['parentbreak', 'alcohol',
                    'arthritis', 'diabetes',
                    'oralster', 'smoke', 'obreak',
                    # 'respdisease', 'hbp','heartdisease',
                    # 'ptunsteady', 'wasfractdue2fall', 'cholesterol',
                    # 'ptfall', 'shoulder', 'wrist', 'bmdtest_10yr_caroc'
                    ]
    dataset = data.copy()

    for feature in cat_features:
        cat_one_hot = pd.get_dummies(dataset[feature], prefix=f'{feature}', drop_first=False)
        dataset = dataset.drop(feature, axis=1)
        dataset = dataset.join(cat_one_hot)

    return dataset


# scale numerical data
def scale_data(data):
    cols_to_scale = ['PatientAge', 'bmdtest_weight', 'bmdtest_height']
    scaler.fit(data[cols_to_scale])
    data[cols_to_scale] = scaler.transform(data[cols_to_scale])
    return data

# will be used for validation purposes
# validations will be done in batches
@router.post("/risk_level/model/batch_inference")
def risk_level_batch_inference(file: UploadFile = File(...)):

    # read file
    csv_data = pd.read_csv(file.file)

    # encode categorical data
    data = encode_cat_data(csv_data)

    # keep only feature columns
    data = data[feature_columns]

    # scale numerical data
    data = scale_data(data)

    # drop patient id
    data.drop('PatientId', axis=1, inplace=True)

    # calculate risk level
    risk_level_prediction = model.predict(data)

    # make dict of risk_level_prediction for payload
    risk_level_dict = {}
    for index, i in enumerate(risk_level_prediction):
        risk_level_dict[index] = f"{i}"

    payload = dict(risk=risk_level_dict)

    return payload


# this will get single patient info and return single risk level result
@router.post("/risk_level/model/single_inference")
def risk_level_single_inference(patientInfo : PatientInfo):

    data = pd.DataFrame(patientInfo)
    data_array = np.array(data)
    n_features = len(data)
    data_array = data_array.reshape(-1, n_features)

    # just like batch processing,
    # we need to encode categorical data
    # filter out the columns we don't need
    # and use the scaler to normalize
    data_array = encode_cat_data(data)
    data_array = data[feature_columns]
    data_array = scale_data(data)

    # PatientId is not in the PatientInfo class so there is no need to drop it
    # data.drop('PatientId', axis=1, inplace=True)

    # calculate risk level
    risk_level_prediction = model.predict(data_array)

    payload = dict(risk=risk_level_prediction[0])

    return payload
