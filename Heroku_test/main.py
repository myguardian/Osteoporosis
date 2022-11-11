from fastapi import FastAPI, File, UploadFile
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

app = FastAPI()

encode = ['sex', 'island']

BASE_DIR = Path(__file__).resolve(strict=True).parent

@app.get("/")
async def root():
    return {"message" : "Hello from FastAPI"}

@app.post("/predict")
def predict(file: UploadFile = File(...)):

    input_df = pd.read_csv(file.file)

    penguins_raw = pd.read_csv(f"{BASE_DIR}/penguins_cleaned.csv")
    penguins = penguins_raw.drop(columns=['species'])
    df = pd.concat([input_df, penguins], axis=0)

    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy], axis=1)
        del df[col]
    df = df[:1]

    model = pickle.load(open(f"{BASE_DIR}/penguins_clf.pkl", 'rb'))
    pred = model.predict(df)

    penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
    payload = penguins_species[pred[0]]
    return {"prediction" : payload}