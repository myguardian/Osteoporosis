import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import sys
import pycaret
import datetime
import logging
from pycaret.classification import *


def setup_data(path):
    dataset = pd.read_csv(path)
    return dataset


if __name__ == "__main__":

    main_data = None 

    try:
        # Get the data from the argument
        file_name = sys.argv[1]
        # file_name = '../Clean_Data_Main.csv'
        logging.info('Loading Data {file_name} \n')

        # Perform the analysis and generate the images
        main_data = setup_data(file_name)
        
    except ValueError as e:
        logging.error(e)
        logging.error('Unable to load the CSV File')

    if main_data is not None:
        dataset = main_data
        obreak_date = pd.to_datetime(dataset.obreak_date)
        datebone = pd.to_datetime(dataset.datebone)
        y = ( abs( datebone - obreak_date))
        dataset["days"] = y.dt.days
        dataset['second_IFR'] = np.where(dataset['days'] <= 730, True, False)
        dataset = dataset.drop('days', axis=1)
        dataset = dataset.drop('DateSurveyed', axis=1)
        dataset = dataset.drop('obreak_date', axis=1)
        dataset = dataset.drop('PostalCode', axis=1)
        data = dataset[["PatientAge","PatientGender","obreak_frac_count","whereliv", "ptunsteady","oralster", "obreak_hip","parentbreak","alcohol","ptfall","wtcurr_lbs", "smoke", "marital", 'arthritis', 'diabetes', "education","second_IFR"]] 
        data["obreak_frac_count"].fillna(0, inplace=True)
        exp_clf101 = setup(data = data, target = 'second_IFR', session_id=2, fix_imbalance = True, remove_multicollinearity = True, categorical_features = ["PatientGender","obreak_frac_count","whereliv", "ptunsteady","oralster", "obreak_hip","parentbreak","alcohol","ptfall", "smoke", "marital", 'arthritis', 'diabetes', "education"], numeric_features=["PatientAge", "wtcurr_lbs"]) 
        best = compare_models()
        best_results = pull()
        b = open('best_results.txt', 'w+')
        b.write(best_results.to_string())
        b.close()
        tuned_results = tune_model(best)
        tune_results = pull()
        g = open('tune_results.txt', 'w+')
        g.write(tune_results.to_string())
        g.close()
