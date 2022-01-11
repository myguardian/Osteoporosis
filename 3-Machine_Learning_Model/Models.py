import logging
import sys

import numpy as np
import statistics
from collections import Counter
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
# import the os module
import os
from pycaret.regression import *


def set_directory():
    # detect the current working directory and add the sub directory
    main_path = os.getcwd()
    absolute_path = main_path + "/models_results"
    try:
        os.mkdir(absolute_path)
    except OSError:
        logging.info("Creation of the directory %s failed. Folder already exists." % absolute_path)
    else:
        logging.info("Successfully created the directory %s " % absolute_path)


def setup_data(path):
    dataset = pd.read_csv(path)

    missing = pd.DataFrame(dataset.isnull().sum(), columns=['Total'])
    missing['%'] = (missing['Total'] / dataset.shape[0]) * 100
    missing.sort_values(by='%', ascending=False)

    size = dataset.shape[0]
    dataset = dataset.dropna()

    print('Number of rows in the dataset after the rows with missing values were removed: {}.\n{} rows were removed.'
          .format(dataset.shape[0], size - dataset.shape[0]))

    cols = ['bmdtest_height', 'bmdtest_weight']

    for c in cols:
        upper_level = dataset[c].mean() + 3 * dataset[c].std()
        lower_level = dataset[c].mean() - 3 * dataset[c].std()
        dataset = dataset[(dataset[c] > lower_level) & (dataset[c] < upper_level)]

    print('Number of rows in the dataset after the rows with missing values were removed: {}.\n{} rows were removed.'
          .format(dataset.shape[0], size - dataset.shape[0]))

    data = dataset.sample(frac=0.9, random_state=786)
    data_unseen = dataset.drop(data.index)

    data.reset_index(drop=True, inplace=True)
    data_unseen.reset_index(drop=True, inplace=True)

    print('Data for Modeling: ' + str(data.shape))
    print('Unseen Data For Predictions: ' + str(data_unseen.shape))

    return dataset


if __name__ == "__main__":
    try:
        # Get the data from the argument
        # file_name = sys.argv[1]
        file_name = '../Data Cleaning/Clean_Data_Main.csv'
        logging.info(f'Loading Data {file_name}\n')

        # Create the directory where the CSV files and images are going to be saved
        set_directory()

        # Perform the analysis and generate the images
        main_data = setup_data(file_name)

        if main_data is not None:
            exp_name = setup(data=main_data, target='bmdtest_tscore_fn', silent=True, session_id=123)
            best_model = compare_models(exclude=['ransac'], n_select=10)

        else:
            logging.error('No data exists.')

    except ValueError as e:
        logging.error(e)
        logging.error('Unable to load the CSV File')
