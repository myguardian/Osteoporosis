import logging

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import sweetviz
import os
import sys

logging.basicConfig(level=logging.INFO)

# Create Data frame Variable
df = []

# Load the different variables we want to use

patient_id_col = ['PatientId']

# 4 Columns are numerical
numerical_col = ['PatientAge',
                 'bmdtest_height',
                 'bmdtest_weight',
                 'bmdtest_tscore_fn']

# We will fill null cells with mode
nominal_col = [
    'PatientGender',
    'parentbreak',
    'ptunsteady',
    'whereliv',
    'education',
    'alcohol',
    'wasfractdue2fall',
    'ptfall',
    'fxworried',
    'notworking',
    'marital']

# We will fill null cells with 0 for these columns
special_nominal = ['arthritis',
                   'cancer',
                   'diabetes',
                   'heartdisease',
                   'respdisease',
                   'howbreak',
                   'hip',
                   'ankle',
                   'clavicle',
                   'elbow',
                   'femur',
                   'spine',
                   'wrist',
                   'shoulder',
                   'tibfib', ]


def set_directory():
    # detect the current working directory and add the sub directory
    main_path = os.getcwd()
    absolute_path = main_path + "/pre_analysis_results"
    try:
        os.mkdir(absolute_path)
    except OSError:
        logging.info("Creation of the directory %s failed. Folder already exists." % absolute_path)
    else:
        logging.info("Successfully created the directory %s " % absolute_path)


# __________________________________________________________

# function to covert lbs to kg
def lbs_to_kg(weight_value):
    weightLb = weight_value
    if weightLb is not None:
        weightKg = weightLb * 0.45359237
        return weightKg


# Converting Values into Metric
def data_to_metric(idx, height_value, weight_value):
    try:

        # Create variables to hold the values
        heightCm = 0
        weightKg = 0
        metric = True

        if 1 < height_value < 2.2:
            # Convert METERS to CM
            heightCm = height_value * 100

        elif 50 < height_value < 84:
            # Height value is too high to be METERS or FEET and too low to be CM.
            # Assume the height is INCHES and convert to CM
            heightIn = height_value
            heightCm = heightIn * 2.54
            metric = False  # we use this flag later with weight...

        elif height_value > 125:
            # The height is probably in CM
            heightCm = height_value
            metric = True

        if metric:
            # The assumption is that if units are missing, they still used the same units type (metric vs imperial)
            # It's impossible to do the same inference as with height because the "reasonable value" ranges are less
            # distinct.
            weightKg = weight_value

            # check if the weight thats assumed to be kg, is actually lbs (ex 95 kg or 95 lbs)

            # add a buffer to each weight to account for overweight individuals
            buffer = 30

            # for each weight, check if its greater than the BMI normal weight range, for the appopriate height interval
            if height_value <= 152.4 and weightKg >= 56.7 + buffer:
                # less than 5ft 0
                weightKg = lbs_to_kg(weightKg)

            elif 152.4 < height_value <= 154.9 and weightKg >= 56.7 + buffer:
                # 5ft0 to 5ft1
                weightKg = lbs_to_kg(weightKg)

            elif 152.4 < height_value <= 157.5 and weightKg >= 59 + buffer:
                # 5ft1 to 5ft2
                weightKg = lbs_to_kg(weightKg)

            elif 157.5 < height_value <= 160 and weightKg >= 61.2 + buffer:
                # 5ft2 < 5ft3
                weightKg = lbs_to_kg(weightKg)

            elif 160 < height_value <= 162.6 and weightKg >= 63.5 + buffer:
                # 5ft3 < 5ft4
                weightKg = lbs_to_kg(weightKg)

            elif 162.6 < height_value <= 165.1 and weightKg >= 65.8 + buffer:
                # 5ft4 < 5ft5
                weightKg = lbs_to_kg(weightKg)

            elif 165.1 < height_value <= 167.6 and weightKg >= 68 + buffer:
                # 5ft5 < 5ft6
                weightKg = lbs_to_kg(weightKg)

            elif 167.6 < height_value <= 170.2 and weightKg >= 70.3 + buffer:
                # 5ft6 < 5ft7
                weightKg = lbs_to_kg(weightKg)

            elif 170.2 < height_value <= 172.7 and weightKg >= 72.6 + buffer:
                # 5ft7 < 5ft8
                weightKg = lbs_to_kg(weightKg)

            elif 172.7 < height_value <= 175.3 and weightKg >= 74.8 + buffer:
                # 5ft8 < 5ft9
                weightKg = lbs_to_kg(weightKg)

            elif 175.3 < height_value <= 177.8 and weightKg >= 77.1 + buffer:
                # 5ft9 < 5ft10
                weightKg = lbs_to_kg(weightKg)

            elif 177.8 < height_value <= 180.3 and weightKg >= 79.4 + buffer:
                # 5ft10 < 5ft11
                weightKg = lbs_to_kg(weightKg)

            elif 180.3 < height_value <= 182.9 and weightKg >= 81.6 + buffer:
                # 5ft11 < 6ft0
                weightKg = lbs_to_kg(weightKg)

            elif 182.9 < height_value <= 185.4 and weightKg >= 83.9 + buffer:
                # 6ft0 < 6ft1
                weightKg = lbs_to_kg(weightKg)

            elif 185.4 < height_value <= 188 and weightKg >= 86.2 + buffer:
                # 6ft1 < 6ft2
                weightKg = lbs_to_kg(weightKg)

            elif 188 < height_value <= 190.5 and weightKg >= 88.5 + buffer:
                # 6ft2 < 6ft3
                weightKg = lbs_to_kg(weightKg)

            elif 190.5 < height_value and weightKg >= 90.7 + buffer:
                # max range
                # 6ft3 < 6ft4
                weightKg = lbs_to_kg(weightKg)

        else:
            # data is imperial
            # Convert data from Lbs to KGs
            weightKg = lbs_to_kg(weight_value)

        return heightCm, weightKg

    except ValueError as err:
        logging.error(err)
        logging.error(f'Unable to convert height to metric for patient id = {idx}')


def fill_zeros_in_height_weight_with_mean():
    df['bmdtest_height'].replace(0, df['bmdtest_height'].mean(), inplace=True)
    df['bmdtest_weight'].replace(0, df['bmdtest_weight'].mean(), inplace=True)


# __________________________________________________________


# Remove the duplicates using the Patient ID and Baseline ID. ID's are unique, meaning we shouldn't have duplicates
def remove_duplicates_with_id():
    try:
        df.drop_duplicates(subset=['PatientId'], inplace=True)
        df.reset_index(drop=True, inplace=True)

    except ValueError as er:
        logging.error(er)


# We are going to use the mean to fill in the values
def fill_numerical_with_mean():
    for column in numerical_col:
        try:
            df[column].fillna(df[column].mean(), inplace=True)
        except ValueError as er:
            logging.error(er)


# Fill non-special columns with mode
def fill_nominal_with_mode():
    for column in nominal_col:
        try:
            df[column].fillna(df[column].mode()[0], inplace=True)
        except ValueError as er:
            logging.error(er)


# Fill special columns with 0
def fill_special_nominal_with_zero():
    for column in special_nominal:
        try:
            df[column].fillna(0, inplace=True)
        except ValueError as er:
            logging.error(er)


if __name__ == "__main__":

    # Loading the data
    try:
        file_name = sys.argv[1]
        logging.info(f'Loading Data {file_name}\n')
        df = pd.read_csv(file_name)

        # Create the directory where the CSV files and images are going to be saved
        set_directory()

    except ValueError as e:
        logging.error(e)
        quit()

    # ----------------------------------------------------------------------
    # Selecting all features required for the model building process
    try:
        logging.info("Selecting features from Data\n")
        all_features = np.concatenate((patient_id_col, numerical_col, nominal_col, special_nominal))
        df = df[all_features]

    except ValueError as e:
        logging.error(e)
        quit()

    try:
        logging.info(f'Creating feature data frame html graph')
        analysis = sweetviz.analyze(df)
        analysis.show_html('pre_analysis_results/pre_analysis.html', open_browser=False)

    except ValueError as e:
        logging.error(e)
        logging.error('Error showing feature report')

    # ----------------------------------------------------------------------
    # Dealing with Duplicates
    try:
        logging.info("Removing duplicates.\n")
        remove_duplicates_with_id()
    except ValueError as e:
        logging.error(e)
        quit()

    # ----------------------------------------------------------------------
    # Converting values into metric
    try:

        logging.info('Converting Height and Weight to Metric\n')
        converted_data_tuple = [
            data_to_metric(df.loc[idx, 'PatientId'], df.loc[idx, 'bmdtest_height'], df.loc[idx, 'bmdtest_weight'])
            for idx
            in range(len(df))]

        # Get heights from tuple
        df['bmdtest_height'] = [x[0] for x in converted_data_tuple]

        # Get weights from tuple
        df['bmdtest_weight'] = [x[1] for x in converted_data_tuple]

        fill_zeros_in_height_weight_with_mean()
    except ValueError as e:
        logging.error(e)
        quit()

    # ----------------------------------------------------------------------
    # Imputing values into missing cells
    try:
        logging.info("Imputing Data into missing Columns\n")

        fill_numerical_with_mean()
        fill_nominal_with_mode()
        fill_special_nominal_with_zero()

    except ValueError as e:
        logging.error(e)
        quit()

    # ----------------------------------------------------------------------
    # Saving data to the CSV File
    try:
        logging.info('Saving Data to CSV file\n')

        path = Path("Clean_Data_Main.csv")
        df.replace(r'\s+', np.nan, regex=True)
        df.to_csv(path, index=False)

        logging.info(f'Data saved to {path}\n')

    except ValueError as e:
        logging.error(e)
        quit()
