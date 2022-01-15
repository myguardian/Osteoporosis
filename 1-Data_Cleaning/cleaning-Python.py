import logging

import pandas as pd
import numpy as np
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO)

# Create Data frame Variable
df = []

# Load the different variables we want to use

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


# __________________________________________________________


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

        else:
            # Convert data from Lbs to KGs
            weightLb = weight_value
            if weightLb is not None:
                weightKg = weightLb * 0.45359237

        return heightCm, weightKg

    except ValueError as err:
        logging.error(err)
        logging.error(f'Unable to convert height to metric for patient id = {idx}')


def fill_zeros_in_height_weight_with_mean():
    df['bmdtest_height'].replace(0, df['bmdtest_height'].mean(), inplace=True)
    df['bmdtest_weight'].replace(0, df['bmdtest_weight'].mean(), inplace=True)


# __________________________________________________________

# Understanding duplicates in the data 
def count_duplicates():
    try:
        # we know that column 'id' is unique, but what if we drop it?
        df_dedupped = df.drop('PatientId', axis=1).drop_duplicates()

        # there were duplicate rows
        logging.info(df.shape)
        logging.info(df_dedupped.shape)

        logging.info(f'{df.shape[0] - df_dedupped.shape[0]} rows were duplicates')
        logging.info(f'{df.shape[1] - df_dedupped.shape[1]} feature(s) are duplicates\n')

    except ValueError as er:
        logging.error(er)


# Remove the duplicates using the Patient ID and Baseline ID. ID's are unique, meaning we shouldn't have duplicates
def remove_duplicates_with_id():
    try:
        df.drop_duplicates(subset=['BaselineId'], inplace=True)
        df.drop_duplicates(subset=['PatientId'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Remove the ID's
        df.drop(['BaselineId'], axis=1)
        df.drop(['PatientId'], axis=1)
        df.reset_index(drop=True, inplace=True)
    except ValueError as e:
        logging.error(e)


# __________________________________________________________

# Dealing with missing data
def remove_all_rows_with_null_columns():
    try:
        df.dropna(axis=0, how='all', inplace=True)
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

    except ValueError as e:
        logging.error(e)
        quit()

    # ----------------------------------------------------------------------
    # Dealing with Duplicates
    try:
        logging.info("Counting and removing duplicates.\n")
        count_duplicates()
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
    # Selecting all features required for the model building process
    try:
        logging.info("Selecting features from Data\n")
        all_arrays = np.concatenate((numerical_col, nominal_col, special_nominal))
        df = df[all_arrays]

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
    # Count how many missing cells
    try:
        logging.info(f"Count total NaN at each column in a DataFrame\n{df.isnull().sum()}")
    except ValueError:
        logging.error(ValueError)

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
