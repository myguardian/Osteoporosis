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

nominal_col_gender = ['PatientGender']

# 16 Columns are nominal and none bone related
nominal_col = [
               'parentbreak',
               'arthritis',
               'cancer',
               'ptunsteady',
               'whereliv',
               'education',
               'diabetes',
               'heartdisease',
               'respdisease',
               'alcohol',
               'howbreak',
               'wasfractdue2fall',
               'ptfall',
               'fxworried',
               'notworking',
               'marital']

# 9 Columns are bone related and nominal
nominal_col_bone = ['hip',
                    'ankle',
                    'clavicle',
                    'elbow',
                    'femur',
                    'spine',
                    'wrist',
                    'shoulder',
                    'tibfib', ]

# No columns in the data are ordinal
ordinal_col = []


# __________________________________________________________

# Converting Values into Metric
def height_to_metric(idx, height, measure):
    try:
        if measure == 2:
            return height * 2.54
        elif measure == 1:
            return height
        else:
            logging.error(f'Patient Id = {idx} does not have a Height Unit')
            return None
    except ValueError:
        logging.error(ValueError)
        logging.error(f'Unable to convert height to metric for patient id = {idx}')


# Converting Values into Metric
def weight_to_metric(idx, weight, measure):
    try:
        if measure == 2:
            return weight * 0.45359237
        elif measure == 1:
            return weight
        else:
            logging.error(f'Patient Id = {idx} does not have a Weight Unit')
            return None
    except ValueError:
        logging.error(ValueError)
        logging.error(f'Unable to convert weight to metric for patient id = {idx}')


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

    except ValueError:
        logging.error(ValueError)


# A more general dropping algorithm. This version of dropping is not used.
def remove_duplicates_with_no_id():
    df.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)


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
    except ValueError:
        logging.error(ValueError)


# __________________________________________________________

# Dealing with missing data
def remove_all_rows_with_null_columns():
    try:
        df.dropna(axis=0, how='all', inplace=True)
    except ValueError:
        logging.error(ValueError)


# We are going to use the mean to fill in the values
def fill_numerical_with_mean():
    for column in numerical_col:
        try:
            df[column].fillna(df[column].mean(), inplace=True)
        except ValueError:
            logging.error(ValueError)


def fill_numerical_with_median():
    for column in numerical_col:
        try:
            df[column].fillna(df[column].median(), inplace=True)
        except ValueError:
            logging.error(ValueError)


# For none bone data
def fill_nominal_with_mode():
    for column in nominal_col:
        try:
            df[column].fillna(df[column].mode()[0], inplace=True)
        except ValueError:
            logging.error(ValueError)


# Fill gender with the mode
def fill_nominal_gender_with_mode():
    for column in nominal_col_gender:
        try:
            df[column].fillna(df[column].mode()[0], inplace=True)
        except ValueError:
            logging.error(ValueError)


# We will use this function to fill values that are categorical
def fill_nominal_with_zero():
    for column in nominal_col:
        try:
            df[column].fillna(0, inplace=True)
        except ValueError:
            logging.error(ValueError)


# For bone data CALL THIS FUNCTION AT THE END of filling all the other columns
def fill_nominal_bone_with_zero():
    for column in nominal_col_bone:
        try:
            df[column].fillna(0, inplace=True)
        except ValueError:
            logging.error(ValueError)


if __name__ == "__main__":

    # Loading the data
    try:

        file_name = sys.argv[1]
        logging.info(f'Loading Data {file_name}\n')
        df = pd.read_csv(file_name)

    except ValueError:
        logging.error(ValueError)
        quit()

    # Converting values into metric
    try:
        logging.info('Converting Height to Metric\n')
        df['bmdtest_height'] = [
            height_to_metric(df.loc[idx, 'PatientId'], df.loc[idx, 'bmdtest_height'],
                             df.loc[idx, 'bmdtest_height_units'])
            for idx
            in range(len(df))]

        logging.info('Converting Weight to Metric\n')
        df['bmdtest_weight'] = [
            weight_to_metric(df.loc[idx, 'PatientId'], df.loc[idx, 'bmdtest_weight'],
                             df.loc[idx, 'bmdtest_weight_units'])
            for idx
            in range(len(df))]

    except ValueError:
        logging.error(ValueError)
        quit()

    # Dealing with Duplicates
    try:
        logging.info("Counting and removing duplicates.\n")
        count_duplicates()
        remove_duplicates_with_id()

    except ValueError:
        logging.error(ValueError)
        quit()

    # Selecting all features required for the model building process
    try:
        logging.info("Selecting features from Data\n")
        all_arrays = np.concatenate((numerical_col, nominal_col_gender, nominal_col, nominal_col_bone, ordinal_col))
        df = df[all_arrays]

    except ValueError:
        logging.error(ValueError)
        quit()

    # Imputing values into missing cells
    try:
        logging.info("Imputing Data into missing Columns\n")

        fill_numerical_with_mean()
        fill_nominal_gender_with_mode()
        fill_nominal_with_mode()
        fill_nominal_bone_with_zero()

    except ValueError:
        logging.error(ValueError)
        quit()

    # Count how many missing cells
    try:
        logging.info(f"Count total NaN at each column in a DataFrame\n{df.isnull().sum()}")
    except ValueError:
        logging.error(ValueError)

    # Saving data to the CSV File
    try:
        logging.info('Saving Data to CSV file\n')

        path = Path("Clean_Data_Main.csv")
        df.replace(r'\s+', np.nan, regex=True)
        df.to_csv(path, index=False)

        logging.info(f'Data saved to {path}\n')

    except ValueError:
        logging.error(ValueError)
        quit()
