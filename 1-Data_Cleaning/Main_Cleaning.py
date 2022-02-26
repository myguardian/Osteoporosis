import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import sys

# Use logging commands instead of print
logging.basicConfig(level=logging.INFO)

# Create the global variables
df = []

# Used when removing duplicate rows
patient_id_col = ['PatientId']

# We will fill null cells with mean
numerical_cols = ['PatientAge',
                  'bmdtest_height',
                  'bmdtest_weight',
                  'bmdtest_tscore_fn']

# We will fill null cells with mode
nominal_cols = [
    'PatientGender',
    'parentbreak',
    'ptunsteady',
    'alcohol',
    'wasfractdue2fall',
    'ptfall',
    'oralster',
    'smoke'
    ]

# We will fill null cells with 0
special_nominal_cols = ['arthritis',
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
                        'tibfib'
                        ]


# Remove the duplicates using the Patient ID and Baseline ID. ID's are unique, meaning we shouldn't have duplicates
def remove_duplicates_with_id():
    try:
        df.drop_duplicates(subset=['PatientId'], inplace=True)
        df.reset_index(drop=True, inplace=True)

    except ValueError as er:
        logging.error(str(er))


# Weight values are in kilograms
def lbs_to_kg(weight_value):
    if weight_value is not None:
        weightKg = weight_value * 0.45359237
        return weightKg


# Height and Weight values are in kilograms
def calculate_bmi(idx, height_value, weight_value):
    try:
        return weight_value / ((height_value / 100) ** 2)
    except ValueError as er:
        logging.error(str(er))
        logging.error(f'Unable to calculate BMI for patient id = {idx}')


# Converting Values into Metric
def data_to_metric(idx, height_value, weight_value):
    try:

        heightCm = 0
        weightKg = 0
        isMetric = True  # Flag is used when checking if the weight is metric

        # Lets convert the height
        if 1 < height_value < 2.2:
            heightCm = height_value * 100  # Convert METERS to CM

        elif 50 < height_value < 84:
            # Height value is too high to be METERS or FEET and too low to be CM.
            # Assume the height is INCHES and convert to CM
            heightCm = height_value * 2.54
            isMetric = False

        elif height_value > 125:
            # The height is probably in CM
            heightCm = height_value

        # Lets convert the weight
        if isMetric:
            weightKg = weight_value
        else:
            weightKg = lbs_to_kg(weight_value)

        return heightCm, weightKg

    except ValueError as er:
        logging.error(str(er))
        logging.error(f'Unable to convert height and weight to metric for patient id = {idx}')


def fill_bmi_with_mean():
    try:
        df['bmi'].replace(0, df['bmi'].mean(), inplace=True)
        df['bmi'].fillna((df['bmi'].mean()), inplace=True)
    except ValueError as er:
        logging.error(str(er))


def fill_numerical_with_mean():
    for column in numerical_cols:
        try:
            mean = df[column].mean()
            df[column].fillna(mean, inplace=True)
            df[column].replace(0, mean, inplace=True)
        except ValueError as er:
            logging.error(str(er))



def fill_nominal_with_mode():
    for column in nominal_cols:
        try:
            mode = df[column].mode()[0]
            df[column].fillna(mode, inplace=True)
        except ValueError as er:
            logging.error(str(er))


def fill_special_nominal_with_zero():
    for column in special_nominal_cols:
        try:
            df[column].fillna(0, inplace=True)
        except ValueError as er:
            logging.error(str(er))


if __name__ == "__main__":
    # Loading the data
    try:
        logging.info(f'Loading File\n')
        file_name = sys.argv[1]
        df = pd.read_csv(file_name)

    except ValueError as e:
        logging.error(str(e))
        quit()

    try:
        logging.info("Selecting features from Data\n")
        all_columns = np.concatenate((patient_id_col, numerical_cols, nominal_cols, special_nominal_cols))
        df = df[all_columns]

    except ValueError as e:
        logging.error(str(e))
        quit()

    try:
        logging.info(f'Creating feature data frame html graph\n')
        # analysis = sw.analyze(df)
        # analysis.show_html('pre_analysis_results/pre_analysis.html', open_browser=False)

    except ValueError as e:
        logging.error(str(e))
        logging.error('Error showing feature report\n')

    try:
        logging.info("Removing duplicates\n")
        remove_duplicates_with_id()
    except ValueError as e:
        logging.error(str(e))
        quit()

    try:

        logging.info('Converting Height and Weight to Metric\n')

        converted_data_tuple = [
            data_to_metric(df.loc[idx, 'PatientId'],
                           df.loc[idx, 'bmdtest_height'],
                           df.loc[idx, 'bmdtest_weight'])
            for idx
            in range(len(df))]

        # Get heights from tuple
        df['bmdtest_height'] = [x[0] for x in converted_data_tuple]

        # Get weights from tuple
        df['bmdtest_weight'] = [x[1] for x in converted_data_tuple]

    except ValueError as e:
        logging.error(str(e))
        quit()

    try:
        logging.info("Imputing Data into numerical Columns\n")
        fill_numerical_with_mean()
    except ValueError as e:
        logging.error(str(e))
        quit()

    try:
        logging.info("Imputing Data into nominal Columns\n")
        fill_nominal_with_mode()
    except ValueError as e:
        logging.error(str(e))
        quit()

    try:
        logging.info("Imputing Data into special nominal Columns\n")
        fill_special_nominal_with_zero()
    except ValueError as e:
        logging.error(str(e))
        quit()

    try:
        logging.info('Creating BMI column\n')

        df['bmi'] = 0
        df['bmi'] = [
            calculate_bmi(df.loc[idx, 'PatientId'],
                          df.loc[idx, 'bmdtest_height'],
                          df.loc[idx, 'bmdtest_weight'])
            for idx
            in range(len(df))]

        fill_bmi_with_mean()
    except ValueError as e:
        logging.error(str(e))
        quit()

    try:
        logging.info('Saving Data to CSV file\n')

        path = Path("Clean_Data_Main.csv")
        df.replace(r'\s+', np.nan, regex=True)
        df.to_csv(path, index=False)

        logging.info(f'Data saved to {path}\n')

    except ValueError as e:
        logging.error(str(e))
        quit()
