import sweetviz
import logging
import pandas as pd
import numpy as np
import sys
from pathlib import Path
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
                  # 'bmdtest_tscore_fn'
                  ]

# We will fill null cells with mode
nominal_cols = [
    'PatientGender',
    'parentbreak',
    'alcohol',
    'oralster',
    'smoke',
]

# We will fill null cells with 0
special_nominal_cols = ['arthritis',
                        'diabetes',
                        'obreak'
                        ]

target_cols = ['FraxRiskLevel']


def set_directory(temp_path):
    # detect the current working directory and add the sub directory
    main_path = os.getcwd()
    absolute_path = main_path + temp_path
    try:
        os.mkdir(absolute_path)
    except OSError:
        logging.error("Creation of the directory %s failed. Folder already exists." % absolute_path)
    else:
        logging.info("Successfully created the directory %s " % absolute_path)


# Remove the duplicates using the Patient ID and Baseline ID. ID's are unique, meaning we shouldn't have duplicates
def remove_duplicates_with_id():
    try:
        df.drop_duplicates(subset=['PatientId'], inplace=True)
        df.reset_index(drop=True, inplace=True)

    except ValueError as er:
        logging.error(str(er))


# function to covert lbs to kg
def lbs_to_kg(weight_value):
    weightLb = weight_value
    if weightLb is not None:
        weightKg = weightLb * 0.45359237
        return weightKg


# Include max buffer
def bmi_with_buff(height_value, weightKg):
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

            # Lets convert the weight via buffers
            doBmi = False
            if doBmi:
                weightKg = bmi_with_buff(height_value, weightKg)
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
            # mean[1] for female avg, mean[2] for male avg
            mean = df.groupby(['PatientGender'])[column].mean()
            df.loc[(df.PatientGender == 1) & (df[column].isnull()), column] = mean[1]
            df.loc[(df.PatientGender == 1) & (df[column] == 0), column] = mean[1]
            df.loc[(df.PatientGender == 2) & (df[column].isnull()), column] = mean[2]
            df.loc[(df.PatientGender == 2) & (df[column] == 0), column] = mean[2]
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


# def fill_target_cols():
#     for column in target_cols:
#         for idx, age in enumerate(df['PatientAge']):
#             high_bmd = 0
#             low_bmd = 0
#             if pd.isna(df[column][idx]):
#                 # for women
#                 if df['PatientGender'][idx] == 1:
#                     if age < 60:
#                         high_bmd = -2.5
#                         low_bmd = -3.8
#                     elif age < 65:
#                         high_bmd = -2.3
#                         low_bmd = -3.7
#                     elif age < 70:
#                         high_bmd = -1.9
#                         low_bmd = -3.5
#                     elif age < 75:
#                         high_bmd = -1.7
#                         low_bmd = -3.2
#                     elif age < 80:
#                         high_bmd = -1.2
#                         low_bmd = -2.9
#                     elif age < 85:
#                         high_bmd = -0.5
#                         low_bmd = -2.6
#                     else:
#                         high_bmd = 0.1
#                         low_bmd = -2.2
#                 # for men
#                 else:
#                     if age < 60:
#                         high_bmd = -2.5
#                         low_bmd = -3.9
#                     elif age < 65:
#                         high_bmd = -2.5
#                         low_bmd = -3.7
#                     elif age < 70:
#                         high_bmd = -2.4
#                         low_bmd = -3.7
#                     elif age < 75:
#                         high_bmd = -2.3
#                         low_bmd = -3.7
#                     elif age < 80:
#                         high_bmd = -2.3
#                         low_bmd = -3.8
#                     elif age < 85:
#                         high_bmd = -2.1
#                         low_bmd = - 3.8
#                     else:
#                         high_bmd = -2.0
#                         low_bmd = -3.8
#
#                 if df['bmdtest_tscore_fn'][idx] > high_bmd:
#                     df[column][idx] = 0
#                 elif df['bmdtest_tscore_fn'][idx] > low_bmd:
#                     df[column][idx] = 1
#                 elif df['bmdtest_tscore_fn'][idx] <= low_bmd:
#                     df[column][idx] = 2
#
#                 if (df['oralster'][idx] == 1 or df['obreak'][idx] > 1 or \
#                         df['ankle'][idx] == 1 or df['clavicle'][idx] == 1 or \
#                         df['elbow'][idx] == 1 or df['femur'][idx] == 1 or \
#                         df['wrist'][idx] == 1 or df['shoulder'][idx] == 1 or \
#                         df['tibfib'][idx] == 1) and df[column][idx] < 2:
#                     df[column][idx] += 1
#
#                 if df['hip'][idx] == 1 or df['spine'][idx] == 1:
#                     df[column][idx] = 2

# Lets create a report using sweetviz
def create_html_report(data, save_path):
    try:
        logging.info(f'Creating sweetviz graph for {save_path}')
        temp_analysis = sweetviz.analyze(data)
        temp_analysis.show_html(save_path, open_browser=False)

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
        logging.info(f'Setting Directories\n')
        set_directory('')
        set_directory('../FRAX_V3/pre_cleaning_results')
        set_directory('../FRAX_V3/analysis_results')
    except ValueError as e:
        logging.error(str(e))
        quit()

    try:
        logging.info("Selecting features from Data\n")
        all_columns = np.concatenate((patient_id_col, numerical_cols, nominal_cols, special_nominal_cols, target_cols))
        df = df[all_columns]

    except ValueError as e:
        logging.error(str(e))
        quit()

    try:
        logging.info(f'Performing analysis on the unclean data\n')
        create_html_report(df, '../FRAX_V3/pre_cleaning_results/pre_analysis.html')

    except ValueError as e:
        logging.error(str(e))

    try:
        logging.info('Performing analysis on the unclean female and male data\n')
        female_data = df[df['PatientGender'] == 1]
        create_html_report(female_data, '../FRAX_V3/pre_cleaning_results/pre_analysis_female.html')

        male_data = df[df['PatientGender'] == 2]
        create_html_report(male_data, '../FRAX_V3/pre_cleaning_results/pre_analysis_male.html')

    except ValueError as e:
        logging.error(str(e))
        quit()

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

    # try:
    #     # Fills in the target column bmdtest_10yr_caroc
    #     logging.info('Filling in target column\n')
    #     pd.options.mode.chained_assignment = None
    #     fill_target_cols()
    #     # Drops any rows with possible misinput/miscalculation
    #     df = df[df.bmdtest_10yr_caroc != 0]
    # except ValueError as e:
    #     logging.error(str(e))
    #     quit()

    try:
        logging.info('Saving Data to CSV file\n')

        path = Path("../FRAX_V3/FRAX_V3.csv")
        df.replace(r'\s+', np.nan, regex=True)
        df.to_csv(path, index=False)

        logging.info(f'Data saved to {path}\n')

    except ValueError as e:
        logging.error(str(e))
        quit()

    try:
        logging.info('Performing Analysis on all the Data\n')

        create_html_report(df, '../FRAX_V3/analysis_results/analysis.html')
    except ValueError as e:
        logging.error(str(e))
        quit()

    try:
        logging.info('Performing Analysis on the female and male data\n')

        female_data = df[df['PatientGender'] == 1]
        create_html_report(female_data, '../FRAX_V3/analysis_results/analysis_female.html')

        male_data = df[df['PatientGender'] == 2]
        create_html_report(male_data, '../FRAX_V3/analysis_results/analysis_male.html')

    except ValueError as e:
        logging.error(str(e))
        quit()
