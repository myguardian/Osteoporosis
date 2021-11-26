import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

# Create Data frame Variable
df = []

# 4 Columns are numerical
numerical_col = ['PatientAge',
                 'bmdtest_height',
                 'bmdtest_weight',
                 'bmdtest_tscore_fn']

# 16 Columns are nominal and none bone related
nominal_col = ['parentbreak',
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

# Check if the data frame has any empty columns/ rows
def check_null():
    print("Count how many are empty:")
    print(df.isnull().count())
    print("\n------------------\n")

    print("Are any empty:")
    print(df.isna().any())
    print("\n------------------\n")


# __________________________________________________________

# Converting Values into Metric
def height_to_metric(height, measure):
    if measure == 2:
        return height * 2.54
    elif measure == 1:
        return height
    else:
        return None



def weight_to_metric(weight, measure):
    if measure == 2:
        return weight * 0.45359237
    elif measure == 1:
        return weight
    else:
        return None


# __________________________________________________________

# Understanding duplicates in the data 
def count_duplicates():
    # we know that column 'id' is unique, but what if we drop it?
    df_dedupped = df.drop('PatientId', axis=1).drop_duplicates()

    # there were duplicate rows
    print(df.shape)
    print(df_dedupped.shape)


def remove_duplicates_with_no_id():
    df.drop_duplicates(subset=None, keep='first', inplace=False, ignore_index=False)


def remove_duplicates_with_id():
    df.drop_duplicates(subset=['BaselineId'], inplace=True)
    df.drop_duplicates(subset=['PatientId'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Remove the ID's
    df.drop(['BaselineId'], axis=1)
    df.drop(['PatientId'], axis=1)
    df.reset_index(drop=True, inplace=True)


# __________________________________________________________

# Dealing with missing data

def remove_all_rows_with_null_columns():
    df.dropna(axis=0, how='all', inplace=True)


def fill_numerical_with_mean():
    for column in numerical_col:
        df[column].fillna(df[column].mean(), inplace=True)


def fill_numerical_with_median():
    for column in numerical_col:
        df[column].fillna(df[column].median(), inplace=True)


# For none bone data
def fill_nominal_with_mode():
    for column in nominal_col:
        df[column].fillna(df[column].mode()[0], inplace=True)


# Best to use this function
def fill_nominal_with_zero():
    for column in nominal_col:
        df[column].fillna(0, inplace=True)


# For bone data CALL THIS FUNCTION AT THE END of filling all the other columns
def fill_nominal_bone_with_zero():
    for column in nominal_col_bone:
        df[column].fillna(0, inplace=True)


if __name__ == "__main__":

    try:
        import sys

        file_name = sys.argv[1]  # prints python_script.py
        print("Loading Data.")
        print(file_name)

        df = pd.read_csv(file_name)
    except:
        print("Error. Could not read file.")
        quit()

    try:
        print("Converting Data to metric.")
        df['bmdtest_height'] = [
            height_to_metric(df.loc[idx, 'bmdtest_height'],
                             df.loc[idx, 'bmdtest_height_units'])
            for idx
            in range(len(df))]

        df['bmdtest_weight'] = [
            weight_to_metric(df.loc[idx, 'bmdtest_weight'],
                             df.loc[idx, 'bmdtest_weight_units'])
            for idx
            in range(len(df))]
    except:
        print("Error. Was unable to convert Imperial Values into Metric.")
        quit()

    try:
        print("Counting and removing duplicates.")
        #count_duplicates()
        #remove_duplicates_with_no_id()
        remove_duplicates_with_id()

    except:
        print("Error. Was unable to drop duplicates.")
        quit()

    try:
        print("Selecting features from Data.")
        df = df[['PatientAge',
                 'PatientGender',
                 'bmdtest_height',
                 'bmdtest_weight',
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
                 'hip',
                 'ankle',
                 'clavicle',
                 'elbow',
                 'femur',
                 'spine',
                 'wrist',
                 'shoulder',
                 'tibfib',
                 'wasfractdue2fall',
                 'ptfall',
                 'fxworried',
                 'notworking',
                 'marital',
                 'bmdtest_tscore_fn']]

    except:
        print("Error. Was unable to filter specific features from Dataframe")
        quit()

    try:
        print("Imputing Data into missing Columns.")
        fill_numerical_with_mean()
        fill_nominal_with_mode()
        fill_nominal_bone_with_zero()

    except:
        print("Error. Was unable to perform data imputation.")
        quit()

    try:
        print("Saving Data to CSV file.\n")
        path = Path("Clean_Data_Main.csv")
        df.replace(r'\s+', np.nan, regex=True)
        df.to_csv(path, index=False)
        print("Data saved to " + str(path))

    except:
        print("Error. Unable to save data to a file.")
        quit()
