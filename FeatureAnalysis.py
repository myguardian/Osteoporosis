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

logging.basicConfig(level=logging.INFO)

# 4 Columns are numerical
numerical_col = {'PatientAge',
                 'bmdtest_height',
                 'bmdtest_weight',
                 'bmdtest_tscore_fn'}

# Columns are nominal
nominal_col = {
    'PatientGender',
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
    'marital',
    'hip',
    'ankle',
    'clavicle',
    'elbow',
    'femur',
    'spine',
    'wrist',
    'shoulder',
    'tibfib'
}


def set_directory():
    # detect the current working directory and add the sub directory
    main_path = os.getcwd()
    absolute_path = main_path + "/analysis_results"
    try:
        os.mkdir(absolute_path)
    except OSError:
        logging.info("Creation of the directory %s failed. Folder already exists." % absolute_path)
    else:
        logging.info("Successfully created the directory %s " % absolute_path)


def create_box_plot_chart(data_frame, feature):
    try:
        logging.info(f'Creating Box Plot for {feature}')
        flag = False
        if data_frame[feature].isnull().sum() > 0:
            flag = True

            # Select only filled data
            data_frame = data_frame.dropna()
            data_frame.reset_index(drop=True, inplace=True)

        sample_data = data_frame[feature]
        plt.boxplot(sample_data)

        if flag:
            plt.title(feature + " Box Plot - With Missing Data")
        else:
            plt.title(feature + " Box Plot - With No Missing Data")
        plt.savefig(f'analysis_results/numerical_{feature}_boxplot')
        plt.clf()
    except ValueError as er:
        logging.error(er)
        logging.error(f'Cannot create box plot for {feature}')


def create_histogram_chart(data_frame, feature):
    try:
        logging.info(f'Creating Histogram for {feature}')
        data_frame.hist(column=feature, bins=50, grid=False, figsize=(5, 5))
        plt.title(feature + " Histogram")
        plt.savefig(f'analysis_results/numerical_{feature}_hist')
        plt.clf()
    except ValueError as er:
        logging.error(er)
        logging.error(f'Cannot create histogram for {feature}')


def create_pie_chart(data_frame, feature):
    try:
        logging.info(f'Creating Piechart for {feature}')
        counter = data_frame[feature].value_counts()
        counter.plot(kind='pie', autopct='%1.1f%%', figsize=(5, 5))
        plt.title(feature + " Piechart")
        plt.savefig(f'analysis_results/nominal_{feature}_pie')
        plt.clf()
    except ValueError as er:
        logging.error(er)
        logging.error(f'Cannot create pie chart for {feature}')


def perform_data_analysis(path):
    # Load the data from the CSV file and select the features
    data = pd.read_csv(path)
    features = list(data.columns.values)

    try:
        # Create a description for each of the features
        logging.info(f'Creating description data frame')
        description = data.describe(include='all')
        description.loc['dtype'] = data.dtypes
        description.loc['%_missing'] = data.isnull().mean()
        extra_columns = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'dtype', '%_missing']
        description.insert(0, "Measure", extra_columns)

        description.to_csv('analysis_results/aggregate_data_descriptions.csv', index=False)
        data.corr().to_csv('analysis_results/aggregate_data_correlation.csv', index=False)
    except ValueError as er:
        logging.error(er)
        logging.error('Unable to create description data frame')

    try:
        # Create a correlation figure, histogram or pie chart for each image
        logging.info(f'Creating images for each feature.')
        y = data['bmdtest_tscore_fn']
        for feature in features:
            x = data[feature]
            plt.title(feature + " Correlation with BMD Test Score")
            plt.xlabel(feature)
            plt.ylabel('bmdtest_tscore_fn')
            plt.scatter(x, y)
            plt.savefig(f'analysis_results/correlation_with_bmdtscore_{feature}')
            plt.clf()

            if feature in numerical_col:
                create_histogram_chart(data, feature)
                create_box_plot_chart(data, feature)
            elif feature in nominal_col:
                create_pie_chart(data, feature)

        logging.info('Success, closing program')
    except ValueError as er:
        logging.error(er)
        logging.error('Unable to iterate through each feature.')


if __name__ == "__main__":

    try:
        # Get the data from the argument
        file_name = sys.argv[1]
        # file_name = 'Clean_Data_Main.csv'
        logging.info(f'Loading Data {file_name}\n')

        # Create the directory where the CSV files and images are going to be saved
        set_directory()

        # Perform the analysis and generate the images
        perform_data_analysis(file_name)
    except ValueError as e:
        logging.error(e)
        logging.error('Unable to load the CSV File')
