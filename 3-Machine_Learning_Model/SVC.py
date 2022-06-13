import logging
import sys
import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ROCAUC
from yellowbrick.model_selection import FeatureImportances

import warnings

def set_directory():
    # detect the current working directory and add the sub directory
    main_path = os.getcwd()
    absolute_path = main_path + "/SVC_Results"
    try:
        os.mkdir(absolute_path)
    except OSError:
        logging.info("Creation of the directory %s failed. Folder already exists." % absolute_path)
    else:
        logging.info("Successfully created the directory %s." % absolute_path)

def plot_analysis(model, X_test, y_test, columns, name):
    os.getcwd()
    # Permutation importance
    importance = permutation_importance(model, X_test, y_test, n_repeats=100, scoring='roc_auc')
    np_columns = np.array(columns)
    sorted_idx = importance.importances_mean.argsort()
    plt.barh(np_columns[sorted_idx], importance.importances_mean[sorted_idx])
    plt.title(f'Permutation Importance SVC - {name}')
    plt.xlabel('Permutation Importance')
    plt.savefig(f'SVC_Results/SVC_Permutation_Importance_{name}.png')
    plt.clf()
    plt.close

def plot_predictions(model, X_train, X_test, y_train, y_test, name):
    classes = ['Moderate', 'High']

    os.getcwd()
    # Prediction error
    fig, ax = plt.subplots()
    viz = ClassPredictionError(model, classes=classes,
                               title=f'Class Prediction Error for SVC - {name}', ax=ax)
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show(outpath=f'SVC_Results/SVC_Class_Prediction_Error_{name}.png')
    plt.clf()

    # Confusion matrix
    fig, ax = plt.subplots()
    viz = ConfusionMatrix(model, classes=classes,
                         title=f'SVC Confusion Matrix - {name}', ax=ax)
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show(outpath=f'SVC_Results/SVC_confusion_matrix_{name}.png')
    plt.clf()

    # Classification report
    fig, ax = plt.subplots()
    viz = ClassificationReport(model, support=True, force_model=True,
                              classes=classes, title=f'SVC Classification Report - {name}', ax=ax)
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show(outpath=f'SVC_Results/SVC_classification_report_{name}.png')
    plt.clf()

    # ROCAUC
    fig, ax = plt.subplots()
    viz = ROCAUC(model, classes=classes, title=f'ROC Curves for SVC - {name}', ax=ax)
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show(outpath=f'SVC_Results/SVC_ROCAUC_{name}.png')
    plt.clf()

    if name == 'Linear':
        fig, ax = plt.subplots()
        viz = FeatureImportances(model, relative=False, title='Feature Importances of 11 Feature using SVC - linear',
                                 ax=ax)
        viz.fit(X_test, y_test)
        viz.show(outpath='SVC_Results/SVC_Feature_Importance_Linear.png')
        viz.finalize()
        plt.clf()

def plot_high_probability_to_tscore(y_prob, tscore, gender, name):
    # scatter plot for high probability vs tscore by gender

    os.getcwd()

    # red for women, blue for men
    colour = {1: 'red', 2: 'blue'}
    plt.scatter(y_prob * 100, tscore, c=gender.map(colour))
    plt.title('High Risk Probability vs T-score')
    plt.ylabel('T-score')
    plt.xlabel('High Risk Probabiity in %')
    plt.grid(True)
    plt.savefig(f'SVC_Results/SVC_High_Risk_Probability_vs_Tscore_{name}.png')
    plt.clf()
    plt.close

def build_run_and_plot(path, c, gamma, kernel, name):

    # read data
    data = pd.read_csv(path)

    # choose features
    columns = ['PatientAge', 'PatientGender', 'bmdtest_height', 'bmdtest_weight',
               'smoke', 'alcohol', 'oralster', 'obreak',
               'ptunsteady', 'ptfall',
               'arthritis', 'heartdisease', 'respdisease', 'hbp', 'cholesterol']
    X = data[columns]
    y = data[['bmdtest_10yr_caroc', 'bmdtest_tscore_fn']]

    # scale numerical values
    scaler = MinMaxScaler()
    X[['PatientAge', 'bmdtest_height', 'bmdtest_weight']] = scaler.fit_transform(
        X[['PatientAge', 'bmdtest_height', 'bmdtest_weight']])

    # label encoder for target value bmdtest_10yr_caroc
    le = LabelEncoder()
    y['bmdtest_10yr_caroc'] = le.fit_transform(y['bmdtest_10yr_caroc'])

    # set random seed
    np.random.seed(20)

    # split train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)

    # build and fit model
    model = SVC(C=c, gamma=gamma, kernel=kernel,
                random_state=20, probability=True)
    model.fit(X_train, y_train['bmdtest_10yr_caroc'])

    # predict the probability and convert the list to data frame
    y_prob = model.predict_proba(X_test)
    y_prob = pd.DataFrame(y_prob)

    # plot necessary graphs
    plot_analysis(model, X_test, y_test['bmdtest_10yr_caroc'], columns, name)
    plot_predictions(model, X_train, X_test, y_train['bmdtest_10yr_caroc'], y_test['bmdtest_10yr_caroc'], name)
    plot_high_probability_to_tscore(y_prob[1], y_test['bmdtest_tscore_fn'], X_test['PatientGender'], name)

if __name__ == '__main__':

    # suppress warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.options.mode.chained_assignment = None

    try:
        # get data file
        file_name = sys.argv[1]
        logging.info(f'Loading Data {file_name}\n')

        # set the directory for results
        set_directory()
    except ValueError as er:
        logging.error(er)
        logging.error('Unable to load the CSV file')

    try:
        build_run_and_plot(file_name, 10, 0.1, 'rbf', 'C10_g0.1')
        build_run_and_plot(file_name, 10, 'scale', 'rbf', 'C10_scale')
    except ValueError as er:
        logging.error(er)
        logging.error('Unable to build and run model')