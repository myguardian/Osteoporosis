import logging
import sys
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ROCAUC

import warnings


def set_directory():
    # detect the current working directory and add the sub directory
    main_path = os.getcwd()
    absolute_path = main_path + "/Stacking_CAROC_SMOTE_Results"
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
    plt.title(f'Permutation Importance Stacking - {name}')
    plt.xlabel('Permutation Importance')
    plt.savefig(f'Stacking_CAROC_SMOTE_Results/Stacking_Permutation_Importance_{name}.png')
    plt.clf()
    plt.close


def plot_predictions(model, X_train, X_test, y_train, y_test, name):
    classes = ['Moderate', 'High']

    os.getcwd()
    # Prediction error
    fig, ax = plt.subplots()
    viz = ClassPredictionError(model, classes=classes,
                               title=f'Class Prediction Error for Stacking - {name}', ax=ax)
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show(outpath=f'Stacking_CAROC_SMOTE_Results/Stacking_Class_Prediction_Error_{name}.png')
    plt.clf()

    # Confusion matrix
    fig, ax = plt.subplots()
    viz = ConfusionMatrix(model, classes=classes,
                         title=f'SVC Confusion Matrix - {name}', ax=ax)
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show(outpath=f'Stacking_CAROC_SMOTE_Results/Stacking_confusion_matrix_{name}.png')
    plt.clf()

    # Classification report
    fig, ax = plt.subplots()
    viz = ClassificationReport(model, support=True, force_model=True,
                              classes=classes, title=f'Stacking Classification Report - {name}', ax=ax)
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show(outpath=f'Stacking_CAROC_SMOTE_Results/Stacking_classification_report_{name}.png')
    plt.clf()

    # ROCAUC
    fig, ax = plt.subplots()
    viz = ROCAUC(model, classes=classes, title=f'ROC Curves for Stacking - {name}', ax=ax)
    viz.fit(X_train, y_train)
    viz.score(X_test, y_test)
    viz.show(outpath=f'Stacking_CAROC_SMOTE_Results/Stacking_ROCAUC_{name}.png')
    plt.clf()


def plot_high_probability_to_tscore(y_prob, tscore, gender, name):
    # scatter plot for high probability vs tscore by gender

    os.getcwd()

    # red for women, blue for men
    colour = {1: 'red', 2: 'blue'}
    plt.scatter(y_prob * 100, tscore, c=gender.map(colour))
    plt.title(f'High Risk Probability vs T-score - Stacking {name}')
    plt.ylabel('T-score')
    plt.xlabel('High Risk Probabiity in %')
    plt.grid(True)
    plt.savefig(f'Stacking_CAROC_SMOTE_Results/Stacking_High_Risk_Probability_vs_Tscore_{name}.png')
    plt.clf()
    plt.close


def build_run_and_plot(path, name):

    # read data
    data = pd.read_csv(path)

    # choose features
    columns = ['PatientAge', 'PatientGender', 'bmdtest_height', 'bmdtest_weight',
               'alcohol', 'obreak', 'smoke', 'oralster',
               'heartdisease', ]

    svc_columns = ['PatientAge', 'bmdtest_height', 'bmdtest_weight']
    rfc_columns = ['PatientAge', 'bmdtest_height', 'bmdtest_weight']
    mlp_columns = ['PatientGender', 'smoke', 'alcohol', 'obreak', 'oralster', 'heartdisease']

    X = data[columns]
    y = data['bmdtest_10yr_caroc']
    print(y)

    # scale numerical values
    scaler = MinMaxScaler()
    X[['PatientAge', 'bmdtest_height', 'bmdtest_weight']] = scaler.fit_transform(
        X[['PatientAge', 'bmdtest_height', 'bmdtest_weight']])

    # label encoder for target value bmdtest_10yr_caroc
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(y)

    # split train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
    #auc = make_scorer(roc_auc_score(multi_class='ovr'))

    print(Counter(y_train))

    # SMOTE
    oversample = SMOTENC(categorical_features=[1], random_state=1)
    X_train, y_train = oversample.fit_resample(X_train, y_train)

    print(Counter(y_train))

    # Create search space for random search
    svc = SVC()
    svc_list = {'C': np.arange(1, 100),
                'gamma': np.arange(0, 5, 0.1),
                'kernel': ['rbf'],
                'class_weight':['balanced', None]}
    svc_search = RandomizedSearchCV(estimator=svc,
                                    param_distributions=svc_list,
                                    n_iter=10,
                                    n_jobs=-1,
                                    verbose=1,
                                    cv=10,
                                    random_state=20,)
    svc_search.fit(X_train[svc_columns], y_train)

    rnd = RandomForestClassifier(random_state=4)
    rnd_list = {'max_depth': [3, 6, 9, 12, 15],
                'n_estimators': [50, 70, 100, 200, 300],
                'max_features': [0.25, 0.5, 0.75, 1.0],
                'criterion': ['gini', 'entropy'],
                'bootstrap': [True, False]}
    rnd_search = RandomizedSearchCV(estimator=rnd,
                                    param_distributions=rnd_list,
                                    n_iter=10,
                                    n_jobs=-1,
                                    cv=10,
                                    verbose=1,
                                    random_state=5,)
    rnd_search.fit(X_train[rfc_columns],  y_train)

    mlp = MLPClassifier(alpha=0.5,
                        max_iter=1000,
                        hidden_layer_sizes=(100,100,50),
                        learning_rate='constant',
                        solver='adam',
                        learning_rate_init=0.001,
                        random_state=1)
    mlp.fit(X_train[mlp_columns], y_train)

    estimators = [
        ('svc', SVC(**svc_search.best_params_, probability=True)),
        ('rnd', RandomForestClassifier(**rnd_search.best_params_, random_state=6)),
        ('mlp', mlp)
    ]

    model = StackingClassifier(estimators=estimators, cv=5)

    np.random.seed(20)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    #roc_auc = roc_auc_score(y_test['bmdtest_10yr_caroc'], y_pred)

    # plot necessary graphs
    plot_analysis(model, X_test, y_test, columns, name)
    plot_predictions(model, X_train, X_test, y_train, y_test, name)
    y_prob = pd.DataFrame(y_prob)
    #plot_high_probability_to_tscore(y_prob[1], y_test['bmdtest_tscore_fn'], X_test['PatientGender'], name)

    #scores = cross_val_score(model, X, y['bmdtest_10yr_caroc'], cv=10,)
    #print("Cross Validation Score - ROCAUC: ", scores.mean(), scores.std())
    #print(scores)


if __name__ == '__main__':

    # suppress warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.options.mode.chained_assignment = None

    try:
        # get data file
        # run with ../1-Data_Cleaning/Output/Clean_Data_Main.csv
        file_name = sys.argv[1]
        logging.info(f'Loading Data {file_name}\n')

        # set the directory for results
        set_directory()
    except ValueError as er:
        logging.error(er)
        logging.error('Unable to load the CSV file')

    try:
        build_run_and_plot(file_name, 'CAROC_SMOTE')
    except ValueError as er:
        logging.error(er)
        logging.error('Unable to build and run model')