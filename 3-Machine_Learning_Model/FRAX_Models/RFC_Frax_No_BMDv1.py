from collections import Counter

from imblearn.over_sampling import SMOTENC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, PolynomialFeatures
import pandas as pd
import shutil
# import the os module
import numpy
import numpy.random as np_random
import os
from glob import glob
from yellowbrick import ROCAUC
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix, ClassPredictionError
from yellowbrick.model_selection import FeatureImportances
import logging
import sys
import pickle
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import dump, load

'''
This script was created to be used with datasets that use the FRAX_Models Risk Assessment tool that can be found here:
https://www.sheffield.ac.uk/FRAX/tool.aspx?country=19 it only uses the features in our local dataset 
'''


def setup_data(path):
    dataset = pd.read_csv(path)

    missing = pd.DataFrame(dataset.isnull().sum(), columns=['Total'])
    missing['%'] = (missing['Total'] / dataset.shape[0]) * 100
    missing.sort_values(by='%', ascending=False)

    size = dataset.shape[0]
    dataset = dataset.dropna()

    print('Number of rows in the dataset after the rows with missing values were removed: {}.\n{} rows were removed.'
          .format(dataset.shape[0], size - dataset.shape[0]))

    # Remove Severe Outliers
    cols = ['bmdtest_height', 'bmdtest_weight']

    for c in cols:
        upper_level = dataset[c].mean() + 3 * dataset[c].std()
        lower_level = dataset[c].mean() - 3 * dataset[c].std()
        dataset = dataset[(dataset[c] > lower_level) & (dataset[c] < upper_level)]

    print('Number of rows in the dataset after the rows with missing values were removed: {}.\n{} rows were removed.'
          .format(dataset.shape[0], size - dataset.shape[0]))

    # Reset the Index
    dataset.reset_index(drop=True, inplace=True)

    # Drop the PatientID column as it is no longer needed
    # dataset.drop(['PatientId'], axis=1, inplace=True)

    # dataset.drop(dataset.index[dataset['ankle'] == 1], inplace=True)
    # print('Number of rows in the dataset after the rows with ankle were removed: {}.\n{} rows were removed.'
    #       .format(dataset.shape[0], size - dataset.shape[0]))
    # Create the dataset that will be used to train the models
    # and the data that will be used to perform the predictions to test the models

    dataset.reset_index(drop=True, inplace=True)

    print('Data for Modeling: ' + str(dataset.shape))

    return dataset


def set_directory():
    """detect the current working directory and add the sub directory"""
    main_path = os.getcwd()
    absolute_path = main_path + "/Output"
    try:
        Path(f'{absolute_path}/RemoteTrainedModels/RFC_Frax_No_BMD').mkdir(parents=True, exist_ok=True)
    except OSError:
        logging.info("Creation of the directory %s failed. Folder already exists." % absolute_path)
    else:
        logging.info("Successfully created the directory %s " % absolute_path)


def create_model_set(data, features, target):
    """Splits the Data into the Features you want to train and the target your model will be predicting"""
    copy = data.copy()
    feature_set = copy[features]
    target_column = copy[target]
    return feature_set, target_column


def encode_cat_data(data):
    """Encode the categorical features using one hot encoding"""
    cat_features = ['parentbreak', 'alcohol',
                    'arthritis', 'diabetes',
                    'oralster', 'smoke', 'obreak',
                    # These features were determined to apply minimal impact even
                    # 'respdisease', 'hbp','heartdisease',
                    # 'ptunsteady', 'wasfractdue2fall', 'cholesterol',
                    # 'ptfall', 'shoulder', 'wrist', 'bmdtest_10yr_caroc'
                    ]
    dataset = data.copy()

    for feature in cat_features:
        cat_one_hot = pd.get_dummies(dataset[feature], prefix=f'{feature}', drop_first=False)
        dataset = dataset.drop(feature, axis=1)
        dataset = dataset.join(cat_one_hot)

    return dataset


def scale_data(data):
    """Scale only the Numerical features in the dataset."""
    cols_to_scale = ['PatientAge', 'bmdtest_weight', 'bmdtest_height']
    scaler = StandardScaler()
    scaler.fit(data[cols_to_scale])
    data[cols_to_scale] = scaler.transform(data[cols_to_scale])
    dump(scaler, 'Output/RemoteTrainedModels/RFC_Frax_No_BMD/scaler_joblib.sav')

    return data


def plot_results(classification_model, x_tr, y_tr, x_te, y_te):
    """A function that takes in a model and plots the results and saves them to the
    models_results directory """
    current_dir = os.getcwd()
    dst_dir = current_dir + "/Output/RemoteTrainedModels/RFC_Frax_No_BMD"
    plot_types = ['classification_report', 'conf_mat', 'ROCAUC', 'class_pred_err', 'feature_importance']
    # classes = ['High', 'Low', 'Moderate']

    try:
        # Create plots for the model
        for plot in plot_types:
            try:
                if plot == 'classification_report':
                    visualizer = ClassificationReport(classification_model, support=True,
                                                      is_fitted=True)
                    visualizer.fit(x_tr, y_tr)
                    visualizer.score(x_te, y_te)
                    visualizer.show(outpath=f"RFC_classification_report.png", clear_figure=True)

                elif plot == 'conf_mat':
                    visualizer = ConfusionMatrix(classification_model, is_fitted=True)
                    visualizer.fit(x_tr, y_tr)
                    visualizer.score(x_te, y_te)
                    visualizer.show(outpath=f"RFC_confusion_matrix.png", clear_figure=True)

                elif plot == 'ROCAUC':
                    visualizer = ROCAUC(classification_model, is_fitted=True)
                    visualizer.fit(x_tr, y_tr)
                    visualizer.score(x_te, y_te)
                    visualizer.show(outpath=f"RFC_ROCAUC.png", clear_figure=True)

                elif plot == 'class_pred_err':
                    visualizer = ClassPredictionError(classification_model, is_fitted=True)
                    visualizer.fit(x_tr, y_tr)
                    visualizer.score(x_te, y_te)
                    visualizer.show(outpath=f"RFC_Class_Prediction_Error.png", clear_figure=True)

                else:
                    result = permutation_importance(classification_model, x_tr, y_tr, n_repeats=50, scoring='roc_auc')
                    sorted_importances_idx = result.importances_mean.argsort()

                    # importance = pd.DataFrame(result.importances.T, columns=X.columns)
                    # importance.to_csv(f'RFC_permutation_importance.csv')

                    plt.barh(X.columns[sorted_importances_idx], result.importances_mean[sorted_importances_idx].T)
                    plt.xlabel('Permutation Importance')
                    plt.savefig(f'RFC_permutation_importance.png')
                    plt.clf()

            except Exception as er:
                logging.error(er)
                logging.error(f"'{plot}' plot is not available for this model. Plotting a different graph.")

        print(f"All Plots for {classification_model} have been created Successfully")

        print(f'Moving plots for {classification_model}')

        # Move the plots to their respective models directory
        files = glob('*.png')
        if len(files) == 0:
            logging.info('There are no Plots to move.')
            return
        for file in files:
            if file.endswith('.png'):
                shutil.move(os.path.join(current_dir, file),
                            os.path.join(dst_dir, file))

    except Exception as er:
        logging.error(er)
        logging.error("Plot Operations were unable to be completed.")


def create_search_space():
    param = {'max_depth': [3, 6, 9, 12, 15],
             'n_estimators': [50, 70, 100, 200, 300],
             'max_features': [0.25, 0.5, 0.75, 1.0],
             'criterion': ['gini', 'entropy'],
             'bootstrap': [True, False]}
    return param


def create_shap_sample(data, num_of_instances):
    sample = shap.utils.sample(data, num_of_instances)
    return sample


def create_explainer(model, sample):
    explainer = shap.Explainer(model.predict, sample)
    return explainer


def plot_summary(explainer, data, feature_names):
    shap_values = explainer(data)
    shap.summary_plot(shap_values, data, feature_names=feature_names, show=False)
    plt.savefig(f'Output/RemoteTrainedModels/RFC_Frax_No_BMD/shap_summary.png')
    plt.clf()


def save_model(filename, model):
    current_dir = os.getcwd()
    dst_dir = current_dir + "/Output/RemoteTrainedModels/RFC_Frax_No_BMD"
    pickle.dump(model, open(filename, 'wb'))
    files = glob('*.sav')
    if len(files) == 0:
        logging.info('There are no Plots to move.')
        return
    for file in files:
        if file.endswith('.sav'):
            shutil.move(os.path.join(current_dir, file),
                        os.path.join(dst_dir, file))


if __name__ == "__main__":
    try:
        set_directory()
        # Get the data from the argument
        file_name = sys.argv[1]
        # file_name = '../FRAX_V3.csv'
        logging.info(f'Loading Data {file_name}\n')

        # Perform the analysis and generate the images
        main_data = setup_data(file_name)

    except ValueError as e:
        logging.error(e)
        logging.error('Unable to load the CSV File')

    rfc = create_search_space()

    if main_data is not None:
        main_data = encode_cat_data(main_data)
        X, y = create_model_set(main_data,
                                ['PatientId', 'PatientAge', "PatientGender", 'bmdtest_weight', 'bmdtest_height',
                                 'parentbreak_1.0', 'alcohol_1.0', 'arthritis_1.0', 'diabetes_1.0',
                                 'oralster_1.0', 'oralster_2.0', 'smoke_1.0'],
                                 'Frax_No_BMD_RiskLevel')
        counter = Counter(y)
        print(counter)

        logging.info('Explanatory Features and Target columns have been created')


        # Scale the Data
        X = scale_data(X)
        logging.info('Data has been scaled')

        # Split the data into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=2)
        copy = X_test.copy(deep=True)
        X_train.drop('PatientId', axis=1, inplace=True)
        X_test.drop('PatientId', axis=1, inplace=True)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, random_state=3)
        print(Counter(y_train))

        # Implement SMOTE on the dataset and apply it only to the training set, this should not be applied on the test set
        oversample = SMOTENC(categorical_features=[1, 4, 5, 6, 7, 8, 9, 10], random_state=1)
        X_train, y_train = oversample.fit_resample(X_train, y_train)
        print(Counter(y_train))
        print(Counter(y_test))
        rnd_search = RandomizedSearchCV(RandomForestClassifier(random_state=4), rfc, n_iter=30, n_jobs=-1, cv=10,
                                        verbose=1, random_state=5)
        rnd_search.fit(X_val, y_val)

        # Random Search
        classifier = RandomForestClassifier(**rnd_search.best_params_, random_state=6)
        # classifier = RandomForestClassifier(max_depth=9, max_features=1.0, n_estimators=200, random_state=1)

        classifier.fit(X_train, y_train)

        # predict from test set
        yhat = classifier.predict(X_test)
        counter = Counter(yhat)
        print(counter)

        # Save Model
        save_model('FRAX_random_forest_model.sav', classifier)

        # This was just used to see the prediction results on our local dataset.
        # copy['Frax_BMD_RiskLevel'] = yhat
        #
        # copy.to_csv('FRAX_RFC_Predictions.csv')

        plot_results(classifier, X_train, y_train, X_test, y_test)

        print('All Operations have been completed. Closing Program.')

    else:
        logging.error('No data exists.')
