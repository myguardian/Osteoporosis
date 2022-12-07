import logging
import os
import pickle
import shutil
import sys
from collections import Counter
from glob import glob
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from yellowbrick import ROCAUC
from yellowbrick.classifier import ClassPredictionError, ConfusionMatrix, ClassificationReport
from yellowbrick.model_selection import FeatureImportances
import shap

'''This code was used to load a saved model that has already been trained 
    and execute predictions on new data in the remote dataset.'''


def set_directory():
    # detect the current working directory and add the subdirectory
    main_path = os.getcwd()
    absolute_path = main_path + "/Output"
    try:
        Path(f'{absolute_path}/PreTrainedModels/RFC').mkdir(parents=True, exist_ok=True)
    except OSError:
        logging.info("Creation of the directory %s failed. Folder already exists." % absolute_path)
    else:
        logging.info("Successfully created the directory %s " % absolute_path)


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
    dataset.drop(['PatientId'], axis=1, inplace=True)

    # dataset.drop(dataset.index[dataset['ankle'] == 1], inplace=True)
    print('Number of rows in the dataset after the rows with ankle were removed: {}.\n{} rows were removed.'
          .format(dataset.shape[0], size - dataset.shape[0]))
    # Create the dataset that will be used to train the models
    # and the data that will be used to perform the predictions to test the models

    dataset.reset_index(drop=True, inplace=True)

    print('Data for Modeling: ' + str(dataset.shape))

    return dataset


def encode_cat_data(data):
    cat_features = ['parentbreak', 'alcohol',
                    'arthritis', 'diabetes',
                    'oralster', 'smoke']
    dataset = data.copy()

    for feature in cat_features:
        cat_one_hot = pd.get_dummies(dataset[feature], prefix=f'{feature}', drop_first=False)
        dataset = dataset.drop(feature, axis=1)
        dataset = dataset.join(cat_one_hot)

    return dataset


def create_model_set(data, features, target):
    """Splits the Data into the Features you want to train and the target your model will be predicting"""
    copy = data.copy()
    feature_set = copy[features]
    target_column = copy[target]
    return feature_set, target_column


def plot_results(classification_model, x_tr, y_tr, x_te, y_te):
    """A function that takes in a model and plots the results and saves them to the
    models_results directory """
    current_dir = os.getcwd()
    dst_dir = current_dir + "/Output/PreTrainedModels/RFC"
    plot_types = ['classification_report', 'conf_mat', 'ROCAUC', 'class_pred_err', 'feature_importance']
    classes = ['High', 'Low', 'Moderate']

    try:
        # Create plots for the model
        for plot in plot_types:
            try:
                if plot == 'classification_report':
                    visualizer = ClassificationReport(classification_model, support=True, classes=classes,
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


def load_model(filename):
    try:
        model = pickle.load(open(filename, 'rb'))
        return model
    except FileNotFoundError:
        logging.error("Cannot find .sav file. Exiting Operation.")


def create_shap_sample(data, num_of_instances):
    sample = shap.utils.sample(data, num_of_instances)
    return sample


def create_explainer(model, sample):
    explainer = shap.Explainer(model.predict, sample)
    return explainer


def plot_summary(explainer, data, feature_names):
    shap_values = explainer(data)
    shap.summary_plot(shap_values, data, feature_names=feature_names, show=False)
    plt.savefig(f'Output/PreTrainedModels/RFC/shap_summary.png')
    plt.clf()


def move_model():
    current_dir = os.getcwd()
    dst_dir = current_dir + "/Output/PreTrainedModels/RFC"
    files = glob('*.sav')
    if len(files) == 0:
        logging.info('There are no Models to move.')
        return
    for file in files:
        if file.endswith('.sav'):
            shutil.move(os.path.join(current_dir, file),
                        os.path.join(dst_dir, file))


if __name__ == "__main__":
    try:
        set_directory()
        # Get the data from the argument
        model_file = sys.argv[1]
        file_name = sys.argv[2]

        # file_name = '../FRAX_V3.csv'
        logging.info(f'Loading Data {file_name}\n')

        # Perform the analysis and generate the images

        main_data = setup_data(file_name)
    except ValueError as e:
        logging.error(e)
        logging.error('Unable to load the CSV File')

    if main_data is not None:

        main_data = encode_cat_data(main_data)

        X, y = create_model_set(main_data, ['PatientAge', "PatientGender", 'bmdtest_weight', 'bmdtest_height',
                                 'parentbreak_1.0', 'alcohol_1.0', 'arthritis_1.0', 'diabetes_1.0',
                                 'oralster_1.0', 'oralster_2.0', 'smoke_1.0'],
                                 'Frax_BMD_RiskLevel')
        counter = Counter(y)
        print(counter)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=2)
        counter = Counter(y_test)
        print(counter)
        model = load_model(model_file)

        if model is not None:

            yhat = model.predict(X_test)

            # shap_sample = create_shap_sample(X_train, int((len(X_train) * 0.2)))
            #
            # model_explainer = create_explainer(model, shap_sample)
            #
            # plot_summary(model_explainer, shap_sample, ['PatientAge', "PatientGender", 'bmi', 'alcohol_1.0',
            #                                             'smoke_1.0', 'arthritis_1.0', 'diabetes_1.0'])

            plot_results(model, X_train, y_train, X_test, y_test)

            move_model()
        else:
            print('Model could not be loaded.')

        print('All Operations have been completed. Closing Program.')

    else:
        logging.error('No data exists.')
