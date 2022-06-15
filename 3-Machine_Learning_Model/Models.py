import logging
import sys

from yellowbrick.regressor import *
from yellowbrick.model_selection import LearningCurve, ValidationCurve, RFECV, FeatureImportances
import matplotlib.pyplot as plt
import numpy as np
import shutil
# import the os module
import os
from glob import glob
from pycaret.regression import *
from pycaret.utils import check_metric
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, make_scorer


def set_directory():
    # detect the current working directory and add the sub directory
    main_path = os.getcwd()
    absolute_path = main_path + "/models_results"
    model1_path = absolute_path + "/model_1"
    model2_path = absolute_path + "/model_2"
    model3_path = absolute_path + "/model_3"
    model4_path = absolute_path + "/model_4"
    try:
        os.mkdir(absolute_path)
        os.mkdir(model1_path)
        os.mkdir(model2_path)
        os.mkdir(model3_path)
        os.mkdir(model4_path)

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

    # Reduce the amount of columns produced by the types of fractures and
    # consolidate them into two columns, fractured and fracture_type
    dataset = dataset.melt(id_vars=['PatientId', 'PatientAge', 'PatientGender', 'bmdtest_weight', 'bmdtest_height',
                                    'parentbreak', 'alcohol', 'arthritis', 'cancer', 'diabetes', 'heartdisease',
                                    'oralster', 'smoke', 'respdisease', 'ptunsteady', 'wasfractdue2fall', 'ptfall',
                                    'bmdtest_tscore_fn'],
                           value_vars=['ankle', 'clavicle', 'elbow', 'femur', 'wrist', 'tibfib'],
                           var_name="fracture_type",
                           value_name='fractured',
                           ignore_index=False)

    # Sort dataset to ensure patients with fractures are kept after cleaning
    dataset = dataset.sort_values(by='fractured', ascending=False)

    # Clean up the duplicated patients from melting the dataset
    updated_dataset = dataset[~(dataset[['PatientId']].duplicated(keep='first'))]

    # Reset the Index
    updated_dataset.reset_index(drop=True, inplace=True)

    # Drop the PatientID column as it is no longer needed
    updated_dataset.drop(['PatientId'], axis=1, inplace=True)

    # Create the dataset that will be used to train the models
    # and the data that will be used to perform the predictions to test the models
    data = updated_dataset.sample(frac=0.9, random_state=786)
    data_unseen = updated_dataset.drop(data.index)

    data.reset_index(drop=True, inplace=True)
    data_unseen.reset_index(drop=True, inplace=True)

    print('Data for Modeling: ' + str(data.shape))
    print('Unseen Data For Predictions: ' + str(data_unseen.shape))

    return data, data_unseen


def perform_predictions(top_models):
    """
    A function that performs the predictions on the top 3 models produced by PyCaret, saves the predictions to a csv
    file per model, and saves the RMSE and R2 scores to a .txt file. It will then move these files to the
    models_results directory and their respective model_num directory.
    """
    current_dir = os.getcwd()
    dst_dir = current_dir + "/models_results"

    # Perform Predictions on the top models with the unseen data
    print('Performing Predictions on unseen data')
    predictions = [predict_model(i, data=unseen_data) for i in top_models]

    # Write the results to a text file and the predictions to csvs
    logging.info('Saving Results to Model_Results.txt')
    with open('Model_Results.txt', 'w') as result_file:
        for i in range(len(top_models)):
            result_file.write(
                f'RMSE for Model {i + 1}: \n{top_models[i]}: \n' +
                f"{check_metric(predictions[i].bmdtest_tscore_fn, predictions[i].Label, 'RMSE')} \n"
            )
            result_file.write(
                f'R2 for Model {i + 1}: \n{top_models[i]}: \n' +
                f"{check_metric(predictions[i].bmdtest_tscore_fn, predictions[i].Label, 'R2')} \n"
            )


    # Move the Model Results csvs to the results directory
    try:
        shutil.move(os.path.join(current_dir, 'Model_Results.txt'), os.path.join(dst_dir, 'Model_Results.txt'))

        print('Model results have been moved successfully.')
    except Exception as er:
        logging.error(er)
        logging.error("There was an error when moving the files")


def plot_results(top_models):
    """A function that takes in the top 3 models produced by PyCaret and plots the results and saves them to the
    models_results directory """
    current_dir = os.getcwd()
    dst_dir = current_dir + "/models_results"
    plot_types = ['residuals', 'error', 'learning', 'vc', 'feature', 'cooks', 'rfe', 'permutation']

    try:
        # Analyze the finalized models by saving their plots against the test data
        for i in range(len(top_models)):
            # Create plots for the top 3 models
            for plot in plot_types:
                try:
                    if plot == 'residuals':
                        visualizer = ResidualsPlot(top_models[i])
                        X_train = get_config('X_train')
                        y_train = get_config('y_train')
                        X_test = get_config('X_test')
                        y_test = get_config('y_test')
                        visualizer.fit(X_train, y_train)
                        visualizer.score(X_test, y_test)
                        visualizer.show(outpath=f'model{i+1}_residuals.png', clear_figure=True)

                    elif plot == 'error':
                        visualizer = PredictionError(top_models[i])
                        X_train = get_config('X_train')
                        y_train = get_config('y_train')
                        X_test = get_config('X_test')
                        y_test = get_config('y_test')
                        visualizer.fit(X_train, y_train)
                        visualizer.score(X_test, y_test)
                        visualizer.show(outpath=f'model{i + 1}_prediction_error.png', clear_figure=True)

                    elif plot == 'learning':
                        visualizer = LearningCurve(top_models[i], scoring='r2', param_name='Training Instances',
                                                   param_range=np.arange(1, 800))
                        X = get_config('X')
                        y = get_config('y')
                        visualizer.fit(X, y)
                        visualizer.show(outpath=f'model{i + 1}_learning_curve.png', clear_figure=True)

                    elif plot == 'vc':
                        # This may not be viable to run in a script easily as each model algorithm uses a different parameter
                        visualizer = ValidationCurve(top_models[i], scoring='r2', param_range=np.arange(1, 800),
                                                     param_name='Training Instances')
                        X = get_config('X')
                        y = get_config('y')
                        visualizer.fit(X, y)
                        visualizer.show(outpath=f'model{i + 1}_validation_curve.png', clear_figure=True)

                    elif plot == 'feature':
                        visualizer = FeatureImportances(top_models[i])
                        X = get_config('X')
                        y = get_config('y')
                        visualizer.fit(X, y)
                        visualizer.show(outpath=f'model{i + 1}_feature_importance.png', clear_figure=True)

                    elif plot == 'cooks':
                        visualizer = CooksDistance()
                        X = get_config('X')
                        y = get_config('y')
                        visualizer.fit(X, y)
                        visualizer.show(outpath=f'model{i + 1}_cooks_distance.png', clear_figure=True)

                    elif plot == 'permutation':
                        X_train = get_config('X_train')
                        y_train = get_config('y_train')
                        plot_permutation_importance(top_models[i], i + 1, X_train, y_train)

                    else:
                        visualizer = RFECV(top_models[i])
                        X = get_config('X')
                        y = get_config('y')
                        visualizer.fit(X, y)
                        visualizer.show(outpath=f'model{i + 1}_recursive_feature_elimination.png', clear_figure=True)

                except Exception as er:
                    logging.error(er)
                    logging.error(f"'{plot}' plot is not available for this model. Plotting a different graph.")

            print(f"All Plots for {top_models[i]} have been created Successfully")

            print(f'Moving plots for {top_models[i]}')

            # Move the plots to their respective models directory
            files = glob('*.png')
            if len(files) == 0:
                logging.info('There are no Plots to move.')
                return
            for file in files:
                # USE THIS IF STATEMENT IF SCRIPT IS NOT RAN IN A CLEAN DIRECTORY
                # if ((file.__contains__('Cooks Distance.png')) 
                #    or (file.__contains__('Feature Importance.png')) or (file.__contains__('Feature Selection.png')) 
                #    or (file.__contains__('Learning Curve.png')) or (file.__contains__('Manifold Learning.png'))
                #    or (file.__contains__('Prediction Error.png')) or (file.__contains__('Residuals.png')) 
                #    or (file.__contains__('Validation Curve.png'))
                #    ):
                if file.endswith('.png'):
                    shutil.move(os.path.join(current_dir, file), os.path.join(dst_dir + f"/model_{i + 1}", file))

    except Exception as er:
        logging.error(er)
        logging.error("Plot Operations were unable to be completed.")


def plot_permutation_importance(model, name, X, y):

    result = permutation_importance(model, X, y, n_repeats=100, scoring='r2')
    sorted_importances_idx = result.importances_mean.argsort()

    plt.barh(X.columns[sorted_importances_idx], result.importances_mean[sorted_importances_idx].T)
    plt.xlabel('Permutation Importance')
    plt.savefig(f'model{name}_permutation_importance.png')


if __name__ == "__main__":
    try:
        # Get the data from the argument
        file_name = sys.argv[1]
        # file_name = '../1-Data_Cleaning/Clean_Data_Main.csv'
        logging.info(f'Loading Data {file_name}\n')

        # Create the directory where the CSV files and images are going to be saved
        set_directory()

        # Perform the analysis and generate the images
        main_data, unseen_data = setup_data(file_name)

        if main_data is not None:
            exp_name = setup(data=main_data, target='bmdtest_tscore_fn', session_id=123, train_size=0.7, fold=10,
                             log_experiment=True, experiment_name='t_score_predictor_v2',
                             feature_interaction=True, feature_ratio=True, feature_selection=True,
                             bin_numeric_features=['bmdtest_weight', 'bmdtest_height'], ignore_low_variance=True,
                             feature_selection_threshold=0.5, feature_selection_method='boruta',
                             categorical_features=['parentbreak', 'alcohol', 'oralster', 'smoke'],
                             normalize=True, normalize_method='minmax', silent=True, html=False)

            best_models = compare_models(sort='RMSE', n_select=3, fold=10)

            best_models.append(create_model('omp'))

            # Tune the Models
            tuned_models = [tune_model(i, optimize='RMSE', n_iter=100) for i in best_models]

            # Finalize the Model
            final_models = [finalize_model(i) for i in tuned_models]

            # Perform the Predictions
            perform_predictions(final_models)

            # Plot the Models
            plot_results(final_models)


            print('All Operations have been completed. Closing Program.')

        else:
            logging.error('No data exists.')

    except ValueError as e:
        logging.error(e)
        logging.error('Unable to load the CSV File')