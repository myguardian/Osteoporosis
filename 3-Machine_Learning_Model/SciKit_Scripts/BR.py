import logging
# import the os module
import os
import shutil
import sys
from glob import glob
import numpy
import numpy.random as np_random
import matplotlib.pyplot as plt
import pandas as pd
import shap
from hyperopt import hp, fmin, tpe
from hyperopt.pyll import scope
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from yellowbrick.model_selection import LearningCurve, ValidationCurve, RFECV, FeatureImportances
from yellowbrick.regressor import *
from sklearn.inspection import permutation_importance


def setup_data(path):
    dataset = pd.read_csv(path)

    missing = pd.DataFrame(dataset.isnull().sum(), columns=['Total'])
    missing['%'] = (missing['Total'] / dataset.shape[0]) * 100
    missing.sort_values(by='%', ascending=False)

    size = dataset.shape[0]
    dataset = dataset.dropna()

    print(
        'Number of rows in the dataset after the rows with missing values were removed: {}.\n{} rows were removed.'
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

    dataset.drop(dataset.index[dataset['ankle'] == 1], inplace=True)

    print('Number of rows in the dataset after the rows with ankle were removed: {}.\n{} rows were removed.'
          .format(dataset.shape[0], size - dataset.shape[0]))

    # Create the dataset that will be used to train the models
    # and the data that will be used to perform the predictions to test the models

    dataset.reset_index(drop=True, inplace=True)

    print('Data for Modeling: ' + str(dataset.shape))

    return dataset


def encode_cat_data(data):
    cat_features = ['parentbreak', 'alcohol',
                    # 'arthritis', 'cancer', 'diabetes', 'heartdisease',
                    # 'oralster', 'smoke', 'respdisease',
                    'ptunsteady', 'wasfractdue2fall',
                    'ptfall', 'ankle', 'clavicle', 'shoulder', 'elbow', 'femur', 'wrist', 'tibfib']
    dataset = data.copy()

    for feature in cat_features:
        cat_one_hot = pd.get_dummies(dataset[feature], prefix=f'pt_response_{feature}', drop_first=True)
        dataset = dataset.drop(feature, axis=1)
        dataset = dataset.join(cat_one_hot)

    return dataset


def scale_data(x_train, scaler):
    cols_to_scale = ['PatientAge', 'bmi']
    scaler.fit(x_train[cols_to_scale].copy())
    x_train[cols_to_scale] = scaler.transform(x_train[cols_to_scale])

    return x_train


def poly_data(x_train):
    poly = PolynomialFeatures(2, include_bias=False, interaction_only=True)

    x_train = pd.DataFrame(poly.fit_transform(x_train),
                           columns=['PatientAge', 'PatientGender', 'bmi', 'Age*Gender', 'Age*bmi', 'Gender*bmi'])
    x_train.drop(['PatientAge', 'PatientGender', 'bmi'], axis=1, inplace=True)

    return x_train


def evaluate_model(regression_model, train_data, X_te, y_te, predictions):
    coefs = []
    for i in range(train_data.shape[1]):
        coefs.append(f'{train_data.columns[i]}' + '=' + f'{regression_model.coef_[i].round(4)}')
    logging.info(
        f"Saving Results for {regression_model} to SciKit_Model_Results.txt")
    with open('SciKit_Model_Results.txt', 'a') as result_file:
        rmse = numpy.sqrt(mean_squared_error(y_te, predictions))
        result_file.write(
            f"\nModel {regression_model}\n\nHyper Parameters: \n{regression_model.get_params()}\n"
            f"RMSE for Model {regression_model}: \n" +
            f"Root Mean Squared Error: {rmse} \n"
        )
        result_file.write(
            f"R2 for Model {regression_model}:\n" +
            f"R^2: {regression_model.score(X_te, y_te)} \n\n"
        )
        result_file.write(
            "Model Coefficients:\n"
        )
        for coef in coefs:
            result_file.writelines(coef + '\n')


def plot_results(regression_model, x_tr, y_tr, x_te, y_te, model_no):
    """A function that takes in a model and plots the results and saves them to the
    models_results directory """
    current_dir = os.getcwd()
    dst_dir = current_dir + "/Output/models_results"
    plot_types = ['residuals', 'error', 'learning', 'vc', 'feature', 'cooks', 'rfe', 'permutation']

    try:
        # Create plots for the model
        for plot in plot_types:
            try:
                if plot == 'residuals':
                    visualizer = ResidualsPlot(regression_model)
                    visualizer.fit(x_tr, y_tr)
                    visualizer.score(x_te, y_te)
                    visualizer.show(outpath=f"model{model_no + 1}_residuals.png", clear_figure=True)

                elif plot == 'error':
                    visualizer = PredictionError(regression_model)
                    visualizer.fit(x_tr, y_tr)
                    visualizer.score(x_te, y_te)
                    visualizer.show(outpath=f"model{model_no + 1}_prediction_error.png", clear_figure=True)

                elif plot == 'learning':
                    visualizer = LearningCurve(regression_model, scoring='r2', param_name='Training Instances',
                                               param_range=numpy.arange(1, 800))
                    visualizer.fit(x_tr, y_tr)
                    visualizer.show(outpath=f"model{model_no + 1}_learning_curve.png", clear_figure=True)

                elif plot == 'vc':
                    visualizer = ValidationCurve(regression_model, scoring='r2',
                                                 param_range=numpy.linspace(start=0.0, stop=3),
                                                 param_name='alpha_1', cv=10)
                    visualizer.fit(x_te, y_te)
                    visualizer.show(outpath=f"model{model_no + 1}_alpha_validation_curve.png",
                                    clear_figure=True)

                elif plot == 'feature':
                    visualizer = FeatureImportances(regression_model, relative=False)
                    visualizer.fit(x_tr, y_tr)
                    visualizer.show(outpath=f"model{model_no + 1}_feature_importance.png", clear_figure=True)

                elif plot == 'cooks':
                    visualizer = CooksDistance()
                    visualizer.fit(X, y)
                    visualizer.show(outpath=f"model{model_no + 1}_cooks_distance.png", clear_figure=True)

                elif plot == 'permutation':
                    plot_permutation_importance(regression_model, model_no + 1, x_tr, y_tr)

                else:
                    visualizer = RFECV(regression_model)
                    visualizer.fit(x_tr, y_tr)
                    visualizer.show(outpath=f"model{model_no + 1}_recursive_feature_elimination.png", clear_figure=True)

            except Exception as er:
                logging.error(er)
                logging.error(f"'{plot}' plot is not available for this model. Plotting a different graph.")

        logging.info(f"All Plots for {regression_model} have been created Successfully")

        logging.info(f'Moving plots for {regression_model}')

        # Move the plots to their respective models directory
        files = glob('*.png')
        if len(files) == 0:
            logging.info('There are no Plots to move.')
            return
        for file in files:
            if file.endswith('.png'):
                shutil.move(os.path.join(current_dir, file),
                            os.path.join(dst_dir + f"/scikit_model_{model_no + 1}", file))

    except Exception as er:
        logging.error(er)
        logging.error("Plot Operations were unable to be completed.")


def objective_function_regression(estimator):
    rmse_array = cross_val_score(estimator, X_train, y_train, cv=10, n_jobs=-1,
                                 scoring=make_scorer(mean_squared_error))
    return numpy.mean(rmse_array)


def create_search_space():
    scope.define(BayesianRidge)
    alpha_1 = hp.uniform('alpha_1', 1e-10, 1)
    alpha_2 = hp.uniform('alpha_2', 1e-10, 1)
    lambda_1 = hp.uniform('lambda_1', 1e-10, 1)
    lambda_2 = hp.uniform('lambda_2', 1e-10, 1)

    est0 = (0.1, scope.BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2))

    search_space_regression = hp.pchoice('estimator', [est0])

    return search_space_regression


def create_model_set(data, features, target):
    """Splits the Data into the Features you want to train and the target your model will be predicting"""
    copy = data.copy()
    feature_set = copy[features]
    target_column = copy[target]
    return feature_set, target_column


def create_shap_sample(data, num_of_instances):
    sample = shap.utils.sample(data, num_of_instances, random_state=120)
    return sample


def create_explainer(model, sample):
    explainer = shap.Explainer(model.predict, sample)
    return explainer


def plot_waterfall(data, explainer, model_no):
    current_dir = os.getcwd()
    female_data = data[data['PatientGender'] == 1]
    male_data = data[data['PatientGender'] == 2]
    dst_dir = current_dir + "/Output/models_results"
    sample_ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    female_shap_values = explainer(female_data)
    male_shap_values = explainer(male_data)

    for ind in sample_ind:
        shap.plots.waterfall(female_shap_values[ind], show=False)
        plt.tight_layout()
        plt.savefig(f'female_waterfall_sample{ind}.png', )
        plt.clf()

    for ind in sample_ind:
        shap.plots.waterfall(male_shap_values[ind], show=False)
        plt.tight_layout()
        plt.savefig(f'male_waterfall_sample{ind}.png', )
        plt.clf()

    # Move the plots to their respective models directory
    files = glob('*.png')
    if len(files) == 0:
        print('There are no Plots to move.')
        return
    for file in files:
        if file.endswith('.png'):
            shutil.move(os.path.join(current_dir, file),
                        os.path.join(dst_dir + f"/scikit_model_{model_no + 1}" + '/waterfalls', file))


def plot_summary(explainer, data, feature_names):
    shap_values = explainer(data)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f'shap_summary.png', )
    plt.clf()


def plot_permutation_importance(model, name, X, y):

    result = permutation_importance(model, X, y, n_repeats=50, scoring=make_scorer(mean_squared_error))
    sorted_importances_idx = result.importances_mean.argsort()

    importance = pd.DataFrame(result.importances.T, columns=X.columns)
    importance.to_csv(f'model{name}_permutation_importance.csv')

    plt.barh(X.columns[sorted_importances_idx], result.importances_mean[sorted_importances_idx].T)
    plt.xlabel('Permutation Importance')
    plt.savefig(f'model{name}_permutation_importance.png')
    plt.clf()


if __name__ == "__main__":
    try:
        # Get the data from the argument
        file_name = sys.argv[1]
        # file_name = '../Clean_Data_Main.csv'
        logging.info(f'Loading Data {file_name}\n')

        # Perform the analysis and generate the images
        main_data = setup_data(file_name)

    except ValueError as e:
        logging.error(e)
        logging.error('Unable to load the CSV File')

    br = create_search_space()

    if main_data is not None:

        # One-hot encode categorical values we are using for training
        main_data = encode_cat_data(main_data)

        # Create feature set and target column
        X, y = create_model_set(main_data,
                                ['PatientId', 'PatientAge', 'PatientGender', 'bmi', 'bmdtest_height', 'bmdtest_weight',
                                 'pt_response_clavicle_1.0', 'pt_response_shoulder_1.0',
                                 'pt_response_elbow_1.0', 'pt_response_femur_1.0', 'pt_response_wrist_1.0',
                                 'pt_response_tibfib_1.0'
                                 ], 'bmdtest_tscore_fn')
        logging.info('Explanatory Features and Target columns have been created')

        poly_features = X[['PatientAge', 'PatientGender', 'bmi']]
        poly_features = poly_data(poly_features)
        logging.info('Polynomial Features have been created')

        X = pd.concat([X, poly_features], axis=1)
        logging.info('Polynomial Features have been added to the dataset')
        # Scale the Data
        scaler = StandardScaler()
        X = scale_data(X, scaler)
        logging.info('Data has been scaled')
        # Set the Random State for HyperOpt
        rstate = np_random.default_rng(42)
        # Split the data into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=786, shuffle=True)

        temp = X_test

        X_train = X_train.drop(['PatientId', 'bmdtest_weight', 'bmdtest_height'], axis=1)
        X_test = X_test.drop(['PatientId', 'bmdtest_weight', 'bmdtest_height'], axis=1)

        best = fmin(fn=objective_function_regression, space=br, algo=tpe.suggest, max_evals=500, rstate=rstate)

        # Create a regressor model using the optimal choices chosen by the Bayesian Optimization
        regressor = BayesianRidge(alpha_1=best['alpha_1'], alpha_2=best['alpha_2'], lambda_1=best['lambda_1'],
                                  lambda_2=best['lambda_2'])

        # Create a sample to be used with Shap
        X100 = create_shap_sample(X_test, 100)

        # Create an explainer using the model and the sample values
        model_explainer = create_explainer(regressor, X100)

        regressor.fit(X_train, y_train)

        # Create Waterfall diagrams of a set of samples from the dataset to explain why they guessed these specific values
        plot_waterfall(X_test, model_explainer, 0)

        # predict from test set
        yhat = regressor.predict(X_test)
        print("{} {}".format('The RMSE on the test set is :', mean_squared_error(y_test, yhat, squared=False)))

        # perform predictions on the unseen data
        evaluate_model(regressor, X_train, X_test, y_test, yhat)
        plot_summary(model_explainer, X_test, ['PatientAge', 'PatientGender', 'bmi', 'pt_response_clavicle_1.0',
                                               'pt_response_shoulder_1.0',
                                               'pt_response_elbow_1.0', 'pt_response_femur_1.0',
                                               'pt_response_wrist_1.0',
                                               'pt_response_tibfib_1.0', 'Age*Gender', 'Age*bmi', 'Gender*bmi'])
        plot_results(regressor, X_train, y_train, X_test, y_test, 0)
        temp['predicted_t_score'] = yhat
        temp = temp.drop(['pt_response_clavicle_1.0', 'pt_response_shoulder_1.0',
                          'pt_response_elbow_1.0', 'pt_response_femur_1.0', 'pt_response_wrist_1.0',
                          'pt_response_tibfib_1.0'], axis=1)
        temp[['PatientAge', 'bmi']] = scaler.inverse_transform(temp[['PatientAge', 'bmi']])
        temp.to_csv('BR_predictions.csv')

        print('All Operations have been completed. Closing Program.')

    else:
        logging.error('No data exists.')
