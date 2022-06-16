from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import shutil
import os
from glob import glob
from yellowbrick.regressor import *
from yellowbrick.model_selection import LearningCurve, ValidationCurve, RFECV, FeatureImportances
from yellowbrick.contrib.wrapper import wrap
from sklearn.metrics import mean_squared_error, make_scorer
import logging
import sys
import shap
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import catboost as cb
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


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

    dataset.drop(dataset.index[dataset['ankle'] == 1], inplace=True)

    # Create the dataset that will be used to train the models
    # and the data that will be used to perform the predictions to test the models

    dataset.reset_index(drop=True, inplace=True)

    print('Data for Modeling: ' + str(dataset.shape))

    return dataset


def scale_data(x_train):
    cols_to_scale = ['PatientAge', 'bmi']
    scaler = MinMaxScaler()
    scaler.fit(x_train[cols_to_scale].copy())
    x_train[cols_to_scale] = scaler.transform(x_train[cols_to_scale])

    return x_train


def evaluate_model(regression_model, X_te, y_te, predictions):
    print(
        f"Saving Results for {regression_model} to SciKit_Model_Results.txt")
    with open('SciKit_Model_Results.txt', 'a') as result_file:
        rmse = np.sqrt(mean_squared_error(y_te, predictions))
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


def plot_results(regression_model, x_tr, y_tr, x_te, y_te, model_no):
    """A function that takes in a model and plots the results and saves them to the
    models_results directory """
    current_dir = os.getcwd()
    dst_dir = current_dir + "/Output/models_results"
    plot_types = ['residuals', 'error', 'learning', 'vc', 'feature', 'cooks', 'rfe', 'permutation']
    regression_model = wrap(regression_model)

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
                                               param_range=np.arange(1, 800))
                    visualizer.fit(x_tr, y_tr)
                    visualizer.show(outpath=f"model{model_no + 1}_learning_curve.png", clear_figure=True)

                elif plot == 'vc':
                    visualizer = ValidationCurve(regression_model, scoring='r2',
                                                 param_range=np.linspace(start=0.0, stop=3),
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

        print(f"All Plots for {regression_model} have been created Successfully")

        print(f'Moving plots for {regression_model}')

        # Move the plots to their respective models directory
        files = glob('*.png')
        if len(files) == 0:
            print('There are no Plots to move.')
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
                shutil.move(os.path.join(current_dir, file),
                            os.path.join(dst_dir + f"/scikit_model_{model_no + 1}", file))

    except Exception as er:
        logging.error(er)
        logging.error("Plot Operations were unable to be completed.")


def create_model_set(data, features, target):
    """Splits the Data into the Features you want to train and the target your model will be predicting"""
    copy = data.copy()
    feature_set = copy[features]
    target_column = copy[target]
    return feature_set, target_column


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


def create_shap_sample(data, num_of_instances):
    sample = shap.utils.sample(data, num_of_instances, random_state=120)
    return sample


def create_explainer(model, sample):
    explainer = shap.Explainer(model.predict, sample)
    return explainer


def plot_summary(explainer, data, feature_names):
    shap_values = explainer.shap_values(data)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f'shap_summary.png', )
    plt.clf()

def plot_permutation_importance(model, name, X, y):

    result = permutation_importance(model, X, y, n_repeats=50, scoring=make_scorer(mean_squared_error))
    sorted_importances_idx = result.importances_mean.argsort()

    importance = pd.DataFrame(result.importances_mean.T, columns=X.columns)
    importance.to_csv(f'model{name}_permutation_importance.csv')

    plt.barh(X.columns[sorted_importances_idx], result.importances_mean[sorted_importances_idx].T)
    plt.xlabel('Permutation Importance')
    plt.savefig(f'model{name}_permutation_importance.png')

if __name__ == "__main__":
    try:
        # Get the data from the argument
        file_name = sys.argv[1]
        # file_name = '../Clean_Data_Main.csv'
        print(f'Loading Data {file_name}\n')

        # Perform the analysis and generate the images
        main_data = setup_data(file_name)

    except ValueError as e:
        logging.error(e)
        logging.error('Unable to load the CSV File')

    if main_data is not None:
        X, y = create_model_set(main_data, ['PatientAge', 'PatientGender', 'bmi',
                                            'clavicle', 'shoulder',
                                            'elbow', 'femur', 'wrist',
                                            'tibfib'
                                            ], 'bmdtest_tscore_fn')

        # Split the data into training and testing data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=786, shuffle=True)

        train_data = cb.Pool(X_train, y_train)
        test_data = cb.Pool(X_test, y_test)

        catboost = cb.CatBoostRegressor(loss_function='RMSE')

        grid = {'iterations': [100, 150, 200, 300],
                'learning_rate': [0.03, 0.1],
                'depth': [2, 4, 6, 8],
                'l2_leaf_reg': [0.2, 0.5, 1, 3],
                'early_stopping_rounds': [10, 20, 40, 80, 160]}

        catboost.grid_search(grid, train_data)

        pred = catboost.predict(X_test)
        rmse = (np.sqrt(mean_squared_error(y_test, pred)))
        r2 = r2_score(y_test, pred)
        print('Testing performance')
        print('RMSE: {:.4f}'.format(rmse))
        print('R2: {:.4f}'.format(r2))

        model_explainer = shap.TreeExplainer(catboost)
        plot_summary(model_explainer, X_test, ['PatientAge', 'PatientGender', 'bmi',
                                               'clavicle', 'shoulder',
                                               'elbow', 'femur', 'wrist',
                                               'tibfib'
                                               ])
        evaluate_model(catboost, X_test, y_test, pred)
        plot_results(catboost, X_train, y_train, X_test, y_test, 4)

        # Create a sample to be used with Shap
        X100 = create_shap_sample(X, 100)

        # Create an explainer using the model and the sample values
        model_explainer = create_explainer(catboost, X100)

        # Create Waterfall diagrams of a set of samples from the dataset to explain why they guessed these specific values
        plot_waterfall(X_test, model_explainer, 4)

        plot_permutation_importance(catboost, 'Catboost', X_train, y_train)

    else:
        logging.error('No data exists.')
