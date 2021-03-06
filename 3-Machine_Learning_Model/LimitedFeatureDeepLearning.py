import logging
import sys
import os
import math
import pandas as pd
import statistics
import keras
import matplotlib.pyplot as plt
from keras import backend as K
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def set_directory():
    # detect the current working directory and add the sub directory
    main_path = os.getcwd()
    absolute_path = main_path + "/deep_learning_results"
    try:
        os.mkdir(absolute_path)
    except OSError:
        logging.info("Creation of the directory %s failed. Folder already exists." % absolute_path)
    else:
        logging.info("Successfully created the directory %s " % absolute_path)
        
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
        
# Function to create model, required for KerasClassifier
def create_model(layer1, layer2, dimensions):
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim=4, activation='relu'))
    model.add(Dense(layer1, activation='relu'))
    model.add(Dense(layer2, activation='relu'))
    model.add(Dense(1))
    # Compile model
    model.compile(loss=keras.losses.MeanAbsoluteError(), optimizer='adam', metrics=['accuracy'])
    return model
    
    
def perform_deep_learning(path):
   

    try:
        # Load the data from the CSV file and select the features
        data = pd.read_csv(path)
        features = list(data.columns.values)
        features.remove('bmdtest_tscore_fn')
        d = len(features)
        
        a = ['PatientAge', 'bmdtest_height', 'bmdtest_weight', 'cancer']
        X = data[a]
        y = data['bmdtest_tscore_fn']

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)
        logging.info(f'Collecting data form the csv')        
        
        models = []
    
        # create model
        models.append(("Convolutional_30_2_2_1", create_model(2, 2, d))) 
        models.append(("Convolutional_30_7_7_1", create_model(7, 7, d)))        
        models.append(("Convolutional_30_15_15_1", create_model(15, 15, d)))
        models.append(("Convolutional_30_50_50_1", create_model(50, 50, d)))
        models.append(("Convolutional_30_100_100_1", create_model(100, 100, d)))
        models.append(("Convolutional_30_500_500_1", create_model(500, 500, d)))
            
    except ValueError as er:
        logging.error(er)
        logging.error('Unable to create open csv')
    
    
    
    
    
    f = open("deep_learning_results/Limited_Feature_RMSE_Results.txt", 'w+')
    
    try:
        # evaluate each model in turn
        for name, model in models:   
           
            model.fit(X_train, y_train, epochs=10000, verbose=0, validation_split = 0.2)
            predictions = model.predict(X_test)
            rsme = math.sqrt(metrics.mean_squared_error(y_test, predictions))

            f.write('%s: RSME: (%f) -  SI: (%f) \n\n' % (name, rsme, rsme/statistics.mean(y_train)))
            
            errorvariance = [(real-predicted)  for (predicted, real) in zip(predictions, y_test)]
            plt.figure()
            plt.title(name + " Error Levels")
            plt.xlabel("real t-scores")
            plt.ylabel("predicted t-scores")
            plt.scatter(y_test, predictions)
            plt.plot([-6, 3], [-6, 3], '--')
            plt.savefig(f'deep_learning_results/Limited_Feature_{name}_Predictions_VS_Real_Scores.png')
            plt.clf()
            
            logging.info(f'Training %s' % (name))
        f.close()
        logging.info(f'Deep Leanrning Analysis Results Complete')
    except ValueError as er:
        logging.error(er)
        logging.error('Unable to create and train deep learning models')
        f.close()
        
if __name__ == "__main__":

    try:
        # Get the data from the argument
        file_name = sys.argv[1]
        # file_name = 'Clean_Data_Main.csv'
        logging.info(f'Loading Data {file_name}\n')

        # Create the directory where the CSV files and images are going to be saved
        set_directory()

        # Perform the analysis and generate the images
        perform_deep_learning(file_name)
        
    except ValueError as e:
        logging.error(e)
        logging.error('Unable to load the CSV File')