import logging
import sys
import os
import math
import pandas as pd
import seaborn as sns
import shap
import numpy as np
import statistics
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
print(tf.__version__)
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from numpy.random import seed
import io


def set_directory():
    # detect the current working directory and add the sub directory
    main_path = os.getcwd()
    absolute_path = main_path + "/L2_results"
    try:
        os.mkdir(absolute_path)
    except OSError:
        logging.info("Creation of the directory %s failed. Folder already exists." % absolute_path)
    else:
        logging.info("Successfully created the directory %s " % absolute_path)
        
# Function to create model, required for KerasRegressor
def build_and_compile_model(norm):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu', kernel_regularizer='l2'),
      layers.Dense(32, activation='relu', kernel_regularizer='l2'),
      #layers.Dense(16, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(0.0001))
  return model

# Function to create model, required for KerasRegressor
def build_and_compile_dropout_model(norm, drop):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation='relu', kernel_regularizer='l2'),
      layers.Dropout(drop, input_shape=(2,)),
      layers.Dense(32, activation='relu', kernel_regularizer='l2'),
      #layers.Dense(16, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(0.0001))
  return model
  
    
def perform_deep_learning_control(path):
   

    try:

        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=1)

        # Load the data from the CSV file and select the features
        data = pd.read_csv(path)
        

        a = ['bmdtest_tscore_fn', 'PatientAge', "PatientGender", 'bmdtest_weight', 'bmdtest_height', "shoulder", "wrist", "heartdisease", "diabetes", "arthritis", "respdisease", "smoke"]
        values = data[a]
        values = values.dropna()
        values = values.apply(pd.to_numeric, errors='coerce').dropna()
        values = values.dropna()
        
                
        b = ['PatientAge', "PatientGender", 'bmdtest_weight', 'bmdtest_height', "shoulder", "wrist", "heartdisease", "diabetes", "arthritis", "respdisease", "smoke"]
        X = values[b]
        d = len(b)
        features = list(X.columns.values)

        y = values['bmdtest_tscore_fn']
        
        #sns.pairplot(values[['bmdtest_tscore_fn', 'PatientAge', "PatientGender", 'bmdtest_weight', 'bmdtest_height', "shoulder", "wrist", "heartdisease", "diabetes", "arthritis", "respdisease", "smoke"]], diag_kind='kde').savefig("L2_results/dropoutPairPlot.png")

        train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.25, random_state=40)

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(X))

        models = []
              
        seed(2)
        tf.random.set_seed(1)
        models.append((f"Control_Dropout20", build_and_compile_dropout_model(normalizer, 0.2))) 
        
        seed(2)
        tf.random.set_seed(1)
        models.append((f"Control_Dropout50", build_and_compile_dropout_model(normalizer,0.5))) 
        
        seed(2)
        tf.random.set_seed(1)
        models.append((f"Control_Full", build_and_compile_model(normalizer))) 
      
    except ValueError as er:
        logging.error(er)
        logging.error('Unable to open csv and create models')
    
    filename = "L2_results/Control_RMSE_Results.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    f = open(filename, 'w+')
    
    try:
        # evaluate each model in turn
        for name, model in models:   
            logging.info(f'Training %s' % (name))
           
            history = model.fit(train_features, train_labels, validation_split=0.2, verbose=0, epochs=1000)

            testResult = model.evaluate(test_features, test_labels, verbose=0)

            rmse = math.sqrt(testResult)
            f.write('%s: RMSE: (%f) \n\n' % (name, rmse))

            test_predictions = model.predict(test_features).flatten()  
            
            #
            # Plot the model adiffrneecd between predicitons and actual values
            #
            plt.figure()
            plt.plot(history.history['loss'], label='loss')
            plt.plot(history.history['val_loss'], label='val_loss')
            plt.ylim([0, 10])
            plt.xlabel('Epoch')
            plt.ylabel('Error [BMD]')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'L2_results/{name}_loss.png')
            plt.clf()
            plt.close
            
            plt.figure()
            a = plt.axes(aspect='equal')
            plt.scatter(test_labels, test_predictions)
            plt.xlabel('True Values [BMD]')
            plt.ylabel('Predictions [BMD]')
            lims = [-5, 3]
            plt.xlim(lims)
            plt.ylim(lims)
            plt.plot(lims, lims)
            plt.savefig(f'L2_results/{name}_Bin.png')
            plt.clf()
            plt.close
            
            plt.figure()
            error = test_predictions - test_labels
            plt.hist(error, bins=25)
            plt.xlabel('Prediction Error [CMD]')
            plt.ylabel('Count')
            plt.savefig(f'L2_results/{name}_error.png')
            plt.clf()
            plt.close
            
            #plt.figure()
            #e = shap.KernelExplainer(model, train_features)
            #shap_values = e.shap_values(test_features)
            #shap.initjs()
            #shap.summary_plot(shap_values[0], test_features, feature_names=features, show=False)
            #plt.savefig(f'L2_results/{name}_ShapSummary.png', bbox_inches = "tight")
            #plt.clf()
            #plt.close
           
                   
        f.close()
        logging.info(f'Deep Learning Analysis Results Complete')
    except ValueError as er:
        logging.error(er)
        logging.error('Unable to create and train deep learning models')
        f.close()

def perform_deep_learning_wrist(path):
   

    try:

        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=1)


        # Load the data from the CSV file and select the features
        data = pd.read_csv(path)
        data = data[data["wrist"]==1]

        a = ['bmdtest_tscore_fn', 'PatientAge', "PatientGender", 'bmdtest_weight', 'bmdtest_height', "heartdisease", "diabetes", "arthritis", "respdisease", "smoke"]
        values = data[a]
        values = values.dropna()
        values = values.apply(pd.to_numeric, errors='coerce').dropna()
        values = values.dropna()
                
        b = ['PatientAge', "PatientGender", 'bmdtest_weight', 'bmdtest_height', "heartdisease", "diabetes", "arthritis", "respdisease", "smoke"]
        X = values[b]
        d = len(b)
        features = list(X.columns.values)

        y = values['bmdtest_tscore_fn']
        
        #sns.pairplot(values[['bmdtest_tscore_fn', 'PatientAge', "PatientGender", 'bmdtest_weight', 'bmdtest_height', "heartdisease", "diabetes", "arthritis", "respdisease", "smoke"]], diag_kind='kde').savefig("L2_results/wristPairPlot.png")

        train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.25, random_state=40)

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(X))

        models = []
        
        seed(2)
        tf.random.set_seed(1)
        models.append((f"Wrist_Dropout20", build_and_compile_dropout_model(normalizer,0.2))) 
        
        seed(2)
        tf.random.set_seed(1)
        models.append((f"Wrist_Dropout50", build_and_compile_dropout_model(normalizer,0.5))) 
        
        seed(2)
        tf.random.set_seed(1)
        models.append((f"Wrist_Full", build_and_compile_model(normalizer))) 
      
    except ValueError as er:
        logging.error(er)
        logging.error('Unable to open csv and create models')
    
    filename = "L2_results/Wrist_RMSE_Results.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    f = open(filename, 'w+')
    
    try:
        # evaluate each model in turn
        for name, model in models:   
           
            logging.info(f'Training %s' % (name))
            history = model.fit(train_features, train_labels, validation_split=0.2, verbose=0, epochs=1000)

            testResult = model.evaluate(test_features, test_labels, verbose=0)

            rmse = math.sqrt(testResult)
            f.write('%s: RMSE: (%f) \n\n' % (name, rmse))

            test_predictions = model.predict(test_features).flatten()   

            
            #
            # Plot the model adiffrneecd between predicitons and actual values
            #
            plt.figure()
            plt.plot(history.history['loss'], label='loss')
            plt.plot(history.history['val_loss'], label='val_loss')
            plt.ylim([0, 10])
            plt.xlabel('Epoch')
            plt.ylabel('Error [BMD]')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'L2_results/{name}_loss.png')
            plt.clf()
            plt.close
            
            plt.figure()
            a = plt.axes(aspect='equal')
            plt.scatter(test_labels, test_predictions)
            plt.xlabel('True Values [BMD]')
            plt.ylabel('Predictions [BMD]')
            lims = [-5, 3]
            plt.xlim(lims)
            plt.ylim(lims)
            plt.plot(lims, lims)
            plt.savefig(f'L2_results/{name}_Bin.png')
            plt.clf()
            plt.close
            
            plt.figure()
            error = test_predictions - test_labels
            plt.hist(error, bins=25)
            plt.xlabel('Prediction Error [CMD]')
            plt.ylabel('Count')
            plt.savefig(f'L2_results/{name}_error.png')
            plt.clf()
            plt.close
                
            #plt.figure()
            #e = shap.KernelExplainer(model, train_features)
            #shap_values = e.shap_values(test_features)
            #shap.initjs()
            #shap.summary_plot(shap_values[0], test_features, feature_names=features, show=False)
            #plt.savefig(f'L2_results/{name}_ShapSummary.png', bbox_inches = "tight")
            #plt.clf()
            #plt.close
            
        f.close()
        logging.info(f'Deep Learning Analysis Results Complete')
    except ValueError as er:
        logging.error(er)
        logging.error('Unable to create and train deep learning models')
        f.close()

def perform_deep_learning_shoulder(path):
   

    try:

        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=1)


        # Load the data from the CSV file and select the features
        data = pd.read_csv(path)
        data = data[data["shoulder"]==1]

        a = ['bmdtest_tscore_fn', 'PatientAge', "PatientGender", 'bmdtest_weight', 'bmdtest_height', "heartdisease", "diabetes", "arthritis", "respdisease", "smoke"]
        values = data[a]
        values = values.dropna()
        values = values.apply(pd.to_numeric, errors='coerce').dropna()
        values = values.dropna()
                
        b = ['PatientAge', "PatientGender", 'bmdtest_weight', 'bmdtest_height', "heartdisease", "diabetes", "arthritis", "respdisease", "smoke"]
        X = values[b]
        d = len(b)
        features = list(X.columns.values)

        y = values['bmdtest_tscore_fn']
        
        #sns.pairplot(values[['bmdtest_tscore_fn', 'PatientAge', "PatientGender", 'bmdtest_weight', 'bmdtest_height', "heartdisease", "diabetes", "arthritis", "respdisease", "smoke"]], diag_kind='kde').savefig("L2_results/shoulderPairPlot.png")

        train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.25, random_state=40)

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(X))

        models = []
        
        seed(2)
        tf.random.set_seed(1)
        models.append((f"Shoulder_Dropout20", build_and_compile_dropout_model(normalizer, 0.2))) 
        
        seed(2)
        tf.random.set_seed(1)
        models.append((f"Shoulder_Dropout50", build_and_compile_dropout_model(normalizer, 0.5))) 
        
        seed(2)
        tf.random.set_seed(1)
        models.append((f"Shoulder_full", build_and_compile_model(normalizer))) 
      
    except ValueError as er:
        logging.error(er)
        logging.error('Unable to open csv and create models')
    
    filename = "L2_results/Shoulder_RMSE_Results.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    f = open(filename, 'w+')
    
    try:
        # evaluate each model in turn
        for name, model in models:   
            logging.info(f'Training %s' % (name))
            history = model.fit(train_features, train_labels, validation_split=0.2, verbose=0, epochs=1000)

            testResult = model.evaluate(test_features, test_labels, verbose=0)

            rmse = math.sqrt(testResult)
            f.write('%s: RMSE: (%f) \n\n' % (name, rmse))

            test_predictions = model.predict(test_features).flatten()   

            
            #
            # Plot the model adiffrneecd between predicitons and actual values
            #
            plt.figure()
            plt.plot(history.history['loss'], label='loss')
            plt.plot(history.history['val_loss'], label='val_loss')
            plt.ylim([0, 10])
            plt.xlabel('Epoch')
            plt.ylabel('Error [BMD]')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'L2_results/{name}_loss.png')
            plt.clf()
            plt.close
            
            plt.figure()
            a = plt.axes(aspect='equal')
            plt.scatter(test_labels, test_predictions)
            plt.xlabel('True Values [BMD]')
            plt.ylabel('Predictions [BMD]')
            lims = [-5, 3]
            plt.xlim(lims)
            plt.ylim(lims)
            plt.plot(lims, lims)
            plt.savefig(f'L2_results/{name}_Bin.png')
            plt.clf()
            plt.close
            
            plt.figure()
            error = test_predictions - test_labels
            plt.hist(error, bins=25)
            plt.xlabel('Prediction Error [CMD]')
            plt.ylabel('Count')
            plt.savefig(f'L2_results/{name}_error.png')
            plt.clf()
            plt.close
                
            #plt.figure()
            #e = shap.KernelExplainer(model, train_features)
            #shap_values = e.shap_values(test_features)
            #shap.initjs()
            #shap.summary_plot(shap_values[0], test_features, feature_names=features, show=False)
            #plt.savefig(f'L2_results/{name}_ShapSummary.png', bbox_inches = "tight")
            #plt.clf()
            #plt.close
            
        f.close()
        logging.info(f'Deep Learning Analysis Results Complete')
    except ValueError as er:
        logging.error(er)
        logging.error('Unable to create and train deep learning models')
        f.close()

def perform_deep_learning_male(path):
   

    try:

        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=1)


        # Load the data from the CSV file and select the features
        data = pd.read_csv(path)
        features = list(data.columns.values)
        data = data[data["PatientGender"]==2]

        a = ['bmdtest_tscore_fn', 'PatientAge', 'bmdtest_weight', 'bmdtest_height', "shoulder", "wrist", "heartdisease", "diabetes", "arthritis", "respdisease", "smoke"]
        values = data[a]
        values = values.dropna()
        values = values.apply(pd.to_numeric, errors='coerce').dropna()
        values = values.dropna()
                
        b = ['PatientAge', 'bmdtest_weight', 'bmdtest_height', "shoulder", "wrist", "heartdisease", "diabetes", "arthritis", "respdisease", "smoke"]
        X = values[b]
        d = len(b)
        features = list(X.columns.values)

        y = values['bmdtest_tscore_fn']
        
        #sns.pairplot(values[['bmdtest_tscore_fn', 'PatientAge', 'bmdtest_weight', 'bmdtest_height', "shoulder", "wrist", "heartdisease", "diabetes", "arthritis", "respdisease", "smoke"]], diag_kind='kde').savefig("L2_results/malePairPlot.png")


        train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.25, random_state=40)

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(X))

        models = []
        
        seed(2)
        tf.random.set_seed(1)
        models.append((f"Male_Dropout20", build_and_compile_dropout_model(normalizer, 0.2))) 
        
        seed(2)
        tf.random.set_seed(1)
        models.append((f"Male_Dropout50", build_and_compile_dropout_model(normalizer, 0.5))) 
        
        seed(2)
        tf.random.set_seed(1)
        models.append((f"Male_Full", build_and_compile_model(normalizer))) 
      
    except ValueError as er:
        logging.error(er)
        logging.error('Unable to open csv and create models')
    
    filename = "L2_results/Male_RMSE_Results.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    f = open(filename, 'w+')
    
    try:
        # evaluate each model in turn
        for name, model in models:   
            logging.info(f'Training %s' % (name))
           
            history = model.fit(train_features, train_labels, validation_split=0.2, verbose=0, epochs=1000)

            testResult = model.evaluate(test_features, test_labels, verbose=0)

            rmse = math.sqrt(testResult)
            f.write('%s: RMSE: (%f) \n\n' % (name, rmse))

            test_predictions = model.predict(test_features).flatten()   

            
            #
            # Plot the model adiffrneecd between predicitons and actual values
            #
            plt.figure()
            plt.plot(history.history['loss'], label='loss')
            plt.plot(history.history['val_loss'], label='val_loss')
            plt.ylim([0, 10])
            plt.xlabel('Epoch')
            plt.ylabel('Error [BMD]')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'L2_results/{name}_loss.png')
            plt.clf()
            plt.close
            
            plt.figure()
            a = plt.axes(aspect='equal')
            plt.scatter(test_labels, test_predictions)
            plt.xlabel('True Values [BMD]')
            plt.ylabel('Predictions [BMD]')
            lims = [-5, 3]
            plt.xlim(lims)
            plt.ylim(lims)
            plt.plot(lims, lims)
            plt.savefig(f'L2_results/{name}_Bin.png')
            plt.clf()
            plt.close
            
            plt.figure()
            error = test_predictions - test_labels
            plt.hist(error, bins=25)
            plt.xlabel('Prediction Error [CMD]')
            plt.ylabel('Count')
            plt.savefig(f'L2_results/{name}_error.png')
            plt.clf()
            plt.close
                
            #plt.figure()
            #e = shap.KernelExplainer(model, train_features)
            #shap_values = e.shap_values(test_features)
            #shap.initjs()
            #shap.summary_plot(shap_values[0], test_features, feature_names=features, show=False)
            #plt.savefig(f'L2_results/{name}_ShapSummary.png', bbox_inches = "tight")
            #plt.clf()
            #plt.close
        f.close()
        logging.info(f'Deep Learning Analysis Results Complete')
    except ValueError as er:
        logging.error(er)
        logging.error('Unable to create and train deep learning models')
        f.close()
        
def perform_deep_learning_female(path):
   

    try:

        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=1)


        # Load the data from the CSV file and select the features
        data = pd.read_csv(path)
        data = data[data["PatientGender"]==1]

        a = ['bmdtest_tscore_fn', 'PatientAge', 'bmdtest_weight', 'bmdtest_height', "shoulder", "wrist", "heartdisease", "diabetes", "arthritis", "respdisease", "smoke"]
        values = data[a]
        values = values.dropna()
        values = values.apply(pd.to_numeric, errors='coerce').dropna()
        values = values.dropna()
                
        b = ['PatientAge', 'bmdtest_weight', 'bmdtest_height', "shoulder", "wrist", "heartdisease", "diabetes", "arthritis", "respdisease", "smoke"]
        X = values[b]
        d = len(b)
        features = list(X.columns.values)

        y = values['bmdtest_tscore_fn']
        
        #sns.pairplot(values[['bmdtest_tscore_fn', 'PatientAge', 'bmdtest_weight', 'bmdtest_height', "shoulder", "wrist", "heartdisease", "diabetes", "arthritis", "respdisease", "smoke"]], diag_kind='kde').savefig("L2_results/femalePairPlot.png")

        train_features, test_features, train_labels, test_labels = train_test_split(X, y, test_size=0.25, random_state=40)

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(X))

        models = []
        
        seed(2)
        tf.random.set_seed(1)
        models.append((f"Female_Dropout20", build_and_compile_dropout_model(normalizer, 0.2))) 
        
        seed(2)
        tf.random.set_seed(1)
        models.append((f"Female_Dropout50", build_and_compile_dropout_model(normalizer, 0.50)))
        
        seed(2)
        tf.random.set_seed(1)
        models.append((f"Female_Full", build_and_compile_model(normalizer))) 
      
    except ValueError as er:
        logging.error(er)
        logging.error('Unable to open csv and create models')
    
    filename = "L2_results/Female_RMSE_Results.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    f = open(filename, 'w+')
    
    try:
        # evaluate each model in turn
        for name, model in models:   
            logging.info(f'Training %s' % (name))
           
            history = model.fit(train_features, train_labels, validation_split=0.2, verbose=0, epochs=1000)

            testResult = model.evaluate(test_features, test_labels, verbose=0)

            rmse = math.sqrt(testResult)
            f.write('%s: RMSE: (%f) \n\n' % (name, rmse))

            test_predictions = model.predict(test_features).flatten()   

            
            #
            # Plot the model adiffrneecd between predicitons and actual values
            #
            plt.figure()
            plt.plot(history.history['loss'], label='loss')
            plt.plot(history.history['val_loss'], label='val_loss')
            plt.ylim([0, 10])
            plt.xlabel('Epoch')
            plt.ylabel('Error [BMD]')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'L2_results/{name}_loss.png')
            plt.clf()
            plt.close
            
            plt.figure()
            a = plt.axes(aspect='equal')
            plt.scatter(test_labels, test_predictions)
            plt.xlabel('True Values [BMD]')
            plt.ylabel('Predictions [BMD]')
            lims = [-5, 3]
            plt.xlim(lims)
            plt.ylim(lims)
            plt.plot(lims, lims)
            plt.savefig(f'L2_results/{name}_Bin.png')
            plt.clf()
            plt.close
            
            plt.figure()
            error = test_predictions - test_labels
            plt.hist(error, bins=25)
            plt.xlabel('Prediction Error [CMD]')
            plt.ylabel('Count')
            plt.savefig(f'L2_results/{name}_error.png')
            plt.clf()
            plt.close
            
            #plt.figure()
            #e = shap.KernelExplainer(model, train_features)
            #shap_values = e.shap_values(test_features)
            #shap.initjs()
            #shap.summary_plot(shap_values[0], test_features, feature_names=features, show=False)
            #plt.savefig(f'L2_results/{name}_ShapSummary.png', bbox_inches = "tight")
            #plt.clf()
            #plt.close
                
        f.close()
        logging.info(f'Deep Learning Analysis Results Complete')
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
        perform_deep_learning_control(file_name)
        perform_deep_learning_shoulder(file_name)
        perform_deep_learning_wrist(file_name)
        perform_deep_learning_male(file_name)
        perform_deep_learning_female(file_name)
        
    except ValueError as e:
        logging.error(e)
        logging.error('Unable to load the CSV File')