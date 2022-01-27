import logging
import sys
from keras import backend as K
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
def create_model(layer1, layer2):
    # create model
    model = Sequential()
    model.add(Dense(30, input_dim=30, activation='relu'))
    model.add(Dense(layer1, activation='relu'))
    model.add(Dense(layer2, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='root_mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model
    

    
def perform_deep_learning(path):
   

    try:
        # Load the data from the CSV file and select the features
        data = pd.read_csv(path)
        features = list(data.columns.values)
        
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)
        logging.info(f'Collecting data form the csv')
        
    except ValueError as er:
        logging.error(er)
        logging.error('Unable to create open csv')

    # create model    
    models.add("Convolutional 30/50/50/1", KerasRegressor(build_fn=create_model(50, 50), epochs=50, batch_size=10, verbose=0))
    models.add("Convolutional 30/100/100/1", KerasRegressor(build_fn=create_model(100, 100), epochs=50, batch_size=10, verbose=0))
    models.add("Convolutional 30/500/500/1", KerasRegressor(build_fn=create_model(500, 500), epochs=50, batch_size=10, verbose=0))
    
    
    
    f = open("deep_learning__results/RMSE_Results.txt", "x")
    
    try:
        # evaluate each model in turn
        for name, model in models:   
           
            estimators = []
            estimators.append(('standardize', StandardScaler()))
            estimators.append(('mlp', Model)
            pipeline = Pipeline(estimators)
            kfold = KFold(n_splits=10)
            results = cross_val_score(pipeline, X, Y, cv=kfold)
            f.write("%s Standardized: %.2f +-(%.2f) RMSE \n" % (name, results.mean(), results.std()))
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            rsme = math.sqrt(metrics.mean_squared_error(y_test, predictions))
            f.write('%s: RSME: (%f) -  SI: (%f) \n\n' % (name, rsme, rsme/statistics.mean(y_train)))
            
            f.close()
            
            logging.info(f'Training %s' % (name))
    except ValueError as er:
        logging.error(er)
        logging.error('Unable to create and train deep learning models')

    logging.info(f'Deep Leanrign Analysis Results Complete')
        
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