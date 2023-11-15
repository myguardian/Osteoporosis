import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import sys
import datetime
import logging
import csv
from pycaret.classification import *
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index


if __name__ == "__main__":

    file_name = sys.argv[1]

    try:
        with open(file_name, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            data_list = []
            row_number = 0

            for row in reader:
                row_number += 1
                try:
                    # Process each row here. For example:
                    row['obreak_date'] = pd.to_datetime(row['obreak_date'])
                    row['datebone'] = pd.to_datetime(row['datebone'])
                    # Add additional processing as needed

                    # Add the processed row to data_list
                    data_list.append(row)

                except Exception as e:
                    # Log the error but do not exit; continue processing the next row
                    logging.error(f"Error processing row {row_number}: {e}")
                    continue  # Skip the current row and continue with the next one

            # After processing all rows, convert data_list to a DataFrame
            data = pd.DataFrame(data_list)

    except Exception as e:
        logging.error(f"Error opening or reading the CSV file: {e}")
        sys.exit(f"Failed to open/read the CSV file: {e}")

    if data is not None:
        # data = file_name
        obreak_date = pd.to_datetime(data.obreak_date)
        datebone = pd.to_datetime(data.datebone)
        y = ( abs( datebone - obreak_date))
        X = pd.DataFrame({
            "PatientAge": data.PatientAge,
            "PatientGender": data.PatientGender,
            
        })


        dropList = []
        for i in data:
            if data[i].dtypes == 'O':
                dropList.append(data[i].name)
        dropList.append("CompletedSurveyId")
        dropList.append("PatientId")

        print("Columns to drop:", dropList)
        print("Data columns:", data.columns)

        X = data.drop(dropList,axis=1)
        X.fillna(0,inplace=True)
        y = pd.DataFrame({"time":y})


        y['event'] = y.time.apply(lambda x: x.days != 0 )
        structured_array = y.to_records(index=False)

        swapped = pd.DataFrame({
            "event": y.event,
            "time": y.time.apply(lambda x: x.days)
        })
        (swapped.time < 100).value_counts()


        swapped.event = swapped.event.astype(bool)
        swapped.event
        structured_array = np.rec.array(swapped.to_records(index=False))
        X['specialistReferral'].value_counts()       

        X_train, X_test, y_train, y_test = train_test_split(X, structured_array, test_size=0.2, random_state=42)

        # Create an instance of the RandomSurvivalForest model
        model = RandomSurvivalForest(random_state=10)

        # Fit the model on the training data
        model.fit(X_train, y_train)

        # Calculate the baseline performance
        baseline_score = concordance_index(y_test['time'], -model.predict(X_test), y_test['event'])

        # Initialize an array to store the feature importances
        feature_importances = np.zeros(X_train.shape[1])

        # Perform feature importance calculation
        for i in range(X_train.shape[1]):
            # Make a copy of the test set
            X_permuted = X_test.copy()

            # Permute the values of the feature at index i
            X_permuted.iloc[:, i] = np.random.permutation(X_permuted.iloc[:, i])

            # Calculate the permuted score
            permuted_score = concordance_index(y_test['time'], -model.predict(X_permuted), y_test['event'])

            # Calculate the feature importance as the difference between the baseline score and permuted score
            feature_importances[i] = baseline_score - permuted_score

        # Normalize the feature importances
        feature_importances /= np.sum(feature_importances)

        # Print the feature importances
        feature_names = X_train.columns

        #for feature_name, importance in zip(feature_names, feature_importances):
            #print(f"Feature: {feature_name}, Importance: {importance}")

        df = pd.DataFrame()
        for name, importance in zip(feature_names, feature_importances):
            df = pd.concat([df, pd.DataFrame({'Feature Name': [name], 'Feature Importance': [importance]})], ignore_index=True)

        df = df.sort_values('Feature Importance', ascending=False)

        # Calculate the c-index on the test set
        c_index = concordance_index(y_test['time'], -model.predict(X_test), y_test['event'])
        # print("C-index:", c_index)


        deleted_columns = []
        cutoff = 0.1
        while(len(deleted_columns) < 180):
                deleted_columns = []
                for i in range(len(feature_importances)):
                        if feature_importances[i] >= 0 and feature_importances[i] < cutoff:
                                deleted_columns.append(feature_names[i])    
                        elif feature_importances[i] < 0 and feature_importances[i] > -cutoff:
                                deleted_columns.append(feature_names[i]) 
                cutoff += 0.1
        X = data.drop((deleted_columns + dropList),axis=1)
        X = X.fillna(0)


        estimator = CoxPHSurvivalAnalysis()
        estimator.fit(X, structured_array)

        # Calculate the c-index on the test set
        c_index = concordance_index(structured_array['time'], -estimator.predict(X), structured_array['event'])
        print("C-index:", c_index)

        b = open('c_index.txt', 'w+')
        b.write(str(c_index))
        b.close()

        import matplotlib.pyplot as plt

        pred_surv = estimator.predict_survival_function(X.loc[:15])
        time_points = np.arange(1, 1000)
        for i, surv_func in enumerate(pred_surv):
            plt.step(time_points, surv_func(time_points), where="post",
                    label="Sample %d" % (i + 1))
        plt.ylabel("est. probability of survival $\hat{S}(t)$")
        plt.xlabel("time $t$")
        plt.legend(loc="best")
        plt.savefig('survival.png')
