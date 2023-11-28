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


def main():
    file_name = sys.argv[1]

    selectedColumns = [
        'PatientAge', 'PatientGender', 'parentbreak', 'alcohol',
        'arthritis', 'diabetes', 'obreak_date',
        'oralster', 'smoke', 'obreak', 'ptunsteady',
        'obreak_hip', 'marital', 'whereliv', 'ptfall',
        'obreak_frac_count', 'specialistReferral',
        'fpp_info', 'obreak_spine', 'obreak_pelvis'
    ]

    df = clean_csv_by_column(file_name, selectedColumns)
    if df is None:
        print("Error: DataFrame could not be created. Exiting program.")
        return


    if data is not None:
        # data = file_name
        obreak_date = pd.to_datetime(data.obreak_date)
        datebone = pd.to_datetime(data.datebone)
        y = ( abs( datebone - obreak_date))
        X = pd.DataFrame({
            "PatientAge": data.PatientAge,
            "PatientGender": data.PatientGender,
            
        })


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



#---------------------------------------------
#Cleaning
#---------------------------------------------

def remove_null_rows(df, columns):
    """
    Removes rows with NULL or NaN values in the specified columns.

    Parameters:
    - df: A pandas DataFrame.
    - columns: A list of column names to check for NULL or NaN values.

    Returns:
    A DataFrame with rows containing NULL or NaN values in the specified columns removed.
    """
    # Record the number of rows before removing NaN values
    initial_row_count = df.shape[0]

    # Drop rows with any NaN values in the specified columns
    df_cleaned = df.dropna(subset=columns)

    # Calculate the number of rows removed
    rows_removed = initial_row_count - df_cleaned.shape[0]

    return df_cleaned


def clean_csv_by_column(csv_path, columns):
    """
    Removes not selected colums and cleans selected ones from NULL or NaN values.

    Parameters:
    - csv_path: A csv path to a dataframe.
    - columns: A list of column names to check for NULL or NaN values.

    Returns:
    A cleaned DataFrame without NULL or NaN values, non specified columns removed.
    """

    # Read the CSV file
    # df = pd.read_csv('IFR Extract with selected columns 15-5-23.csv')
    # df = pd.read_csv('IFR_Extract_with_selected_columns_15-5-23.csv')
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f'Error reading CSV file: {e}')
        return None


    # /home/inovex_data_admin/extracts/V4IFRFullExtractLIVE2023-09-18_2.csv

    # Calculate the time difference between two date columns
    first_break = pd.to_datetime(df['datebone'])
    second_break = pd.to_datetime(df['obreak_date'])
    # Should we make sure that this value is not negative, not just use abs?
    df['obreak_date'] = abs(second_break - first_break)


    # Create an event indicator column (1 if the event occurred, 0 if censored)
    df = df[columns]

    # Assuming 'timedelta_column' is the name of the column containing timedeltas
    df['obreak_date'] = (df['obreak_date'] /
                         pd.Timedelta(days=1)).astype(int)  # type: ignore

    # Convert NaN values to integers for specific columns
    columns_to_fill = ['obreak_hip', 'obreak_frac_count',
                       'arthritis', 'diabetes', 'obreak_spine', 'obreak_pelvis']
    df[columns_to_fill] = df[columns_to_fill].fillna(0).astype(int)

    # Check for 'marital' and 'whereliv' and replace null values with -1
    for col in ['marital', 'whereliv']:
        if col in columns and col in df.columns:
            df[col] = df[col].fillna(-1)

    # Save the cleaned data to a new CSV file
    # df.to_csv('new_csv.csv')

    # Display the first few rows of the cleaned DataFrame
    # df.info()
    return df


if __name__ == "__main__":
    main()