import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import sys
import datetime
import logging
from pycaret.classification import *
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index

def setup_data(path):
    dataset = pd.read_csv(path)
    return dataset


if __name__ == "__main__":

    main_data = None 

    try:
        # Get the data from the argument
        file_name = sys.argv[1]
        # file_name = '../Clean_Data_Main.csv'
        logging.info('Loading Data {file_name} \n')

        # Perform the analysis and generate the images
        main_data = setup_data(file_name)
        
    #Local, cant be visible on the pipeline
    except ValueError as e:
        logging.error(e)
        logging.error('Unable to load the CSV File')

    if main_data is not None:
        data = main_data
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




#         [12:52 PM] Doug Sauer
# (py310) inovex_data_admin@CMIAnalysis:~/AnalysisScripts/SurvivalAnalysis$ ./RunClean_Data_Main.sh

# /home/inovex_data_admin/AnalysisScripts/SurvivalAnalysis/IFRdata-cleaning-2.py:42: DtypeWarning: Columns (61) have mixed types. Specify dtype option on import or set low_memory=False.

#   df = pd.read_csv('/home/inovex_data_admin/extracts/V4IFRFullExtractLIVE2023-09-18_2.csv')

# Traceback (most recent call last):

#   File "/home/inovex_data_admin/AnalysisScripts/SurvivalAnalysis/IFRdata-cleaning-2.py", line 63, in <module>

#     df['obreak_date'] = (df['obreak_date'] / pd.Timedelta(days=1)).astype(int)

#   File "/home/inovex_data_admin/anaconda3/envs/py310/lib/python3.10/site-packages/pandas/core/generic.py", line 6240, in astype

#     new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)

#   File "/home/inovex_data_admin/anaconda3/envs/py310/lib/python3.10/site-packages/pandas/core/internals/managers.py", line 448, in astype

#     return self.apply("astype", dtype=dtype, copy=copy, errors=errors)

#   File "/home/inovex_data_admin/anaconda3/envs/py310/lib/python3.10/site-packages/pandas/core/internals/managers.py", line 352, in apply

#     applied = getattr(b, f)(**kwargs)

#   File "/home/inovex_data_admin/anaconda3/envs/py310/lib/python3.10/site-packages/pandas/core/internals/blocks.py", line 526, in astype

#     new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)

#   File "/home/inovex_data_admin/anaconda3/envs/py310/lib/python3.10/site-packages/pandas/core/dtypes/astype.py", line 299, in astype_array_safe

#     new_values = astype_array(values, dtype, copy=copy)

#   File "/home/inovex_data_admin/anaconda3/envs/py310/lib/python3.10/site-packages/pandas/core/dtypes/astype.py", line 230, in astype_array

#     values = astype_nansafe(values, dtype, copy=copy)

#   File "/home/inovex_data_admin/anaconda3/envs/py310/lib/python3.10/site-packages/pandas/core/dtypes/astype.py", line 140, in astype_nansafe

#     return _astype_float_to_int_nansafe(arr, dtype, copy)

#   File "/home/inovex_data_admin/anaconda3/envs/py310/lib/python3.10/site-packages/pandas/core/dtypes/astype.py", line 182, in _astype_float_to_int_nansafe

#     raise IntCastingNaNError(

# pandas.errors.IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer

# (py310) inovex_data_admin@CMIAnalysis:~/AnalysisScripts/SurvivalAnalysis$
# [12:54 PM] Doug Sauer
# (py310) inovex_data_admin@CMIAnalysis:~/AnalysisScripts/SurvivalAnalysis$ ./Execute_Tests.sh /home/inovex_data_admin/extracts/V4IFRFullExtractLIVE2023-09-18_2.csv

# Traceback (most recent call last):

#   File "/home/inovex_data_admin/AnalysisScripts/SurvivalAnalysis/SurvivalAnalysis_V0002.py", line 80, in <module>

#     model.fit(X_train, y_train)

#   File "/home/inovex_data_admin/anaconda3/envs/py310/lib/python3.10/site-packages/sksurv/ensemble/forest.py", line 93, in fit

#     event, time = check_array_survival(X, y)

#   File "/home/inovex_data_admin/anaconda3/envs/py310/lib/python3.10/site-packages/sksurv/util.py", line 188, in check_array_survival

#     event, time = check_y_survival(y)

#   File "/home/inovex_data_admin/anaconda3/envs/py310/lib/python3.10/site-packages/sksurv/util.py", line 155, in check_y_survival

#     yt = check_array(yt, ensure_2d=False)

#   File "/home/inovex_data_admin/anaconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/validation.py", line 921, in check_array

#     _assert_all_finite(

#   File "/home/inovex_data_admin/anaconda3/envs/py310/lib/python3.10/site-packages/sklearn/utils/validation.py", line 161, in _assert_all_finite

#     raise ValueError(msg_err)

# ValueError: Input contains NaN.

# (py310) inovex_data_admin@CMIAnalysis:~/AnalysisScripts/SurvivalAnalysis$
