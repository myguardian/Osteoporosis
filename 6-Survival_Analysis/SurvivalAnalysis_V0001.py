import numpy as np
import pandas as pd
import sys
import logging
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest, ExtraSurvivalTrees
from lifelines.utils import concordance_index
from multiprocess import Pool, cpu_count

# Setup logging
logging.basicConfig(level=logging.INFO)


# Data cleaning logic from snippet 2
def clean_data(filename):
    data = pd.read_csv(filename)
    obreak_date = pd.to_datetime(data.obreak_date)
    datebone = pd.to_datetime(data.datebone)
    y = (abs(datebone - obreak_date))
    X = data.drop(["obreak_date", "datebone"], axis=1)
    selectedColumns = ['PatientAge', "PatientGender", 'parentbreak', 'alcohol',
                       'arthritis', 'diabetes', 'oralster', 'smoke', 'obreak', 'ptunsteady', 'obreak_hip', 'marital',
                       'whereliv', 'ptfall', 'obreak_frac_count', 'specialistReferral', 'fpp_info', 'obreak_spine',
                       'obreak_pelvis']
    dropList = []
    for i in data:
        if data[i].dtypes == 'O':
            dropList.append(data[i].name)
    dropList.append("CompletedSurveyId")
    dropList.append("PatientId")
    X = data.drop(dropList, axis=1)
    X.fillna(0, inplace=True)
    y = pd.DataFrame({"time": y})
    y['event'] = y.time.apply(lambda x: x.days != 0)
    structured_array = y.to_records(index=False)
    swapped = pd.DataFrame({
        "event": y.event,
        "time": y.time.apply(lambda x: x.days)
    })
    swapped.event = swapped.event.astype(bool)
    structured_array = np.rec.array(swapped.to_records(index=False))
    mergedBeforeEncoding = pd.concat([X[selectedColumns], swapped], axis=1)
    cat_features = ['parentbreak', 'alcohol', 'oralster', 'smoke', 'ptunsteady', 'marital', 'whereliv', 'ptfall',
                    'obreak_frac_count', 'specialistReferral', 'fpp_info', 'obreak_spine', 'obreak_pelvis']
    for feature in cat_features:
        if mergedBeforeEncoding is not None:
            if feature in mergedBeforeEncoding.columns:
                cat_one_hot = pd.get_dummies(mergedBeforeEncoding[feature], prefix=f'{feature}', drop_first=False)
                mergedBeforeEncoding = mergedBeforeEncoding.drop(feature, axis=1)
                mergedBeforeEncoding = mergedBeforeEncoding.join(cat_one_hot)
    filterDays = 365  # Default to 1 year
    mergedBeforeEncoding = mergedBeforeEncoding.loc[mergedBeforeEncoding['time'] > filterDays]
    X = mergedBeforeEncoding.drop(['event', 'time'], axis=1)
    y = mergedBeforeEncoding[['event', 'time']]
    y = np.rec.array(y.to_records(index=False))
    return X, y

# Data processing for survival analysis from snippet 3
feature_comparisons = {
    'gradient': {
        'value': 0,
        'features': []
    },
    'random_survival': {
        'value': 0,
        'features': []
    },
    'coxph': {
        'value': 0,
        'features': []
    },
    'est': {
        'value': 0,
        'features': []
    }
}

models = list(feature_comparisons.keys())

def compare_models(X, y, features):
    X_train, X_test, y_train, y_test = train_test_split(X[features], y, test_size=0.4, random_state=42)
    
    # Initialize the GBST survival regressor
    gbst = GradientBoostingSurvivalAnalysis(n_estimators=100, random_state=20)
    gbst.fit(X_train, y_train)
    survival_times_gbst = gbst.predict(X_test)

    # RandomSurvivalForest model
    rsf = RandomSurvivalForest(n_estimators=100, random_state=20, n_jobs=-1)
    rsf.fit(X_train, y_train)
    survival_times_rsf = rsf.predict(X_test)

    # ExtraSurvivalTrees model
    est = ExtraSurvivalTrees(n_estimators=100, random_state=20, n_jobs=-1)
    est.fit(X_train, y_train)
    survival_times_est = est.predict(X_test)

    # CoxPHSurvivalAnalysis model
    y_coxph = np.array(list(zip(y_train['event'], y_train['time'])), dtype=[('event', bool), ('time', float)])
    coxph = CoxPHSurvivalAnalysis(alpha=1e-9)
    coxph.fit(X_train, y_coxph)
    survival_times_coxph = coxph.predict(X_test)

    # Compute the concordance index
    c_index_gradient = concordance_index(y_test['time'], -survival_times_gbst, y_test['event'])
    c_index_rf = concordance_index(y_test['time'], -survival_times_rsf, y_test['event'])
    c_index_coxph = concordance_index(y_test['time'], -survival_times_coxph, y_test['event'])
    c_index_est = concordance_index(y_test['time'], -survival_times_est, y_test['event'])

    return {
        "gradient": c_index_gradient,
        "random_survival": c_index_rf,
        "coxph": c_index_coxph,
        "est": c_index_est
    }

def compare_models_helper(args):
    return compare_models(*args)

def explore_feature_combinations(X, y, max_iterations=8000):
    best_features = []
    best_performance = 0
    best_model_name = ""
    current_features = set()
    remaining_features = set(X.columns)
    num_cores = cpu_count()
    while remaining_features:
        performance_gain = False
        best_gain = 0
        best_feature = None
        with Pool(processes=num_cores) as pool:
            args_list = [(X, y, list(current_features) + [feature]) for feature in remaining_features]
            outcome = pool.map(compare_models_helper, args_list)
        for i, feature in enumerate(remaining_features):
            features_to_try = current_features | {feature}
            X_subset = X[list(features_to_try)]
            for model_name in models:
                if outcome[i][model_name] > best_performance:
                    gain = outcome[i][model_name] - best_performance
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_model_name = model_name
        if best_feature is not None:
            current_features.add(best_feature)
            remaining_features


if __name__ == "__main__":

    X, y = None, None

    try:
        # Get the data from the argument
        file_name = sys.argv[1]
        logging.info(f'Loading Data {file_name} \n')

        # Clean the data
        X, y = clean_data(file_name)

    except ValueError as e:
        logging.error(e)
        logging.error('Unable to load the CSV File')

    if X is not None and y is not None:
        best_model_name, best_features, best_performance = explore_feature_combinations(X, y, max_iterations=len(X.columns))

        with open('best_results.txt', 'w+') as b:
            b.write(f"Best feature combination: {best_features}\n")
            b.write(f"Highest C-index: {best_performance}\n")
            b.write(f"Best Model: {best_model_name}\n")
            for model, comparison in feature_comparisons.items():
                b.write(f"{model}: {comparison}\n")
