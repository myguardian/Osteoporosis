import CleanSurvivalAnalysis
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest, ExtraSurvivalTrees
from lifelines.utils import concordance_index
from multiprocess import Pool,cpu_count
import numpy as np
import pandas as pd

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
    # Fit the model to the training data
    gbst.fit(X_train, y_train)
    # Predict the survival times for the testing data
    survival_times_gbst = gbst.predict(X_test)

    # Create an instance of the RandomSurvivalForest model
    rsf = RandomSurvivalForest(n_estimators=100, random_state=20, n_jobs=-1)
    # Fit the model on the training data
    rsf.fit(X_train, y_train)
    # Predict the survival times for the testing data
    survival_times_rsf = rsf.predict(X_test)

    est = ExtraSurvivalTrees(n_estimators=100, random_state=20, n_jobs=-1)
    est.fit(X_train, y_train)
    survival_times_est = est.predict(X_test)

    # Prepare the target variable for CoxPHSurvivalAnalysis
    y_coxph = np.array(list(zip(y_train['event'], y_train['time'])), dtype=[('event', bool), ('time', float)])

    # Create an instance of the CoxPHSurvivalAnalysis model with regularization
    coxph = CoxPHSurvivalAnalysis(alpha=1e-9)
    # Fit the model on the training data
    coxph.fit(X_train, y_coxph)
    # Predict the survival times for the testing data
    survival_times_coxph = coxph.predict(X_test)

    # Compute the concordance index to evaluate the model performance
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
    remaining_features = set(X.columns)  # Initialize remaining_features with all column names of X

    num_cores = cpu_count()  # Number of available cores minus 1

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
            remaining_features.remove(best_feature)
            best_performance += best_gain
            performance_gain = True
            for model in models:
                if feature_comparisons[model]['value'] < outcome[i][model]:
                    feature_comparisons[model]['features'] = list(features_to_try)
                    feature_comparisons[model]['value'] = outcome[i][model]

        if not performance_gain or len(current_features) >= max_iterations:
            break

    return best_model_name, list(current_features), best_performance

# Call the function with a maximum of iterations equal to the total number of features
best_model_name, best_features, best_performance = explore_feature_combinations(CleanSurvivalAnalysis.X, CleanSurvivalAnalysis.y, max_iterations=len(CleanSurvivalAnalysis.X.columns))

print("Best feature combination:", best_features)
print("Highest C-index:", best_performance)
print("Best Model:", best_model_name)
for model, comparison in feature_comparisons.items():
    print(model, comparison)