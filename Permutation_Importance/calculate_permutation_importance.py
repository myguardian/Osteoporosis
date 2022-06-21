import pandas as pd
import math
from operator import itemgetter

if __name__ == '__main__':

    # model names
    scikit_models_names = ['model1', 'model2', 'model3', 'model4']
    dnn_models_names = ['Control_Dropout20', 'Control_Dropout50', 'Control_Full',
                        'Female_Dropout20', 'Female_Dropout50', 'Female_Full',
                        'Male_Dropout20', 'Male_Dropout50', 'Male_Full',
                        'Shoulder_Dropout20', 'Shoulder_Dropout50', 'Shoulder_Full',
                        'Wrist_Dropout20', 'Wrist_Dropout50', 'Wrist_Full']

    scikit_models = []
    dnn_models = []

    # read scikit models csv
    for model in scikit_models_names:
        scikit_models.append(pd.read_csv(f'../3-Machine_Learning_Model/SciKit_Scripts/'
                                         f'{model}_permutation_importance.csv'))

    # change column names to match CB model columns
    for model in scikit_models:
        model.rename(columns={'pt_response_clavicle_1.0': 'clavicle',
                              'pt_response_shoulder_1.0': 'shoulder',
                              'pt_response_elbow_1.0': 'elbow',
                              'pt_response_femur_1.0': 'femur',
                              'pt_response_wrist_1.0': 'wrist',
                              'pt_response_tibfib_1.0': 'tibfib',
                              'pt_response_parentbreak_1.0': 'parentbreak',
                              'pt_response_alcohol_1.0': 'alcohol',
                              'pt_response_arthritis_1.0': 'arthritis',
                              'pt_response_cancer_1.0': 'cancer',
                              'pt_response_diabetes_1.0': 'diabetes',
                              'pt_response_heartdisease_1.0': 'heartdisease',
                              'pt_response_smoke_1.0': 'smoke',
                              'pt_response_respdisease_1.0': 'respdisease',
                              'Age*Gender': 'age_gender',
                              'Age*bmi': 'age_bmi',
                              'Gender*bmi': 'gender_bmi'},
                     inplace=True)
    # import CB and concat
    model5 = pd.read_csv('../3-Machine_Learning_Model/SciKit_Scripts/'
                         'model5_permutation_importance.csv')
    scikit_importance = pd.concat([scikit_models[0], scikit_models[1], scikit_models[2], scikit_models[3], model5])
    scikit_importance.to_csv('scikit_importance.csv')

    # read dnn models csv
    for model in dnn_models_names:
        dnn_models.append(pd.read_csv(f'../4-Deep_Learning_Models/2layer_results/{model}_permutation_importance.csv'))
        dnn_models.append(pd.read_csv(f'../4-Deep_Learning_Models/L1_results/{model}_permutation_importance.csv'))
        dnn_models.append(pd.read_csv(f'../4-Deep_Learning_Models/L2_results/{model}_permutation_importance.csv'))
        dnn_models.append(pd.read_csv(f'../4-Deep_Learning_Models/simplified_results/{model}_permutation_importance.csv'))

    dnn_importance = pd.concat(dnn_models)
    # dropping shoulder and wrist
    dnn_importance = dnn_importance.drop(['shoulder', 'wrist'], axis=1)
    dnn_importance.to_csv('dnn_importance.csv')

    # sci mean
    sci_age = scikit_importance.PatientAge.mean()
    sci_gender = scikit_importance.PatientGender.mean()
    sci_bmi = scikit_importance.bmi.mean()
    sci_clavicle = scikit_importance.clavicle.mean()
    sci_shoulder = scikit_importance.shoulder.mean()
    sci_elbow = scikit_importance.elbow.mean()
    sci_femur = scikit_importance.femur.mean()
    sci_wrist = scikit_importance.wrist.mean()
    sci_tibfib = scikit_importance.tibfib.mean()
    sci_parentbreak = scikit_importance.parentbreak.mean()
    sci_alcohol = scikit_importance.alcohol.mean()
    sci_arthritis = scikit_importance.arthritis.mean()
    sci_cancer = scikit_importance.cancer.mean()
    sci_diabetes = scikit_importance.diabetes.mean()
    sci_heartdisease = scikit_importance.heartdisease.mean()
    sci_smoke = scikit_importance.smoke.mean()
    sci_respdisease = scikit_importance.respdisease.mean()
    sci_age_gender = scikit_importance.age_gender.mean()
    sci_age_bmi = scikit_importance.age_bmi.mean()
    sci_gender_bmi = scikit_importance.gender_bmi.mean()

    sci_mean = [['sci_age', sci_age], ['sci_gender', sci_gender], ['sci_bmi', sci_bmi],
                ['sci_clavicle', sci_clavicle], ['sci_shoulder', sci_shoulder], ['sci_elbow', sci_elbow],
                ['sci_femur', sci_femur], ['sci_wrist', sci_wrist], ['sci_tibfib', sci_tibfib],
                ['sci_parentbreak', sci_parentbreak], ['sci_alcohol', sci_alcohol],
                ['sci_arthritis', sci_arthritis], ['sci_cancer', sci_cancer],
                ['sci_heartdisease', sci_heartdisease], ['sci_smoke', sci_smoke],
                ['sci_respdisease', sci_respdisease], ['sci_age_gender', sci_age_gender],
                ['sci_age_bmi', sci_age_bmi], ['sci_gender_bmi', sci_gender_bmi]]

    for idx, data in enumerate(sci_mean):
        sci_mean[idx] = [data[0], math.sqrt(abs(data[1]))]

    # dnn mean
    dnn_age = dnn_importance.PatientAge.mean()
    dnn_gender = dnn_importance.PatientGender.mean()
    dnn_weight = dnn_importance.bmdtest_weight.mean()
    dnn_height = dnn_importance.bmdtest_height.mean()
    dnn_heartdisease = dnn_importance.heartdisease.mean()
    dnn_diabetes = dnn_importance.heartdisease.mean()
    dnn_arthritis = dnn_importance.arthritis.mean()
    dnn_respdisease = dnn_importance.respdisease.mean()
    dnn_smoke = dnn_importance.smoke.mean()
    dnn_alcohol = dnn_importance.alcohol.mean()
    dnn_oralster = dnn_importance.oralster.mean()

    dnn_mean = [['dnn_age', dnn_age], ['dnn_gender', dnn_gender], ['dnn_weight', dnn_weight],
                ['dnn_height', dnn_height], ['dnn_heartdisease', dnn_heartdisease], ['dnn_diabetes', dnn_diabetes],
                ['dnn_arthritis', dnn_arthritis], ['dnn_respdisease', dnn_respdisease],
                ['dnn_smoke', dnn_smoke], ['dnn_alcohol', dnn_alcohol], ['dnn_oralster', dnn_oralster]]

    for idx, data in enumerate(dnn_mean):
        dnn_mean[idx] = [data[0], math.sqrt(abs(data[1]))]

    # sort the means
    sorted_sci_mean = sorted(sci_mean, key=itemgetter(1), reverse=True)
    sorted_dnn_mean = sorted(dnn_mean, key=itemgetter(1), reverse=True)

    # sorted means to text files
    try:
        with open('sorted_sci_mean.txt', 'w') as f:
            for mean in sorted_sci_mean:
                f.write('%s\n' % mean)
    except FileNotFoundError:
        print('The filepath does not exist')

    try:
        with open('sorted_dnn_mean.txt', 'w') as f:
            for mean in sorted_dnn_mean:
                f.write('%s\n' % mean)
    except FileNotFoundError:
        print('The filepath does not exist')