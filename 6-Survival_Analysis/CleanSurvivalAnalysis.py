from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis, RandomSurvivalForest, ExtraSurvivalTrees
from lifelines.utils import concordance_index
from multiprocess import Pool
import numpy as np
import pandas as pd
import random

filename = input("file name: ")
data = pd.read_csv(filename)

obreak_date = pd.to_datetime(data.obreak_date)
datebone = pd.to_datetime(data.datebone)
y = ( abs( datebone - obreak_date))
X = data.drop(["obreak_date","datebone"],axis=1)
selectedColumns = [ 'PatientAge', "PatientGender",'parentbreak', 'alcohol',
                'arthritis', 'diabetes',
                'oralster', 'smoke', 'obreak','ptunsteady','obreak_hip','marital','whereliv','ptfall','obreak_frac_count','specialistReferral','fpp_info','obreak_spine','obreak_pelvis']
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
(swapped.event).value_counts()
swapped.event = swapped.event.astype(bool)
structured_array = np.rec.array(swapped.to_records(index=False))

mergedBeforeEncoding = pd.concat([X[selectedColumns],swapped],axis=1)

cat_features = ['parentbreak', 'alcohol',
                'oralster', 'smoke','ptunsteady','marital','whereliv','ptfall','obreak_frac_count','specialistReferral','fpp_info','obreak_spine','obreak_pelvis'
                # These features were determined to apply minimal impact even
                # 'respdisease', 'hbp','heartdisease',
                # 'ptunsteady', 'wasfractdue2fall', 'cholesterol',
                # 'ptfall', 'shoulder', 'wrist', 'bmdtest_10yr_caroc'
                ]

for feature in cat_features:
    if mergedBeforeEncoding is not None:
        if feature in mergedBeforeEncoding.columns:
            cat_one_hot = pd.get_dummies(mergedBeforeEncoding[feature], prefix=f'{feature}', drop_first=False)
            mergedBeforeEncoding = mergedBeforeEncoding.drop(feature, axis=1)
            mergedBeforeEncoding = mergedBeforeEncoding.join(cat_one_hot)
filterDays = input("Enter the number of days to filter by:")
mergedBeforeEncoding = mergedBeforeEncoding.loc[mergedBeforeEncoding['time'] > int(filterDays)]            
X = mergedBeforeEncoding.drop(['event','time'],axis=1)
y = mergedBeforeEncoding[['event','time']]

y = np.rec.array(y.to_records(index=False))