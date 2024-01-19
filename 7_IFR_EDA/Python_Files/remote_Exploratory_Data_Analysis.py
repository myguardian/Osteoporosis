#!/usr/bin/env python
# coding: utf-8


# Import Libraries
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



filename = input("file path: ")
df = pd.read_csv(filename)
    


# Close Warnings
warnings.filterwarnings("ignore", category=UserWarning)



# Check for NULL values in each column
null_columns = df.columns[df.isnull().all()]

# Save the column names with all NULL values to a text file
output_file_path = 'null_columns.txt'
with open(output_file_path, 'w') as file:
    for column_name in null_columns:
        file.write(column_name + '\n')

print(f"Columns with all NULL values saved to {output_file_path}")



# Check for columns with the same non-null value in each row
same_value_columns = df.columns[(df.nunique(dropna=False) == 1) & (df.notna().all())]

# Save the column names with the same value for each row to a text file
output_file_path = 'same_value_columns.txt'
with open(output_file_path, 'w') as file:
    for column_name in same_value_columns:
        file.write(column_name + '\n')

print(f"Columns with the same value for each row saved to {output_file_path}")



# Count the occurrences of each value in the specified column
column_name = 'knownwt'
value_counts = df[column_name].value_counts()

# Check if 1 and 2 are present in the column
count_1 = value_counts.get(1, 0)
count_2 = value_counts.get(2, 0)

# Total number of rows in the DataFrame
total_rows = len(df)

# Write the results to a text file
output_file_path = 'count_results.txt'
with open(output_file_path, 'w') as file:
    file.write(f"Number of 1's in {column_name}: {count_1}\n")
    file.write(f"Number of 2's in {column_name}: {count_2}\n")
    file.write(f"Total number of rows in the DataFrame: {total_rows}\n")

print(f"Results saved to {output_file_path}")



# Select relevant columns for exploratory data analysis
df = df[['wtcurr_lbs', 
         'wtcurr_kg', 
         'knownwt',
         'obreak_date', 
         'datebone', 
         'PatientAge', 
         'PatientGender', 
         'parentbreak', 
         'smoke', 
         'oralster',
         'arthritis',
         'alcohol',
         'diabetes',
         'respdisease',
         'cancer',
         'DateSurveyed',
         'obreaK_wrist',
         'obreak_elbow',
         'obreak_shoulder',
         'obreak_spine',
         'obreak_pelvis',
         'obreak_hip',
         'obreak_femur',
         'obreak_tibfib',
         'obreak_clavicle',
         'obreak_obone']]



# Create a column for weight in lbs for all patients has weight information.
# 0 means that patient has no weight information.
both_missing = df['wtcurr_lbs'].isnull() & df['wtcurr_kg'].isnull()
missing_lbs = df['wtcurr_lbs'].isnull() & ~df['wtcurr_kg'].isnull()
df.loc[missing_lbs, 'wtcurr_lbs'] = round(df.loc[missing_lbs, 'wtcurr_kg'] * 2.20462)
fill_value = 0
df['wtcurr_lbs'] = df['wtcurr_lbs'].fillna(fill_value).astype('int64')
df.drop(columns=['wtcurr_kg'], inplace=True)



# IFR has 2 years time frame so we need to eliminate records exceeding this time frame.
obreak_date = pd.to_datetime(df['obreak_date'])
datebone = pd.to_datetime(df['datebone'])
df['date_difference'] = (datebone - obreak_date).dt.days
df_filtered = df[df['date_difference'] <= 730]
df_filtered = df_filtered.drop(columns=['date_difference'])



# Exploratory Analysis and Visualization
def saveStats(columnName, fileName):
    risk_factor_stats = df_filtered[columnName].describe()
    risk_factor_stats_str = risk_factor_stats.to_string()
    file_path = f'{fileName}_stats.txt'
    with open(file_path, 'w') as file:
        file.write(risk_factor_stats_str)
    print(f"{fileName}_stats.txt created!")

def saveHistPlot(columnName, labelName, fileName):
    plt.figure(figsize=(10, 6))
    sns.histplot(df_filtered[columnName], bins=30, kde=True, color='skyblue', edgecolor='black')
    plt.xlabel(labelName)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {labelName}')
    plt.savefig(f'{fileName}_distribution.png')
    print(f'{fileName}_distribution.png created!')

def saveBarPlot(specificColumnName, labelName, fileName):
    percentage_values = df_filtered[specificColumnName].value_counts(normalize=True) * 100
    plt.figure(figsize=(10, 6))
    percentage_values.plot(kind='bar', color='skyblue')
    plt.xlabel(f'{labelName}')
    plt.ylabel('Percentage')
    plt.title(f'Percentage Distribution of {labelName}')
    for index, value in enumerate(percentage_values):
        plt.text(index, value + 0.5, f'{value:.2f}%', ha='center', va='bottom')
    plt.savefig(f'{fileName}_percentage_distribution.png') 
    print(f'{fileName}_percentage_distribution.png created!')   



# Age - Risk Factor

# Variables
column_name = 'PatientAge'
label_name = 'Age'
file_name = 'age'

saveStats(column_name, file_name)
saveHistPlot(column_name, label_name, file_name)



# Sex - Risk Factor

# Variables
column_name = 'PatientGender'
label_name = 'Sex'
file_name = 'sex'

saveStats(column_name, file_name)

# Mapping values for visualization
mapping = {1: 'Sex 1', 2: 'Sex 2'}
specific_column = f'{column_name}_mapped'
df_filtered[specific_column] = df_filtered[column_name].map(mapping).fillna('Not Known')

saveBarPlot(specific_column, label_name, file_name)

# Dropping column created for visualization
df_filtered = df_filtered.drop(specific_column, axis=1)



# #Weight - Risk Factor

# Variables
column_name = 'wtcurr_lbs'
label_name = 'Weight'
file_name = 'weight'

df_filtered_knownwt = df_filtered[df_filtered[column_name] != 0]

# Statistics for the risk factor in filtered dataset for IFR to a txt file
risk_factor_stats = df_filtered_knownwt[column_name].describe()
risk_factor_stats_str = risk_factor_stats.to_string()
file_path = f'{file_name}_stats.txt'
with open(file_path, 'w') as file:
    file.write(risk_factor_stats_str)
print(f"{file_name}_stats.txt created!")

# Visualization of Age risk factor in filtered dataset for IFR
plt.figure(figsize=(10, 6))
sns.histplot(df_filtered_knownwt[column_name], bins=30, kde=True, color='skyblue', edgecolor='black')
plt.xlabel(label_name)
plt.ylabel('Frequency')
plt.title(f'Distribution of {label_name}')
plt.savefig(f'{file_name}_distribution.png')
print(f'{file_name}_distribution.png created!')



# Parent Fractured Hip - Risk Factor

# Variables
column_name = 'parentbreak'
label_name = 'Parent Fractured Hip'
file_name = 'parent_fractured_hip'

# Statistics for the risk factor in filtered dataset for IFR to a txt file
saveStats(column_name, file_name)

# Mapping values for visualization
mapping = {0: 'No', 1: 'Yes'}
specific_column = f'{column_name}_mapped'
df_filtered[specific_column] = df_filtered[column_name].map(mapping).fillna('Not Known')

# Visualization of the risk factor in filtered dataset for IFR
saveBarPlot(specific_column, label_name, file_name)

# Dropping column created for visualization
df_filtered = df_filtered.drop(specific_column, axis=1)


# Current Smoking - Risk Factor

# Variables
column_name = 'smoke'
label_name = 'Current Smoking'
file_name = 'current_smoking'

# Statistics for the risk factor in filtered dataset for IFR to a txt file
saveStats(column_name, file_name)

# Mapping values for visualization
mapping = {0: 'No', 1: 'Yes'}
specific_column = f'{column_name}_mapped'
df_filtered[specific_column] = df_filtered[column_name].map(mapping).fillna('Not Known')

# Visualization of the risk factor in filtered dataset for IFR
saveBarPlot(specific_column, label_name, file_name)

# Dropping column created for visualization
df_filtered = df_filtered.drop(specific_column, axis=1)



# Glucocorticoids - Risk Factor

# In[33]:


# Variables
column_name = 'oralster'
label_name = 'Glucocorticoids'
file_name = 'glucocorticoids'

# Statistics for the risk factor in filtered dataset for IFR to a txt file
saveStats(column_name, file_name)

# Mapping values for visualization
mapping = {0: 'No', 1: 'Yes'}
specific_column = f'{column_name}_mapped'
df_filtered[specific_column] = df_filtered[column_name].map(mapping).fillna('Not Known')

# Visualization of the risk factor in filtered dataset for IFR
saveBarPlot(specific_column, label_name, file_name)

# Dropping column created for visualization
df_filtered = df_filtered.drop(specific_column, axis=1)



# Rheumatoid Arthritis - Risk Factor

# Variables
column_name = 'arthritis'
label_name = 'Rheumatoid Arthritis'
file_name = 'rheumatoid_arthritis'

# Statistics for the risk factor in filtered dataset for IFR to a txt file
saveStats(column_name, file_name)

# Mapping values for visualization
mapping = {0: 'No', 1: 'Yes'}
specific_column = f'{column_name}_mapped'
df_filtered[specific_column] = df_filtered[column_name].map(mapping).fillna('No')

# Visualization of the risk factor in filtered dataset for IFR
saveBarPlot(specific_column, label_name, file_name)

# Dropping column created for visualization
df_filtered = df_filtered.drop(specific_column, axis=1)

# Alcohol - Risk Factor

# Variables
column_name = 'alcohol'
label_name = 'Alcohol'
file_name = 'alcohol'

# Statistics for the risk factor in filtered dataset for IFR to a txt file
saveStats(column_name, file_name)

# Mapping values for visualization
mapping = {0: 'No', 1: 'Yes'}
specific_column = f'{column_name}_mapped'
df_filtered[specific_column] = df_filtered[column_name].map(mapping).fillna('Not Known')

# Visualization of the risk factor in filtered dataset for IFR
saveBarPlot(specific_column, label_name, file_name)

# Dropping column created for visualization
df_filtered = df_filtered.drop(specific_column, axis=1)



# Diabetes - Risk Factor

# Variables
column_name = 'diabetes'
label_name = 'Diabetes'
file_name = 'diabetes'

# Statistics for the risk factor in filtered dataset for IFR to a txt file
saveStats(column_name, file_name)

# Mapping values for visualization
mapping = {0: 'No', 1: 'Yes'}
specific_column = f'{column_name}_mapped'
df_filtered[specific_column] = df_filtered[column_name].map(mapping).fillna('No')

# Visualization of the risk factor in filtered dataset for IFR
saveBarPlot(specific_column, label_name, file_name)

# Dropping column created for visualization
df_filtered = df_filtered.drop(specific_column, axis=1)



# Respiratory Disease - Risk Factor

# Variables
column_name = 'respdisease'
label_name = 'Respiratory Disease'
file_name = 'respiratory_disease'

# Statistics for the risk factor in filtered dataset for IFR to a txt file
saveStats(column_name, file_name)

# Mapping values for visualization
mapping = {0: 'No', 1: 'Yes'}
specific_column = f'{column_name}_mapped'
df_filtered[specific_column] = df_filtered[column_name].map(mapping).fillna('No')

# Visualization of the risk factor in filtered dataset for IFR
saveBarPlot(specific_column, label_name, file_name)

# Dropping column created for visualization
df_filtered = df_filtered.drop(specific_column, axis=1)



# Cancer - Risk Factor

# Variables
column_name = 'cancer'
label_name = 'Cancer'
file_name = 'cancer'

# Statistics for the risk factor in filtered dataset for IFR to a txt file
saveStats(column_name, file_name)

# Mapping values for visualization
mapping = {0: 'No', 1: 'Yes'}
specific_column = f'{column_name}_mapped'
df_filtered[specific_column] = df_filtered[column_name].map(mapping).fillna('No')

# Visualization of the risk factor in filtered dataset for IFR
saveBarPlot(specific_column, label_name, file_name)

# Dropping column created for visualization
df_filtered = df_filtered.drop(specific_column, axis=1)



# Distribution of Patient Age at First Fracture

obreak_date = pd.to_datetime(df_filtered['obreak_date'])
date_surveyed = pd.to_datetime(df_filtered['DateSurveyed'])
patient_age = df_filtered['PatientAge']

df_filtered['AgeAtObreak'] = (obreak_date - date_surveyed).dt.days / 365 + patient_age
age_at_obreak = df_filtered['AgeAtObreak']

df_filtered = df_filtered[(age_at_obreak >= 40) & (age_at_obreak <= patient_age.max())]

# Variables
column_name = 'AgeAtObreak'
label_name = 'Patient Age at First Fracture'
file_name = 'age_at_first_fracture'

# Statistics for the risk factor in filtered dataset for IFR to a txt file
saveStats(column_name, file_name)

# Visualization of Age risk factor in filtered dataset for IFR
saveHistPlot(column_name, label_name, file_name)



# Distribution of Days between Fractures

datebone = pd.to_datetime(df_filtered['datebone'])

df_filtered['DaysBtwFracs'] = (datebone - obreak_date).dt.days
days_btw_fracs = df_filtered['DaysBtwFracs']

df_filtered_clean = df_filtered[(days_btw_fracs >= 0) & (days_btw_fracs <= 730)]

# Variables
column_name = 'DaysBtwFracs'
label_name = 'Days between Fractures'
file_name = 'days_between_fractures'

# Statistics for the risk factor in filtered dataset for IFR to a txt file
saveStats(column_name, file_name)

# Visualization of the risk factor in filtered dataset for IFR
plt.figure(figsize=(10, 6))
sns.histplot(df_filtered_clean[column_name], bins=30, kde=True, color='skyblue', edgecolor='black')
plt.xlabel(label_name)
plt.ylabel('Frequency')
plt.title(f'Distribution of {label_name}')
plt.savefig(f'{file_name}_distribution.png')
print(f'{file_name}_distribution.png created!')


# Percentage Distribution of First Fracture Types

df_obreaks = df_filtered[['obreaK_wrist',
                          'obreak_elbow',
                          'obreak_shoulder',
                          'obreak_spine',
                          'obreak_pelvis',
                          'obreak_hip',
                          'obreak_femur',
                          'obreak_tibfib',
                          'obreak_clavicle',
                          'obreak_obone']]

obreak_types_distribution = df_obreaks.apply(lambda col: col.value_counts(dropna=True)).T

obreak_types_percentage = obreak_types_distribution.div(obreak_types_distribution.sum(axis=0)) * 100

# Statistics for the risk factor in filtered dataset for IFR to a txt file
risk_factor_stats = obreak_types_distribution
risk_factor_stats_str = risk_factor_stats.to_string()
file_path = 'first_fracture_types_stats.txt'
with open(file_path, 'w') as file:
    file.write(risk_factor_stats_str)
print(f"first_fracture_types_stats.txt created!")

# Plotting
ax = obreak_types_percentage.plot(kind='bar', stacked=True, figsize=(10, 6))
ax.set_ylabel('Percentage')
ax.set_xlabel('First Fracture Types')
ax.set_title('Percentage Distribution of First Fracture Types')
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.savefig(f'first_fracture_types_distribution.png')
print('first_fracture_types_distribution.png created!')


warnings.resetwarnings()