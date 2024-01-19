#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8

import pandas as pd

# Function to remove rows with NULL or NaN values in specified columns
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

# Read the CSV file
df = pd.read_csv('IFR Extract with selected columns 15-5-23.csv')

#/home/inovex_data_admin/extracts/V4IFRFullExtractLIVE2023-09-18_2.csv

# Calculate the time difference between two date columns
obreak_date = pd.to_datetime(df['obreak_date'])
date_of_birth = pd.to_datetime(df['datebone'])
df['obreak_date'] = abs(date_of_birth - obreak_date)

# Select and clean the desired columns
selectedColumns = [
    'PatientAge', 'PatientGender', 'parentbreak', 'alcohol',
    'arthritis', 'diabetes', 'obreak_date',
    'oralster', 'smoke', 'obreak', 'ptunsteady',
    'obreak_hip', 'marital', 'whereliv', 'ptfall',
    'obreak_frac_count', 'specialistReferral',
    'fpp_info', 'obreak_spine', 'obreak_pelvis'
]

# Create an event indicator column (1 if the event occurred, 0 if censored)
df = df[selectedColumns]

# Assuming 'timedelta_column' is the name of the column containing timedeltas
df['obreak_date'] = (df['obreak_date'] / pd.Timedelta(days=1)).astype(int)

# Convert NaN values to integers for specific columns
columns_to_fill = ['obreak_hip', 'obreak_frac_count', 'arthritis', 'diabetes', 'obreak_spine', 'obreak_pelvis']
df[columns_to_fill] = df[columns_to_fill].fillna(0).astype(int)

# Remove rows with NULL values in 'marital' and 'whereliv' columns
df = remove_null_rows(df, ['marital', 'whereliv'])

# Save the cleaned data to a new CSV file
df.to_csv('new_csv.csv')

#Display the first few rows of the cleaned DataFrame
df.info()


# In[ ]:





# In[ ]:




