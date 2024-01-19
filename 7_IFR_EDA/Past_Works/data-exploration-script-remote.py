#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ask the user for a file path
file_path = input("Please enter the file path: ")
df = pd.read_csv(file_path)


# In[2]:


# Remove three columns as index base
df.drop(df.columns[[0]], axis=1, inplace=True)

# Display basic information and first few rows of the dataframe
print(df.info())

# Basic statistics for numerical columns
print(df.describe())


# In[6]:


# Distribution of key variables like 'PatientAge', 'PatientGender'
plt.figure(figsize=(10, 5))
sns.histplot(df['PatientAge'], kde=True)
plt.title('Distribution of Patient Age')
plt.savefig('distribution_patient_age.png')
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(x='PatientGender', data=df)
plt.title('Distribution of Patient Gender')
plt.savefig('distribution_patient_gender.png')
plt.show()


# In[7]:


# Exploring health conditions
health_conditions = ['alcohol', 'arthritis', 'diabetes', 'smoke', 'ptunsteady']
for condition in health_conditions:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=condition, data=df)
    plt.title(f'Distribution of {condition.capitalize()}')
    plt.savefig('distribution of {condition.capitalize()}.png')
    plt.show()


# In[8]:


# Correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




