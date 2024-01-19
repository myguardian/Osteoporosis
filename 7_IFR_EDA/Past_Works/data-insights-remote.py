#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[52]:


# Check if the file path argument is provided
if len(sys.argv) > 1:
    file_path = sys.argv[1]  # Command line argument for file path

    # Attempt to load the file
    try:
        df = pd.read_csv(file_path)
        print("File loaded successfully!")
        print(df.info())  # Display the first few rows of the dataframe
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
else:
    print("Please provide a file path as a command line argument.")

# Read the CSV file
df = pd.read_csv(file_path)

filtered_df = df[(df['obreak_date'] >= 0) & (df['obreak_date'] <= 730)]
# Removing the first column 'Unnamed: 0'
filtered_df = filtered_df.drop(columns=['Unnamed: 0'])


# In[53]:


# Visualization of key findings from the filtered dataset

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(filtered_df['PatientAge'], kde=True, bins=50)
plt.title('Distribution of Patient Age in Filtered Dataset')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('age_distribution_in_filtered_ds')
plt.show()


# Gender Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='PatientGender', data=filtered_df)
plt.title('Distribution of Patient Gender in Filtered Dataset')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.savefig('Distribution-of-Gender-in-Filtered-Dataset')
plt.show()

# Health Conditions Distribution
plt.figure(figsize=(15, 10))
for i, condition in enumerate(['alcohol', 'arthritis', 'diabetes', 'smoke', 'ptunsteady', 'obreak_hip', 'obreak_spine', 'obreak_pelvis']):
    plt.subplot(3, 3, i+1)
    sns.countplot(x=condition, data=filtered_df)
    plt.title(f'Distribution of {condition.capitalize()}')
    plt.xlabel(condition.capitalize())
    plt.ylabel('Count')
    plt.savefig(f'Distribution_{condition.capitalize()}')

plt.tight_layout()
plt.show()


# Displaying the frequency of each value in the 'PatientAge' column
# Calculating the frequency of each age
age_frequency = filtered_df['PatientAge'].value_counts()

# Displaying the most frequent ages
print(age_frequency)


# In[49]:


# Finding the age group with the most number of obreak_spine

# Filtering the dataset for patients with an obreak_spine
hip_break_df = filtered_df_cleaned[filtered_df_cleaned['obreak_spine'] == 1]

# Counting the frequency of each age in the spine break group
age_group_hip_break = hip_break_df['PatientAge'].value_counts()

# Finding the age group with the highest frequency
most_frequent_age_group_hip_break = age_group_hip_break.idxmax()
most_frequent_age_group_hip_break_count = age_group_hip_break.max()

print("Most Frequent age group with hip break",most_frequent_age_group_hip_break, "and its count", most_frequent_age_group_hip_break_count)


# In[50]:


# Finding the age group with the most number of obreak_pelvis 

# Filtering the dataset for patients with an obreak_pelvis
hip_break_df = filtered_df_cleaned[filtered_df_cleaned['obreak_pelvis'] == 1]

# Counting the frequency of each age in the hip break group
age_group_hip_break = hip_break_df['PatientAge'].value_counts()

# Finding the age group with the highest frequency
most_frequent_age_group_hip_break = age_group_hip_break.idxmax()
most_frequent_age_group_hip_break_count = age_group_hip_break.max()

print("Most Frequent age group with pelvis break",most_frequent_age_group_hip_break, "and its count", most_frequent_age_group_hip_break_count)


# In[51]:


# Finding the age group with the most number of obreak_hip 

# Filtering the dataset for patients with an obreak_hip
hip_break_df = filtered_df_cleaned[filtered_df_cleaned['obreak_hip'] == 1]

# Counting the frequency of each age in the hip break group
age_group_hip_break = hip_break_df['PatientAge'].value_counts()

# Finding the age group with the highest frequency
most_frequent_age_group_hip_break = age_group_hip_break.idxmax()
most_frequent_age_group_hip_break_count = age_group_hip_break.max()

print("Most Frequent age group with hip break",most_frequent_age_group_hip_break, "and its count", most_frequent_age_group_hip_break_count)


# In[43]:


# Creating a box plot or violin plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=filtered_df, x='arthritis', y='obreak_date')
# or sns.violinplot(data=filtered_df_cleaned, x='arthritis', y='obreak_date')
plt.title('Distribution of obreak_date by Arthritis Status')
plt.xlabel('Arthritis')
plt.ylabel('obreak_date')
plt.savefig('Distribution-of-days-by-Arthritis-Status')
plt.show()


# In[41]:


# Creating a box plot or violin plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=filtered_df_cleaned, x='diabetes', y='obreak_date')
# or sns.violinplot(data=filtered_df_cleaned, x='arthritis', y='obreak_date')
plt.title('Distribution of obreak_date by Diabetes Status')
plt.xlabel('Diabetes')
plt.ylabel('obreak_date')
plt.savefig('Distribution-of-days-by-Diabetes-Status')
plt.show()


# In[44]:


# Creating a box plot or violin plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=filtered_df_cleaned, x='smoke', y='obreak_date')
# or sns.violinplot(data=filtered_df_cleaned, x='arthritis', y='obreak_date')
plt.title('Distribution of obreak_date by Smoke Status')
plt.xlabel('Smoke')
plt.ylabel('obreak_date')
plt.savefig('Distribution-of-days-by-Smoke-Status')
plt.show()


# In[ ]:




