# import preprocessing
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def main():

    # Loading-and-cleaning---------------------------------------------
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




    #-Analysis-----------------------------------------------

    # was run last time
    df_noZero = df[df['obreak_date'] != 0]
    calculate_and_save_stats(df['obreak_date'], 'delta_breaks_statistics.txt')

    # Filtering the DataFrame for time differences of no more than 2 years (730 days)
    # filtered_df = df[df['obreak_date'] <= 730]
    # calculate_and_save_stats(filtered_df['obreak_date'], 'delta_breaks_statistics_2years.txt')

    filtered_df = df[(df['obreak_date'] >= 0) & (df['obreak_date'] <= 730)]
    calculate_and_save_stats(filtered_df['obreak_date'], 'delta_breaks_statistics_2years_NoZero.txt')

    # Plotting Delta Break Distribution
    plt.figure(figsize=(20, 12))
    sns.histplot(df_noZero['obreak_date'], bins=50, color='mediumseagreen', kde=True)
    plt.title('Distribution of Time Difference Between Bone Breaks (in days)', fontsize=26)
    plt.xlabel('Time Difference (days)', fontsize=22)
    plt.ylabel('Number of Patients', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    plt.savefig('deltaBreakDistribution.png')


    # Plotting Delta Break Distribution
    plt.figure(figsize=(20, 12))
    
    # Plotting the histogram for the filtered data
    sns.histplot(filtered_df['obreak_date'], bins=50, color='lightpink', kde=True)
    plt.title('Distribution of Time Difference Between Bone Breaks (up to 2 years)', fontsize=26)
    plt.xlabel('Time Difference (days)', fontsize=22)
    plt.ylabel('Number of Patients', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    plt.savefig('deltaBreakDistribution_Upto2Years.png')

    # Plotting the density plot for the filtered data
    plt.figure(figsize=(20, 12))
    sns.kdeplot(filtered_df['obreak_date'], color='lightpink', fill=True)
    plt.title('Density Distribution of Time Difference Between Bone Breaks (up to 2 years)', fontsize=26)
    plt.xlabel('Time Difference (days)', fontsize=22)
    plt.ylabel('Density', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    plt.savefig('deltaBreakDensityDistribution_Upto2Years.png')

    # age to time break
    bins = [0, 40, 50, 60, 70, 80, 90, 100, 110]
    labels = ['0-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99', '100-109']
    df['AgeGroup'] = pd.cut(df['PatientAge'], bins=bins, labels=labels, right=False)

    # box plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='AgeGroup', y='obreak_date', data=df)
    plt.title('Time Between Bone Breaks for Different Age Groups')
    plt.xlabel('Age Group')
    plt.ylabel('Time Between Breaks (days)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('deltaBreakDensityByAge.png')


    # Setting up the figure layout
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Plotting the distribution of patient's age
    sns.histplot(df['PatientAge'], bins=30, kde=True, ax=ax[0])
    ax[0].set_title('Distribution of Patient Age')
    ax[0].set_xlabel('Age')
    ax[0].set_ylabel('Count')

    # Plotting the distribution of patient's gender
    sns.countplot(data=df, x='PatientGender', ax=ax[1])
    ax[1].set_title('Distribution of Patient Gender')
    ax[1].set_xlabel('Gender')
    ax[1].set_ylabel('Count')
    ax[1].set_xticklabels(['Gender 1', 'Gender 2'])

    plt.tight_layout()
    plt.savefig('AgeGenderDistribution.png')


    # df.info()

def calculate_and_save_stats(column, file_name):
    mean = column.mean()
    median = column.median()
    mode = column.mode()[0]  # Assuming one mode, adjust if needed
    zero_count = (column == 0).sum()
    std_dev = column.std()
    variance = column.var()
    range_value = column.max() - column.min()
    iqr = column.quantile(0.75) - column.quantile(0.25)
    skewness = column.skew()
    kurtosis = column.kurt()
    min_value = column.min()
    max_value = column.max()
    rounded_years = round(max_value / 365, 2)

    # Writing to a text file
    with open(file_name, 'w') as file:
        file.write(f"Mean: {mean}\n")
        file.write(f"Median: {median}\n")
        file.write(f"Mode: {mode}\n")
        file.write(f"Number of zero values: {zero_count}\n")
        file.write(f"Standard Deviation: {std_dev}\n")
        file.write(f"Variance: {variance}\n")
        file.write(f"Range: {range_value}\n")
        file.write(f"Interquartile Range (IQR): {iqr}\n")
        file.write(f"Skewness: {skewness}\n")
        file.write(f"Kurtosis: {kurtosis}\n")
        file.write(f"Minimum value: {min_value}\n")
        file.write(f"Maximum value: {max_value} ({rounded_years} years)\n")




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
    first_break = pd.to_datetime(df['obreak_date'])
    second_break = pd.to_datetime(df['datebone'])
    # Should we make sure that this value is not megative, not just use abs?
    df['obreak_date'] = abs(second_break - first_break)


    # Create an event indicator column (1 if the event occurred, 0 if censored)
    df = df[columns]

    # Assuming 'timedelta_column' is the name of the column containing timedeltas
    # df['obreak_date'] = (df['obreak_date'] /
    #                      pd.Timedelta(days=1)).astype(int)  # type: ignore

    # Ensure 'obreak_date' is a timedelta in days, and handle NaN or infinite values
    df['obreak_date'] = df['obreak_date'] / pd.Timedelta(days=1)

    # Replace NaN or infinite values with a default value (e.g., 0) or drop them
    df['obreak_date'] = pd.to_numeric(df['obreak_date'], errors='coerce').fillna(0)

    # Now convert to integer
    df['obreak_date'] = df['obreak_date'].astype(int)


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
    print("Done. Program Finished Successfully.")


