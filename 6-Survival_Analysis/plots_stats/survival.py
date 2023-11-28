import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter
from datetime import datetime

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

    # Create a directory name with the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = f"./survival_{timestamp}"

    # Create the directory if it does not exist
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    #-Analysis-----------------------------------------------

    ifr = True

    fileNameEnding = ''
    if ifr:
        df = df[(df['obreak_date'] > 0) & (df['obreak_date'] <= 730)]
        fileNameEnding = '_2years_NoZero'


    # Output file
    output_file = open(dir_name + '/survival_analysis_output.txt' + fileNameEnding, 'w')

    # Check for duplicate columns
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    output_file.write(f"Duplicate columns: {duplicate_columns}\n")

    # Cox Proportional Hazards Model
    output_file.write("\n--- Cox Proportional Hazards Model ---\n")
    cph = CoxPHFitter()
    cph.fit(df, duration_col='obreak_date', event_col='obreak')
    cph.print_summary()  # This will print to stdout, not to the file
    ax = cph.plot()
    plt.savefig(dir_name + '/cox_model_coefficients'+fileNameEnding+'.png')  # Save the plot to a file

    # Write the summary to the output file
    cph_summary = cph.summary.to_string()
    output_file.write(f"{cph_summary}\n")

    # Kaplan-Meier Estimator
    output_file.write("\n--- Kaplan-Meier Estimator ---\n")
    kmf = KaplanMeierFitter()
    kmf.fit(df['obreak_date'], event_observed=df['obreak'])

    plt.figure(figsize=(10, 6))
    kmf.plot_survival_function()
    plt.title('Kaplan-Meier Survival Curve')
    plt.xlabel('Time (days)')
    plt.ylabel('Survival Probability')
    plt.savefig(dir_name + '/kaplan_meier_estimator'+fileNameEnding+'.png')  # Save the Kaplan-Meier plot to a file
    plt.close()

    # Write the Kaplan-Meier estimate to the output file
    kmf_summary = kmf.survival_function_.to_string()
    output_file.write(f"{kmf_summary}\n")

    # Accelerated Failure Time Model
    output_file.write("\n--- Accelerated Failure Time Model ---\n")
    aft = WeibullAFTFitter()
    df['obreak_date'] = df['obreak_date'].apply(lambda x: x if x > 0 else 0.01)
    aft.fit(df, duration_col='obreak_date', event_col='obreak')
    aft.print_summary()
    
    plt.figure(figsize=(10, 6))
    aft.plot()
    plt.title('Weibull AFT Model - Cumulative Hazard Function')
    plt.xlabel('Time (days)')
    plt.ylabel('Cumulative Hazard')
    # plt.savefig('weibull_aft_cumulative_hazard.png') # wrong plot
    plt.savefig(dir_name + '/aft_model_coefficients'+fileNameEnding+'.png')
    plt.close()

    # Write the AFT summary to the output file
    aft_summary = aft.summary.to_string()
    output_file.write(f"{aft_summary}\n")

    # Close the output file
    output_file.close()



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


