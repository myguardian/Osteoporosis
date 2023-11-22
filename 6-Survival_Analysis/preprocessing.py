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
    except:
        print('Error: Could not read the CSV file.')
        return


    # /home/inovex_data_admin/extracts/V4IFRFullExtractLIVE2023-09-18_2.csv

    # Calculate the time difference between two date columns
    first_break = pd.to_datetime(df['obreak_date'])
    second_break = pd.to_datetime(df['datebone'])
    # Should we make sure that this value is not megative, not just use abs?
    df['obreak_date'] = abs(second_break - first_break)


    # Create an event indicator column (1 if the event occurred, 0 if censored)
    df = df[columns]

    # Assuming 'timedelta_column' is the name of the column containing timedeltas
    df['obreak_date'] = (df['obreak_date'] /
                         pd.Timedelta(days=1)).astype(int)  # type: ignore

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

