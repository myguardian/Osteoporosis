import glob
import pandas as pd
from pathlib import Path
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)


def set_directory():
    # detect the current working directory and add the sub directory
    main_path = os.getcwd()
    absolute_path = main_path + "/merging_results"
    try:
        os.mkdir(absolute_path)
    except OSError:
        logging.info("Creation of the directory %s failed. Folder already exists." % absolute_path)
    else:
        logging.info("Successfully created the directory %s " % absolute_path)


def save_data(path, data):
    path = Path(path)
    data.to_csv(path, index=False)
    logging.info(f'Data saved to {path}\n')


def not_clean_data():
    try:
        logging.info('Creating merged unclean data')
        df_caroc = pd.concat(map(pd.read_csv, glob.glob('Caroc/*.csv')))
        df_frax = pd.concat(map(pd.read_csv, glob.glob('Frax/*.csv')))
        frames = [df_caroc, df_frax]
        df_merged = pd.concat(frames)

        save_data("merging_results/Raw_Not_Cleaned_CAROC_Data.csv", df_caroc)
        save_data("merging_results/Raw_Not_Cleaned_FRAX_Data.csv", df_caroc)
        save_data("merging_results/Raw_Not_Cleaned_Data.csv", df_merged)

    except ValueError as er:
        logging.error('Error making unclean data')


def clean_data():
    pass


if __name__ == "__main__":
    set_directory()
    not_clean_data()
