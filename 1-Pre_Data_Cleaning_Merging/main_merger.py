import glob
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)


def not_clean_data():
    try:
        logging.info('Creating merged unclean data')
        df_caroc = pd.concat(map(pd.read_csv, glob.glob('Caroc/*.csv')))
        df_frax = pd.concat(map(pd.read_csv, glob.glob('Frax/*.csv')))
        frames = [df_caroc, df_frax]
        result = pd.concat(frames)

        path = Path("Raw_Not_Cleaned_Data.csv")
        result.to_csv(path, index=False)
        logging.info(f'Data saved to {path}\n')
    except ValueError as er:
        logging.error(er)
        logging.error('Error making unclean data')


if __name__ == "__main__":
    not_clean_data()


