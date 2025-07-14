import logging

import pandas as pd


def setup_logging():
    logging.basicConfig(
        level=logging.getLevelName(logging.INFO),
        format=("%(asctime)s [%(threadName)-12.12s]" "[%(levelname)-5.5s] %(message)s"),
        handlers=[
            logging.StreamHandler(),
        ],
    )

    logging.info("SKAO Science Data Challenge Scoring Pipeline")


def load_dataframe(cat_path, columns, skip_n=0):
    """
    Load the catalogue specified by the input path into memory as a pd.DataFrame.

    Drop rows containing NaN values.

    Args:
        cat_path (str): File path of csv catalogue
        skip_n (int) (opt): Number of lines to skip when reading catalogue
    """
    cat_df = pd.read_csv(
        cat_path, skiprows=skip_n, names=columns, delim_whitespace=True
    )

    # Drop NaNs and reset the DataFrame index to avoid missing values
    return cat_df.dropna().reset_index(drop=True)
