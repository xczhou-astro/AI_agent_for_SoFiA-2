import logging

from ska_sdc.common.models.exceptions import (
    BadConfigException,
    InvalidCatalogueException,
)
from ska_sdc.sdc1.dc_defns import CAT_COLUMNS, FREQS


def validate_df(df):
    if any(dt in df.dtypes.values for dt in [str, object]):
        err_msg = (
            "Catalogue contains unsupported data type. "
            "This often occurs if the catalogue header has not been ignored, "
            "or if a string value has been placed."
        )
        logging.error(err_msg)
        raise InvalidCatalogueException(err_msg)
    if df.isna().sum().sum() > 0:
        err_msg = ("Catalogue contains {} NaN values.").format(df.isna().sum().sum())
        logging.error(err_msg)
        raise InvalidCatalogueException(err_msg)
    if list(df.columns) == CAT_COLUMNS:
        return df
    else:
        err_msg = ("Invalid catalogue columns, expected: {}" "\nbut found {}").format(
            ", ".join(CAT_COLUMNS), ", ".join(df.columns)
        )
        logging.error(err_msg)
        raise InvalidCatalogueException(err_msg)


def validate_freq(freq):
    if int(freq) in FREQS:
        return int(freq)
    else:
        err_msg = (
            "Unknown frequency value ({}) for this data challenge. "
            "Please ensure frequencies are given in MHz."
        ).format(freq)
        logging.error(err_msg)
        raise BadConfigException(err_msg)
