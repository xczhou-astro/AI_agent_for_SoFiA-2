import logging

from ska_sdc.common.models.exceptions import (
    BadConfigException,
    InvalidCatalogueException,
)


class Validation:
    @staticmethod
    def is_valid_df(df, cat_column_names):
        if any(dt in df.dtypes.values for dt in [str, object]):
            err_msg = (
                "Catalogue contains unsupported data type. "
                "This often occurs if the catalogue header has not been ignored, "
                "or if a string value has been placed."
            )
            logging.error(err_msg)
            raise InvalidCatalogueException(err_msg)
        if df.isna().sum().sum() > 0:
            err_msg = ("Catalogue contains {} NaN values.").format(
                df.isna().sum().sum()
            )
            logging.error(err_msg)
            raise InvalidCatalogueException(err_msg)
        if list(df.columns) == cat_column_names:
            return df
        else:
            err_msg = (
                "Invalid catalogue columns, expected: {}" "\nbut found {}"
            ).format(", ".join(cat_column_names), ", ".join(df.columns))
            logging.error(err_msg)
            raise InvalidCatalogueException(err_msg)

    @staticmethod
    def is_valid_config(config):
        try:
            config["general"]["sub_cat_column_names"]
            # TODO PH: These properties in the config file are never actually used
            # anywhere; can they be removed?
            # config["cube"]["spatial_resolution"]
            # config["cube"]["frequency_resolution"]
            config["cube"]["beam_size"]
            config["cube"]["rest_freq"]
            config["cube"]["field_centre_ra"]
            config["cube"]["field_centre_dec"]
            config["threshold"]["multid_thr"]
            config["threshold"]["position_thr"]
            config["threshold"]["central_freq_thr"]
            config["threshold"]["flux_thr"]
            config["threshold"]["size_thr"]
            config["threshold"]["pa_thr"]
            config["threshold"]["w20_thr"]
            config["threshold"]["i_thr"]
            config["score"]["max_score"]
        except KeyError:
            raise BadConfigException
