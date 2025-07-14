import logging

import numpy as np
from astropy.coordinates import SkyCoord

from ska_sdc.common.utils.constants import expo_to_gauss, las_to_gauss
from ska_sdc.data.data_resources import pb_info_df
from ska_sdc.sdc1.dc_defns import DEC_CENTRE, RA_CENTRE, TRAIN_LIM


def prepare_data(cat_df, freq, train):
    """
    Prepare the submitted and truth catalogues for crossmatch to run against.

    Args:
    """
    cat_df = clean_catalogue(cat_df)
    cat_df = calculate_log_flux(cat_df)
    cat_df_crop = refine_area(cat_df, freq, train)
    cat_df_pb = calculate_pb_values(cat_df_crop, freq)
    cat_df_prep = calculate_conv_size(cat_df_pb, freq)

    return cat_df_prep


def clean_catalogue(cat_df):
    """
    Remove bad values from the passed catalogue DataFrame. Sources with a NaN value,
    or negative value of flux, b_min, b_maj or core_frac will be dropped.
    """
    cat_df = cat_df.dropna().reset_index(drop=True)
    cat_df = drop_negatives(cat_df, "flux")
    cat_df = drop_negatives(cat_df, "core_frac")
    cat_df = drop_negatives(cat_df, "b_min")
    cat_df = drop_negatives(cat_df, "b_maj")

    # Correct for RA degeneracy (truth values lie in the range -180 < RA [deg] < 180)
    cat_df.loc[cat_df["ra_core"] > 180.0, "ra_core"] -= 360.0
    cat_df.loc[cat_df["ra_cent"] > 180.0, "ra_cent"] -= 360.0

    return cat_df


def drop_negatives(cat_df, col_name):
    cat_df_neg = cat_df[cat_df[col_name] < 0]
    if len(cat_df_neg.index) > 0:
        logging.info(
            "Preparation: dropping {} rows with negative {} values.".format(
                len(cat_df_neg.index), col_name
            )
        )
        cat_df = cat_df[cat_df[col_name] > 0].reset_index(drop=True)
    return cat_df


def refine_area(cat_df, freq_value, train=False):
    """
    Crop the dataframe by area to exclude or include the training area.

    The training area limits are different for each frequency.

    Args:
        cat_df (pd.DataFrame): The catalogue DataFrame for which to refine the
            area
        freq_value (int): The current frequency value
        train (bool): True to include only the training area, False to exclude
            the training area
    """

    # Look up RA and Dec limits for the frequency
    lims_freq = TRAIN_LIM.get(freq_value, None)

    ra_min = lims_freq.get("ra_min")
    ra_max = lims_freq.get("ra_max")
    dec_min = lims_freq.get("dec_min")
    dec_max = lims_freq.get("dec_max")

    if train:
        # Include the training area only
        cat_df = cat_df[
            (cat_df["ra_core"] > ra_min)
            & (cat_df["ra_core"] < ra_max)
            & (cat_df["dec_core"] > dec_min)
            & (cat_df["dec_core"] < dec_max)
        ]
    else:
        # Exclude the training area
        cat_df = cat_df[
            (cat_df["ra_core"] < ra_min)
            | (cat_df["ra_core"] > ra_max)
            | (cat_df["dec_core"] < dec_min)
            | (cat_df["dec_core"] > dec_max)
        ]

    # Reset the DataFrame index to avoid missing values
    return cat_df.reset_index(drop=True)


def calculate_pb_values(cat_df, freq_value):
    """
    Calculate the primary beam (PB) values via intermediary pd.Series

    Args:
        cat_df (pd.DataFrame): The catalogue DataFrame for which to exclude the training
            area and calculate new features
        freq_value (int): The current frequency value
    """
    # The beam info file is a rasterized list; define pixel size
    pix_size = (116.4571 * 1400) / freq_value

    # Radial distance from beam centre used to lookup corresponding PB correction
    coord_centre = SkyCoord(ra=RA_CENTRE, dec=DEC_CENTRE, frame="fk5", unit="deg")
    coord_arr = SkyCoord(
        ra=cat_df["ra_core"].values,
        dec=cat_df["dec_core"].values,
        frame="fk5",
        unit="deg",
    )
    sep_arr = coord_centre.separation(coord_arr)
    i_delta = np.around(sep_arr.arcsecond / pix_size)

    # i_delta is the row of the pb_info dataframe corresponding to each cat_df row's
    # distance from the beam centre.
    # Use these indices to look up the value of the "average" column for every
    # source in cat_df.
    # First zero-index the i_delta.
    i_delta_0ind = np.maximum(i_delta - 1, 0)
    pb_corr_series = pb_info_df["average"].reindex(i_delta_0ind)

    # Divide by 1000 to convert mJy -> Jy
    cat_df = cat_df.assign(pb_corr_series=pb_corr_series.values / 1000.0)

    # Add an 'actual' flux column by multiplying the observed flux by the correction
    # factor calculated
    cat_df["a_flux"] = cat_df["flux"] * cat_df["pb_corr_series"]

    return cat_df


def calculate_log_flux(cat_df):
    """
    Create new log(flux) column

    Args:
        cat_df (pd.DataFrame): The catalogue DataFrame for which to calculate log(flux)
    """
    cat_df["log_flux"] = np.log10(cat_df["flux"])
    return cat_df


def calculate_conv_size(cat_df, freq_value):
    """
    Calculate convolved size; this is necessary to control for the potentially
    small Gaussian source sizes, which could yield an unrepresentative
    positional accuracy.

    Thus we calculate the apparent size by convolving with the beam size.

    Args:
        cat_df (pd.DataFrame): The catalogue DataFrame for which to calculate the
            convolved size
        freq_value (int): The current frequency value
    """
    beam_size = (0.25 / freq_value) * 1400

    # We will use a rectangular positional cross-match, so use the greater of the
    # source dimensions
    cat_df["size_max"] = cat_df[["b_maj", "b_min"]].max(axis=1)

    mask_size_3 = cat_df["size"] == 3
    mask_size_1 = cat_df["size"] == 1

    # Approx convolved size by summing the beam size and source size in quadrature
    cat_df["conv_size"] = ((cat_df["size_max"] ** 2) + (beam_size ** 2)) ** 0.5
    cat_df.loc[mask_size_1, "conv_size"] = (
        (((cat_df.loc[mask_size_1, "size_max"]) * las_to_gauss) ** 2) + (beam_size ** 2)
    ) ** 0.5
    cat_df.loc[mask_size_3, "conv_size"] = (
        (((cat_df.loc[mask_size_3, "size_max"]) * expo_to_gauss) ** 2)
        + (beam_size ** 2)
    ) ** 0.5

    return cat_df
