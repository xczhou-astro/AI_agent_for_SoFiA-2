import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from ska_sdc.common.utils.constants import expo_to_gauss, las_to_gauss
from ska_sdc.sdc1.dc_defns import MODE_CORE


def process_kdtree_cand_df(cand_match_df, mode):
    """
    Process the candidate matches yielded by kd tree.
    """

    cand_match_df_multid = calc_multid_err(cand_match_df, mode)

    # In cases where there are more than one possible match; keep only the one with
    # the lowest multi_d_err value
    cand_match_df_multid.sort_values(by=["id", "multi_d_err"], inplace=True)
    cand_match_df_best = cand_match_df_multid.drop_duplicates(
        subset=["id"], keep="first"
    ).reset_index(drop=True)

    # Similar procedure to remove the (rare) situations where a truth catalogue source
    # is matched to more than one submitted source
    cand_match_df_best.sort_values(by=["id_t", "multi_d_err"], inplace=True)
    cand_match_df_sieved = cand_match_df_best.drop_duplicates(
        subset=["id_t"], keep="first"
    ).reset_index(drop=True)

    # Not strictly necessary; reorder by 'id' to match the ingested catalogue order
    cand_match_df_sieved = cand_match_df_sieved.sort_values("id").reset_index(drop=True)

    return cand_match_df_sieved


def calc_multid_err(cand_match_df, mode):
    """
    Calculate the multi-dimensional distance parameter - used to find best match
    in the event there are multiple candidates

    Args:
        cand_match_df (pd.DataFrame): The DataFrame of candidate matches produced
            by the crossmatch step
    """

    # Calculate positional error = separation / conv_size
    if mode == MODE_CORE:
        pos_str = "core"
    else:
        pos_str = "cent"
    coord_sub = SkyCoord(
        ra=cand_match_df["ra_{}".format(pos_str)].values,
        dec=cand_match_df["dec_{}".format(pos_str)].values,
        frame="fk5",
        unit="deg",
    )
    coord_truth = SkyCoord(
        ra=cand_match_df["ra_{}_t".format(pos_str)].values,
        dec=cand_match_df["dec_{}_t".format(pos_str)].values,
        frame="fk5",
        unit="deg",
    )
    sep_arr = coord_truth.separation(coord_sub)

    pos_err_series = sep_arr.arcsecond / cand_match_df["conv_size_t"]

    # Calculate flux error
    flux_err_series = (
        cand_match_df["flux"] - cand_match_df["flux_t"]
    ).abs() / cand_match_df["flux_t"]

    # Calculate size error
    convs_df = pd.DataFrame([las_to_gauss, 1.0, expo_to_gauss])

    size_idx_t_series = cand_match_df["size_id_t"] - 1
    size_idx_series = cand_match_df["size_id"] - 1
    size_conv_t_series = convs_df.reindex(size_idx_t_series)
    size_conv_series = convs_df.reindex(size_idx_series)

    # size_conv_series is a DataFrame with a single column, called '0'.
    size_conv_t_series = size_conv_t_series.reset_index(drop=True)[0]
    size_conv_series = size_conv_series.reset_index(drop=True)[0]

    avg_size_t_series = (cand_match_df["b_maj_t"] + cand_match_df["b_min_t"]) / 2
    avg_size_series = (cand_match_df["b_maj"] + cand_match_df["b_min"]) / 2

    size_err_series = (
        (avg_size_t_series * size_conv_t_series) - (avg_size_series * size_conv_series)
    ).abs() / cand_match_df["conv_size_t"]

    # Define some error factors for normalisation; the factor 3 gives a global
    # 1 sigma error from all attributes.
    norm_pos_err = 0.31 * 3.0
    norm_fl_err = 0.12 * 3.0
    norm_size_err = 1.46 * 3.0

    cand_match_df["multi_d_err"] = np.sqrt(
        (pos_err_series / norm_pos_err) ** 2
        + (flux_err_series / norm_fl_err) ** 2
        + (size_err_series / norm_size_err) ** 2
    )

    return cand_match_df
