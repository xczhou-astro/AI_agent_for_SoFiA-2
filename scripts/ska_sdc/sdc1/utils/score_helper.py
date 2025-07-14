import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from ska_sdc.common.utils.constants import (
    expo_to_gauss,
    expo_to_las,
    gauss_to_expo,
    gauss_to_las,
    las_to_expo,
    las_to_gauss,
)
from ska_sdc.sdc1.dc_defns import pa_thr, position_thr, size_thr

# The per-source maximum score
SCORE_MAX = 1.0


def get_pos_acc_series(ra_s, dec_s, ra_t_s, dec_t_s, b_maj_t_s, b_min_t_s, beam_size):
    """
    Calculate positional accuracy series based on passed measured and truth values.

    Args:
        ra_s (pd.Series): Measured RA series.
        dec_s (pd.Series): Measured Dec series.
        ra_t_s (pd.Series): True RA series.
        dec_t_s (pd.Series): True Dec series.
        b_maj_t_s (pd.Series): True major axis series, to estimate source size.
        b_min_t_s (pd.Series): True minor axis series, to estimate source size.
        beam_size (float): The primary beam size
    """
    # Calculate coordinate separation array
    coord_sub = SkyCoord(
        ra=ra_s.values,
        dec=dec_s.values,
        frame="fk5",
        unit="deg",
    )
    coord_truth = SkyCoord(
        ra=ra_t_s.values,
        dec=dec_t_s.values,
        frame="fk5",
        unit="deg",
    )
    sep_arr = coord_truth.separation(coord_sub)

    # Estimate source size
    source_size_s = (b_maj_t_s + b_min_t_s) / 2.0

    # Calculate positional accuracy series
    pos_acc_s = sep_arr.arcsecond / np.sqrt(4 * beam_size ** 2 + source_size_s ** 2)

    return pos_acc_s


def get_size_acc_series(size_s, size_t_s, size_id_s, size_id_t_s):
    """
    Calculate size accuracy series based on measured and truth values, after correcting
    for incorrect size classification.

    Args:
        size_s (pd.Series): Measured size (b_maj or b_min) series.
        size_t_s (pd.Series): True corresponding size series.
        size_id_s (pd.Series): Measured size class series.
        size_id_t_s (pd.Series): True size class series.
    """
    # Copy passed series to suppress SettingWithCopyWarnings
    size_s_co = size_s.copy()

    size_corr_s = correct_size_s(size_s_co, size_id_s, size_id_t_s)
    return (size_corr_s - size_t_s).abs() / size_t_s


def correct_size_s(size_s, size_id_s, size_id_t_s):
    """
    Given the size measurements in size_s, for objects that are incorrectly classified
    in size_id_s (based on the true values in size_id_t_s), correct the apparent sizes
    using defined correction factors.

    Size classification is as follows:
        1 - Largest Angular Scale (LAS)
        2 - Gaussian FWHM
        3 - Exponential

    Args:
        size_s (pd.Series): Measured size (b_maj or b_min) series.
        size_id_s (pd.Series): Measured size class series.
        size_id_t_s (pd.Series): True size class series.
    """

    # LAS -> Gaussian
    mask_12 = (size_id_s == 1) & (size_id_t_s == 2)
    size_s.loc[mask_12] = size_s.loc[mask_12] * las_to_gauss

    # LAS -> Expo
    mask_13 = (size_id_s == 1) & (size_id_t_s == 3)
    size_s.loc[mask_13] = size_s.loc[mask_13] * las_to_expo

    # Gauss -> LAS
    mask_21 = (size_id_s == 2) & (size_id_t_s == 1)
    size_s.loc[mask_21] = size_s.loc[mask_21] * gauss_to_las

    # Gauss -> Expo
    mask_23 = (size_id_s == 2) & (size_id_t_s == 3)
    size_s.loc[mask_23] = size_s.loc[mask_23] * gauss_to_expo

    # Expo -> LAS
    mask_31 = (size_id_s == 3) & (size_id_t_s == 1)
    size_s.loc[mask_31] = size_s.loc[mask_31] * expo_to_las

    # Expo -> Gauss
    mask_32 = (size_id_s == 3) & (size_id_t_s == 2)
    size_s.loc[mask_32] = size_s.loc[mask_32] * expo_to_gauss

    return size_s


def get_pa_acc_series(pa_s, pa_t_s):
    """
    Calculate position angle (PA) accuracy series based on measured and truth values,
    after correcting for angle degeneracies.

    Args:
        pa_s (pd.Series): Measured PA series.
        pa_t_s (pd.Series): True corresponding PA series.
    """
    # Copy passed series to suppress SettingWithCopyWarnings
    pa_s_co = pa_s.copy()
    pa_t_s_co = pa_t_s.copy()

    # Correct for angle degeneracies
    pa_s_co.loc[pa_s_co > 180] -= 180
    pa_s_co.loc[pa_s_co > 90] -= 90
    pa_s_co.loc[pa_s_co > 45] -= 45
    pa_s_co.loc[pa_s_co < -45] += 45

    pa_t_s_co.loc[pa_t_s_co > 180] -= 180
    pa_t_s_co.loc[pa_t_s_co > 90] -= 90
    pa_t_s_co.loc[pa_t_s_co > 45] -= 45
    pa_t_s_co.loc[pa_t_s_co < -45] += 45

    return (pa_s_co - pa_t_s_co).abs()


def get_core_frac_acc_series(core_frac_s, core_frac_t_s):
    """
    Calculate core fraction accuracy series based on measured and truth values.
    The mean core fraction for unresolved AGN is 0.75.

    Args:
        core_frac_s (pd.Series): Measured core fraction series.
        core_frac_t_s (pd.Series): True core fraction series.
    """
    return (core_frac_s - core_frac_t_s).abs() / 0.75


def get_class_acc_series(class_s, class_t_s):
    """
    Calculate classification accuracy series; this is simply 0 or 1 for each source.

    Args:
        class_s (pd.Series): Predicted class series.
        class_t_s (pd.Series): True class series.
    """
    # Initialise scores to zero
    class_acc_series = class_s * 0

    # Set correct classifications to score_max
    class_mask = class_s == class_t_s

    class_acc_series[class_mask] = SCORE_MAX

    return class_acc_series


def get_position_scores(core_acc_series, cent_acc_series):
    """
    Compute the position scores; take the closest match out of core and centroid
    accuracies, and calculate the score, weighted by the set position threshold value.

    The maximum score per source is SCORE_MAX

    Args:
        core_acc_series (pd.Series): Positional accuracy (core) series.
        cent_acc_series (pd.Series): Positional accuracy (centroid) series.
    """
    pos_acc_min_series = pd.concat([cent_acc_series, core_acc_series], axis=1).min(
        axis=1
    )
    pos_score_frac_series = (SCORE_MAX / pos_acc_min_series) * position_thr

    return np.minimum(SCORE_MAX, pos_score_frac_series)


def get_b_min_scores(b_min_acc_s, size_id_t_s):
    """
    Compute the b_min size scores; this differs from b_maj as b_min is not clearly
    defined for steep-spectrum AGN. Scores for this class of object are set to
    SCORE_MAX.

    Args:
        b_min_acc_s (pd.Series): Size accuracy (b_min) series.
        size_id_t_s (pd.Series): True size ID series.
    """
    mask_ssagn = size_id_t_s == 1

    b_min_acc_frac_series = (SCORE_MAX / b_min_acc_s) * size_thr
    scores_b_min = np.minimum(SCORE_MAX, b_min_acc_frac_series)

    scores_b_min.loc[mask_ssagn] = SCORE_MAX

    return scores_b_min


def get_pa_scores(pa_acc_s, size_id_t_s):
    """
    Compute the position angle scores; as with b_min this is not clearly
    defined for steep-spectrum AGN. Scores for this class of object are set to
    SCORE_MAX.

    Args:
        pa_acc_s (pd.Series): Position angle accuracy series.
        size_id_t_s (pd.Series): True size ID series.
    """
    mask_ssagn = size_id_t_s == 1

    pa_acc_frac_series = (SCORE_MAX / pa_acc_s) * pa_thr
    scores_pa = np.minimum(SCORE_MAX, pa_acc_frac_series)

    scores_pa.loc[mask_ssagn] = SCORE_MAX

    return scores_pa
