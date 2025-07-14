import numpy as np
from astropy.coordinates import SkyCoord


def get_pos_acc_series(ra_s, dec_s, ra_t_s, dec_t_s, hi_size_t_s, beam_size):
    """
    Calculate positional accuracy series based on passed measured and truth values.

    Args:
        ra_s (pd.Series): Measured RA series.
        dec_s (pd.Series): Measured Dec series.
        ra_t_s (pd.Series): True RA series.
        dec_t_s (pd.Series): True Dec series.
        hi_size_t_s (pd.Series): True HI size series, to estimate source size.
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
    # This matches values from postprocessing
    sep_arr = coord_truth.separation(coord_sub)

    # Estimate source size
    source_size_s = hi_size_t_s

    # Calculate positional accuracy series
    # TODO PH: Action needed? 'This also matches, if factor 4 is removed (calibration
    # factor)'
    pos_acc_s = sep_arr.arcsecond / np.sqrt(4 * beam_size ** 2 + source_size_s ** 2)

    return pos_acc_s


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
    # (for HI, pa is between 0 and 360)
    deg2rad = (2 * np.pi) / 360
    pa_diff = (
        np.arctan2(
            np.sin(pa_s_co.mul(deg2rad) - pa_t_s_co.mul(deg2rad)),
            np.cos(pa_s_co.mul(deg2rad) - pa_t_s_co.mul(deg2rad)),
        )
        / deg2rad
    ).abs()

    return pa_diff


def get_position_scores(pos_acc_series, position_thr, max_score):
    """
    Compute the position scores;  calculate the score, weighted by the set position
    threshold value.

    The maximum score per source is SCORE_MAX

    Args:
        core_acc_series (pd.Series): Positional accuracy (core) series.
        cent_acc_series (pd.Series): Positional accuracy (centroid) series.
    """

    pos_acc_min_series = pos_acc_series  #  for now

    pos_score_frac_series = (max_score / pos_acc_min_series) * position_thr

    return np.minimum(max_score, pos_score_frac_series)


def get_pa_scores(pa_acc_s, pa_thr, max_score):
    """
    Compute the position angle scores; calculate the score, weighted by the set pa
    threshold value.

    Args:
        pa_acc_s (pd.Series): Position angle accuracy series.

    """

    pa_acc_frac_series = (max_score / pa_acc_s) * pa_thr
    scores_pa = np.minimum(max_score, pa_acc_frac_series)

    return scores_pa


def get_i_scores(i_acc_s, i_thr, max_score):
    """
    Compute the inclination angle scores; calculate the score, weighted by the set i
    threshold value.

    Args:
        i_acc_s (pd.Series): i accuracy series.

    """

    i_acc_frac_series = (max_score / i_acc_s) * i_thr

    scores_i = np.minimum(max_score, i_acc_frac_series)

    return scores_i
