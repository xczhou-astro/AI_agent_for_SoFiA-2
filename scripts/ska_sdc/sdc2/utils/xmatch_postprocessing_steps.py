import numpy as np
from astropy.coordinates import SkyCoord


class XMatchPostprocessingStep:
    """
    Base class for postprocessing steps.
    """

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)


class XMatchPostprocessingStepStub(XMatchPostprocessingStep):
    """
    Stub class for a postprocessing step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute(self):
        """
        Execute the step.

        Returns:
            :class:`pandas.DataFrame`: Processed catalogue.
        """
        # Logic placeholder
        return self.cat


class CalculateMultidErr(XMatchPostprocessingStep):
    """
    Calculate the multi-dimensional distance parameter - used to find best match
    in the event there are multiple candidates

    Args:
        cand_match_df (pd.DataFrame): The DataFrame of candidate matches produced
            by the crossmatch step
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute(self):
        # Calculate positional error = separation / conv_size
        #
        coord_sub = SkyCoord(
            ra=self.cat["ra"].values,
            dec=self.cat["dec"].values,
            frame="fk5",
            unit="deg",
        )
        coord_truth = SkyCoord(
            ra=self.cat["ra_t"].values,
            dec=self.cat["dec_t"].values,
            frame="fk5",
            unit="deg",
        )

        sep_arr = coord_truth.separation(coord_sub)
        pos_err_arr = (sep_arr.arcsecond / self.cat["conv_size_t"]).values

        # Filter so only matches within distance of conv_size_t remain
        cand_match_df = self.cat.loc[pos_err_arr < 1].reset_index(drop=True)

        # Apply same rejection to pos_err_arr (np.array has no index to reset)
        pos_err_arr = pos_err_arr[pos_err_arr < 1]

        # Filter so that only those with positions within distance of spectral_size_t
        # remain.
        # This can be greater than 1 since it is the sub radius search range is but
        # divided by the truth radius.
        freq_err_arr = (
            (cand_match_df["central_freq"] - cand_match_df["central_freq_t"]).abs()
            / cand_match_df["spectral_size_t"]
        ).values

        cand_match_df = cand_match_df.loc[freq_err_arr < 1].reset_index(drop=True)

        # Apply same rejection to error arrays
        pos_err_arr = pos_err_arr[freq_err_arr < 1]
        freq_err_arr = freq_err_arr[freq_err_arr < 1]

        # Calculate flux error
        flux_err_series = (
            cand_match_df["line_flux_integral"] - cand_match_df["line_flux_integral_t"]
        ).abs() / cand_match_df["line_flux_integral_t"]

        # Calculate size error
        size_err_series = (
            cand_match_df["hi_size"] - cand_match_df["hi_size_t"]
        ).abs() / cand_match_df["conv_size_t"]

        w20_err_series = (
            cand_match_df["w20"] - cand_match_df["w20_t"]
        ).abs() / cand_match_df["w20_t"]

        # Define some error factors for normalisation; the factor 3 gives a global
        # 1 sigma error from all attributes.
        # These are taken from real submissions which are not yet available
        # norm_pos_err = 0.31 * 3.0
        # norm_fl_err = 0.12 * 3.0
        # norm_size_err = 1.46 * 3.0
        # norm_freq_err = 0.1 * 3.0

        norm_pos_err = 1  # 0.1 * 3.0
        norm_freq_err = 1  # 0.1 * 3.0
        norm_fl_err = 1  # 0.1 * 3.0
        norm_size_err = 1  # 0.1 * 3.0
        norm_w20_err = 1  # 0.1 * 3.0

        cand_match_df["multi_d_err"] = np.sqrt(
            (pos_err_arr / norm_pos_err) ** 2
            + (freq_err_arr / norm_freq_err) ** 2
            + (flux_err_series / norm_fl_err) ** 2
            + (size_err_series / norm_size_err) ** 2
            + (w20_err_series / norm_w20_err) ** 2
        )

        return cand_match_df


class Sieve(XMatchPostprocessingStep):
    """
    Process the candidate matches yielded by the K-D tree crossmatch step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute(self):
        # In cases where there are more than one possible match; keep only the one with
        # the lowest multi_d_err value
        self.cat.sort_values(by=["id", "multi_d_err"], inplace=True)
        cand_match_df_best = self.cat.drop_duplicates(
            subset=["id"], keep="first"
        ).reset_index(drop=True)

        # Not strictly necessary; reorder by 'id_t' to match the truth catalogue order
        cand_match_df_best = cand_match_df_best.sort_values("id_t").reset_index(
            drop=True
        )
        # Keep all hits that match a truth source, but mark as duplicates
        # for scaling score later
        cand_match_df_best["n_dup"] = cand_match_df_best.groupby(by="id_t",)[
            "pa"
        ].transform(np.size)

        return cand_match_df_best
