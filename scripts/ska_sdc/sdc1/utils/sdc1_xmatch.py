import logging

import numpy as np
from ska_sdc.common.models.exceptions import BadConfigException
from ska_sdc.common.utils.xmatch import KDTreeXMatch
from ska_sdc.sdc1.dc_defns import CAT_COLUMNS, MODE_CENTR, MODE_CORE
from sklearn.neighbors import KDTree


class Sdc1XMatch(KDTreeXMatch):
    """
    Crossmatch sources for the SDC1 scoring use case.
    """

    def get_kdtree(self):
        """
        Get the point K-D tree object. This is constructed using the positions (RA/Dec)
        of sources in the truth catalogue.

        The mode attribute tells the XMatch whether to use core or centroid positions.

        Returns:
            :class:`sklearn.neighbors.KDTree`: k-dimensional tree space partitioning
            data structure.
        """
        if self.mode == MODE_CORE:
            truth_coord_arr = np.array(
                list(
                    zip(
                        self.cat_truth["ra_core"].values,
                        self.cat_truth["dec_core"].values,
                    )
                )
            )
        elif self.mode == MODE_CENTR:
            truth_coord_arr = np.array(
                list(
                    zip(
                        self.cat_truth["ra_cent"].values,
                        self.cat_truth["dec_cent"].values,
                    )
                )
            )
        else:
            err_msg = (
                "Unknown mode, use {}, {} "
                "for core and centroid position modes respectively"
            ).format(MODE_CORE, MODE_CENTR)
            logging.error(err_msg)
            raise BadConfigException(err_msg)

        # Construct k-d tree
        return KDTree(truth_coord_arr)

    def get_query_coords(self):
        """
        Get the submitted catalogue positions to query the KDTree.

        The mode attribute tells the XMatch whether to use core or centroid positions.

        Returns:
            :class:`numpy.array`: The submitted catalogue coordinate pairs, used to
            query the KDTree.
        """
        if self.mode == MODE_CORE:
            sub_coord_arr = np.array(
                list(
                    zip(
                        self.cat_sub["ra_core"].values,
                        self.cat_sub["dec_core"].values,
                    )
                )
            )
        elif self.mode == MODE_CENTR:
            sub_coord_arr = np.array(
                list(
                    zip(
                        self.cat_sub["ra_cent"].values,
                        self.cat_sub["dec_cent"].values,
                    )
                )
            )
        else:
            err_msg = (
                "Unknown mode, use {}, {} "
                "for core and centroid position modes respectively"
            ).format(MODE_CORE, MODE_CENTR)
            logging.error(err_msg)
            raise BadConfigException(err_msg)

        return sub_coord_arr

    def get_radius_arr(self):
        """
        Get the size array that sets the maximum distance a submitted source can lie
        from a truth source to be considered a candidate match. In the SDC1 case this
        is the convolved size of the submitted source.
        """
        size_series = self.cat_sub["conv_size"].astype("float64") * (1 / 3600)
        return size_series.values

    def get_all_col(self):
        """
        Get the column names in truth and submitted catalogues which should be stored in
        the output candidate match dataframe
        """
        return CAT_COLUMNS + [
            "a_flux",
            "conv_size",
        ]

    def refine_match_df(self, cand_match_df):
        """
        Rename size column to size_id for clarity
        """
        return cand_match_df.rename(columns={"size": "size_id", "size_t": "size_id_t"})

    def crossmatch_kdtree(self):
        """
        Explicit declaration of super class method
        """
        return super().crossmatch_kdtree()
