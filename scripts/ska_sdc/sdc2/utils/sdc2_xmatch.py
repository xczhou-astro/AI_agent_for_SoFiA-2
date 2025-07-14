import numpy as np
from ska_sdc.common.utils.xmatch import KDTreeXMatch
from sklearn.neighbors import KDTree


class Sdc2XMatch(KDTreeXMatch):
    """
    Crossmatch sources for the SDC2 scoring use case.
    """

    def get_kdtree(self):
        """
        Get the point K-D tree object. This is constructed considering the positions and
        frequency of sources in the truth catalogue.

        Returns:
            :class:`sklearn.neighbors.KDTree`: k-dimensional tree space partitioning
            data structure.
        """
        truth_coord_arr = np.array(
            list(
                zip(
                    self.cat_truth["ra_offset_physical"].values,
                    self.cat_truth["dec_offset_physical"].values,
                    self.cat_truth["d_a"].values,
                )
            )
        )

        return KDTree(truth_coord_arr)

    def get_query_coords(self):
        """
        Get the submitted catalogue positions to query the KDTree.

        The mode attribute tells the XMatch whether to use core or centroid positions.

        Returns:
            :class:`numpy.array`: The submitted catalogue coordinate pairs, used to
            query the KDTree.
        """
        return np.array(
            list(
                zip(
                    self.cat_sub["ra_offset_physical"].values,
                    self.cat_sub["dec_offset_physical"].values,
                    self.cat_sub["d_a"].values,
                )
            )
        )

    def get_radius_arr(self):
        """
        Get the size array that sets the maximum distance a submitted source can lie
        from a truth source to be considered a candidate match.

        In the SDC2 case this is stored as the largest_size in the submitted source
        catalogue.
        """
        return self.cat_sub["largest_size"].values

    def get_all_col(self):
        """
        Get the column names in truth and submitted catalogues which should be stored in
        the output candidate match dataframe.

        In the SDC2 case, these are set as a class member.
        """
        return self.all_col

    def crossmatch_kdtree(self):
        """
        Explicit declaration of super class method.
        """
        return super().crossmatch_kdtree()
