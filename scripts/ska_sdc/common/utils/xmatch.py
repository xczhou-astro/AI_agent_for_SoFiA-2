import abc
import inspect
import logging
import time

import pandas as pd


class XMatch:
    """
    Crossmatch sources.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            cat_sub:class (:`pandas.DataFrame`): Submission catalogue.
            cat_truth (:class:`pandas.DataFrame`): Truth catalogue.
        """
        self.__dict__.update(kwargs)

    def _stub(self):
        """
        A stub crossmatching function.

        Returns:
            :class:`pandas.DataFrame`: Crossmatched catalogue.
        """
        return pd.DataFrame()

    def execute(self, func_name="_stub", **kwargs):
        start = time.time()

        exec_func = getattr(self, func_name)
        cat_rtn = exec_func(**kwargs)

        logging.info(
            "[{}.{}] complete in {:.2f}s".format(
                self.__class__.__name__, inspect.stack()[0][3], time.time() - start
            )
        )
        return cat_rtn


class KDTreeXMatch(XMatch):
    """
    Abstract base class for KDTree-based XMatch processes.

    These will query for candidate matches within a defined radius of a set of points
    in an N-D space.
    """

    @abc.abstractmethod
    def get_all_col(self):
        """
        Must be overridden by subclasses
        """
        raise Exception("Not Implemented")

    @abc.abstractmethod
    def get_kdtree(self):
        """
        Must be overridden by subclasses
        """
        raise Exception("Not Implemented")

    @abc.abstractmethod
    def get_radius_arr(self):
        """
        Must be overridden by subclasses
        """
        raise Exception("Not Implemented")

    @abc.abstractmethod
    def get_query_coords(self):
        """
        Must be overridden by subclasses
        """
        raise Exception("Not Implemented")

    def refine_match_df(self, cand_match_df):
        """
        Can be overridden by subclasses to perform any minor post-xmatch refinement of
        the candidate match DataFrame.

        This may include renaming columns, changing the index, etc. - more extensive
        operations should be separated.
        """
        return cand_match_df

    def get_truth_suffix(self):
        """
        Get the suffix label to be added to columns corresponding to truth properties.

        Can be overridden by subclasses.
        """
        return "_t"

    def crossmatch_kdtree(self):
        """
        Query for all submitted sources within a defined radius of the truth catalogue.

        Uses subclass-overridden methods to construct the KDTree, radius array and query
        arrays. Subclass must also define the columns to be written to the output
        catalogue.

        Returns:
            :class:`pandas.DataFrame`: The candidate match catalogue.
        """
        all_col = self.get_all_col()

        self.cat_truth = self.cat_truth.reset_index(drop=True)
        self.cat_sub = self.cat_sub.reset_index(drop=True)

        truth_val_map = {}
        sub_val_map = {}

        match_truth_map = {}
        match_sub_map = {}

        # For performance, unpack dataframes into arrays of values
        for col in all_col:
            truth_val_map[col] = self.cat_truth[col].values
            sub_val_map[col] = self.cat_sub[col].values

        for col in all_col:
            match_truth_map[col] = []
            match_sub_map[col] = []

        sub_coord_arr = self.get_query_coords()
        size_arr = self.get_radius_arr()
        point_kdtree = self.get_kdtree()

        if len(sub_coord_arr) > 0:
            for sub_index, (_center, group) in enumerate(
                zip(sub_coord_arr, point_kdtree.query_radius(sub_coord_arr, r=size_arr))
            ):
                for match_index in group:
                    for col in all_col:
                        match_sub_map[col].append(sub_val_map[col][sub_index])
                        match_truth_map[col].append(truth_val_map[col][match_index])

        # Construct final candidate match DF:
        truth_suffix = self.get_truth_suffix()
        match_df_data = {}
        for col in all_col:
            match_df_data[col] = match_sub_map[col]
        for col in all_col:
            match_df_data[col + truth_suffix] = match_truth_map[col]

        cand_match_df = self.refine_match_df(pd.DataFrame(match_df_data))
        return cand_match_df
