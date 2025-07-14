import configparser
import logging
import time

from ska_sdc.common.models.exceptions import NoScoreException
from ska_sdc.common.utils.cat_io import load_dataframe, setup_logging
from ska_sdc.data.data_resources import SDC2_CONFIG_PATH
from ska_sdc.sdc2.utils.create_score import create_sdc_score
from ska_sdc.sdc2.utils.sdc2_xmatch import Sdc2XMatch
from ska_sdc.sdc2.utils.xmatch_postprocessing import XMatchPostprocessing
from ska_sdc.sdc2.utils.xmatch_preprocessing import XMatchPreprocessing
from ska_sdc.sdc2.validate import Validation


class Sdc2Scorer:
    """
    The SDC2 scorer class.

    Args:
        cat_sub (:obj:`pandas.DataFrame`): The submission catalogue.
        cat_truth (:obj:`pandas.DataFrame`): The truth catalogue.
    """

    def __init__(self, cat_sub, cat_truth):
        self.cat_sub = cat_sub
        self.cat_truth = cat_truth
        self.config = configparser.ConfigParser()
        self.config.read(SDC2_CONFIG_PATH)

        self._score = None
        self._scoring_complete = False

        # Run validation.
        #
        self._validate()

    @classmethod
    def from_txt(
        cls,
        sub_path,
        truth_path,
        sub_skiprows=0,
        truth_skiprows=0,
    ):
        """
        Create an SDC2 scorer class from two source catalogues in text format.

        The catalogues must have a header row of column names that matches the expected
        column names in the config file.

        Args:
            sub_path (:obj:`str`): Path to the submission catalogue.
            truth_path (:obj:`str`): Path to the truth catalogue.
            sub_skiprows (:obj:`int`, optional): Number of rows to skip in
                submission catalogue. Defaults to 0.
            truth_skiprows (:obj:`int`, optional): Number of rows to skip in
                truth catalogue. Defaults to 0.
        """
        # Column names are inferred from header and can be validated against the config
        cat_sub = load_dataframe(sub_path, columns=None, skip_n=sub_skiprows)
        cat_truth = load_dataframe(truth_path, columns=None, skip_n=truth_skiprows)

        return cls(cat_sub, cat_truth)

    def _create_score(self, train, detail):
        """
        Execute the scoring pipeline.
        """
        setup_logging()
        pipeline_start = time.time()
        logging.info("Scoring pipeline started")

        # Preprocess input submission and truth catalogues.
        #
        n_det = len(self.cat_sub)

        cat_sub_prep = XMatchPreprocessing(
            step_names=["ScaleAndCalculateLargestSize"]
        ).preprocess(cat=self.cat_sub, config=self.config)
        cat_truth_prep = XMatchPreprocessing(
            step_names=["ScaleAndCalculateLargestSize"]
        ).preprocess(cat=self.cat_truth, config=self.config)

        # Perform crossmatch to generate dataframe of candidate matches.
        #
        all_col = self.get_sub_cat_columns() + ["conv_size", "spectral_size"]
        cand_cat_sub = Sdc2XMatch(
            cat_sub=cat_sub_prep, cat_truth=cat_truth_prep, all_col=all_col
        ).execute(func_name="crossmatch_kdtree")

        # Postprocess crossmatched catalogue.
        #
        cand_cat_sub_postp = XMatchPostprocessing(
            step_names=["CalculateMultidErr", "Sieve"]
        ).postprocess(cat=cand_cat_sub, config=self.config)

        # Construct sdc_score object.
        #

        sdc_score = create_sdc_score(
            self.config,
            cand_cat_sub_postp,
            n_det,
            train,
            detail,
        )
        logging.info(
            "Scoring pipeline complete. Elapsed time: {:.2f}s".format(
                time.time() - pipeline_start
            )
        )
        logging.info("Final score: {:.2f}".format(sdc_score.value))

        return sdc_score, cand_cat_sub_postp

    def get_sub_cat_columns(self):
        return self.config["general"]["sub_cat_column_names"].split(",")

    def _validate(self):
        """
        Validate DataFrames and config.
        """
        Validation.is_valid_config(self.config)

        sub_cat_column_names = self.get_sub_cat_columns()
        Validation.is_valid_df(self.cat_sub, sub_cat_column_names)
        Validation.is_valid_df(self.cat_truth, sub_cat_column_names)

    def is_scoring_complete(self):
        return self._scoring_complete

    def run(self, train=False, detail=False):
        """
        Run the scoring pipeline.

        Returns:
            :class:`ska_sdc.sdc2.models.sdc2_score.Sdc2Score`: The calculated
            SDC2 score object
        """

        self._score, self._cand_cat_sub_postp = self._create_score(train, detail)
        self._scoring_complete = True

        return self.score, self._cand_cat_sub_postp

    @property
    def score(self):
        """
        Get the resulting Sdc2Score object.

        Returns:
            :class:`ska_sdc.sdc2.models.sdc2_score.Sdc2Score`: The calculated SDC2 score
            object
        """
        if self._score is None:
            err_msg = "No score calculated. Use the run method to calculate a score."
            logging.error(err_msg)
            raise NoScoreException(err_msg)
        return self._score
    
    
    def candidates(self):

        if self._cand_cat_sub_postp is None:
            err_msg = "No score calculated. Use the run method to calculate a score."
            logging.error(err_msg)
            raise NoScoreException(err_msg)
        
        
        return self._cand_cat_sub_postp