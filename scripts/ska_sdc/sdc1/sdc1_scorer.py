import logging
import time

from ska_sdc.common.models.exceptions import NoScoreException
from ska_sdc.common.utils.cat_io import load_dataframe, setup_logging
from ska_sdc.sdc1.dc_defns import CAT_COLUMNS, MODE_CORE
from ska_sdc.sdc1.utils.create_score import create_sdc_score
from ska_sdc.sdc1.utils.prep import prepare_data
from ska_sdc.sdc1.utils.sdc1_xmatch import Sdc1XMatch
from ska_sdc.sdc1.utils.sieve import process_kdtree_cand_df
from ska_sdc.sdc1.validate import validate_df, validate_freq


class Sdc1Scorer:
    """
    The SDC1 scorer class.

    Args:
        sub_df (:obj:`pandas.DataFrame`): The submission catalogue
            DataFrame of detected sources and properties
        truth_path (:obj:`pandas.DataFrame`): The truth catalogue
            DataFrame
        freq (:obj:`int`): Image frequency band (560, 1400 or 9200 MHz)
    """

    def __init__(self, sub_df, truth_df, freq):
        self.sub_df = sub_df
        self.truth_df = truth_df
        self.freq = freq

        self._score = None
        self._scoring_complete = False

        self._validate()

    @property
    def score(self):
        """
        Get the resulting Sdc1Score object.

        Returns:
            :class:`ska_sdc.sdc1.models.sdc1_score.Sdc1Score`: The calculated
            SDC1 score object
        """
        if self._score is None:
            err_msg = "No score calculated. Use the run method to calculate a score."
            logging.error(err_msg)
            raise NoScoreException(err_msg)
        return self._score

    @classmethod
    def from_txt(
        cls,
        sub_path,
        truth_path,
        freq,
        sub_skiprows=1,
        truth_skiprows=0,
    ):
        """
        Create an SDC1 scorer class from two source catalogues in text format.

        Args:
            sub_path (:obj:`str`): The path of the submission catalogue of
                detected sources and properties
            truth_path (:obj:`str`): The path of the truth catalogue
            freq (:obj:`int`): Image frequency band (560, 1400 or 9200 MHz)
            sub_skiprows (:obj:`int`, optional): Number of rows to skip in
                submission catalogue. Defaults to 1.
            truth_skiprows (:obj:`int`, optional): Number of rows to skip in
                truth catalogue. Defaults to 0.
        """
        truth_df = load_dataframe(
            truth_path, columns=CAT_COLUMNS, skip_n=truth_skiprows
        )
        sub_df = load_dataframe(sub_path, columns=CAT_COLUMNS, skip_n=sub_skiprows)
        return cls(sub_df, truth_df, freq)

    def _create_score(self, mode, train, detail):
        """
        Execute the scoring pipeline, according to the following steps:

        #. prepare_data: Pre-process truth/submitted catalogues to unify
        #. crossmatch_kdtree: Crossmatch sources between submission and truth
        #  catalogues
        #. process_kdtree_cand_df: Sieve and standardise crossmatch output
        #. create_sdc_score: Generate the sdc_score object
        """
        setup_logging()
        pipeline_start = time.time()
        logging.info("Scoring pipeline started")

        # Prepare data catalogues
        sub_df_prep = prepare_data(self.sub_df, self.freq, train)
        truth_df_prep = prepare_data(self.truth_df, self.freq, train)
        logging.info(
            "Catalogue preparation complete. Elapsed time: {:.2f}s".format(
                time.time() - pipeline_start
            )
        )

        # Perform crossmatch to generate dataframe of candidate matches
        cand_sub_df = Sdc1XMatch(
            cat_sub=sub_df_prep, cat_truth=truth_df_prep, mode=mode
        ).execute(func_name="crossmatch_kdtree")
        logging.info(
            "Crossmatch runs complete. Elapsed time: {:.2f}s".format(
                time.time() - pipeline_start
            )
        )

        # Sieve results and calculate score:
        sieved_sub_df = process_kdtree_cand_df(cand_sub_df, mode)
        logging.info(
            "Sieving complete. Elapsed time: {:.2f}s".format(
                time.time() - pipeline_start
            )
        )

        # Construct sdc_score object:
        n_det = len(sub_df_prep.index)
        sdc_score = create_sdc_score(
            sieved_sub_df, self.freq, n_det, mode, train, detail
        )
        logging.info(
            "Scoring pipeline complete. Elapsed time: {:.2f}s".format(
                time.time() - pipeline_start
            )
        )
        logging.info("Final score: {:.2f}".format(sdc_score.value))

        return sdc_score

    def _validate(self):
        """
        Validate user input.
        """
        validate_df(self.sub_df)
        validate_df(self.truth_df)
        validate_freq(self.freq)

    def run(self, mode=MODE_CORE, train=False, detail=False):
        """
        Run the scoring pipeline.

        Args:
            mode (:obj:`int`, optional): 0 or 1 to use core or centroid
                positions for scoring
            train (:obj:`bool`, optional): If True, will only evaluate
                score based on training area, else will exclude training
                area
            detail (:obj:`bool`, optional): If True, will return the
                catalogue of matches and per source scores.

        Returns:
            :class:`ska_sdc.sdc1.models.sdc1_score.Sdc1Score`: The calculated
                SDC1 score object
        """
        self._scoring_complete = False
        self._score = self._create_score(mode, train, detail)
        self._scoring_complete = True

        return self._score

    def is_scoring_complete(self):
        return self._scoring_complete
