from ska_sdc.common.models.sdc_score import SdcScore


class Sdc2Score(SdcScore):
    """
    Simple data container class for collating data relating to an SDC2 score.

    This is created by the SDC2 Scorer's run method.
    """

    def __init__(self, train=False, detail=False):
        SdcScore.__init__(self)

        self.train = train
        self.detail = detail

        self._n_det = 0
        self._n_bad = 0
        self._n_match = 0
        self._n_false = 0
        self._score_det = 0.0
        self._acc_pc = 0.0
        self._scores_df = None
        self._match_df = None

    @property
    def train(self):
        """
        If True, has evaluated score based on training area, else excludes
        training area.

        Returns:
            :obj:`bool`
        """
        return self._train

    @train.setter
    def train(self, train):
        self._train = train

    @property
    def detail(self):
        """
        If True, has returned the catalogue of matches and per source scores.

        Returns:
            :obj:`bool`
        """
        return self._detail

    @detail.setter
    def detail(self, detail):
        self._detail = detail

    @property
    def n_det(self):
        """
        The total number of detected sources in the submission.

        Returns:
            :obj:`int`
        """
        return self._n_det

    @n_det.setter
    def n_det(self, n_det):
        self._n_det = n_det

    @property
    def n_bad(self):
        """
        Number of candidate matches rejected during data cleansing.

        Returns:
            :obj:`int`
        """
        return self._n_bad

    @n_bad.setter
    def n_bad(self, n_bad):
        self._n_bad = n_bad

    @property
    def n_match(self):
        """
        Number of candidate matches below threshold.

        Returns:
            :obj:`int`
        """
        return self._n_match

    @n_match.setter
    def n_match(self, n_match):
        self._n_match = n_match

    @property
    def n_false(self):
        """
        Number of false detections.

        Returns:
            :obj:`int`
        """
        return self._n_false

    @n_false.setter
    def n_false(self, n_false):
        self._n_false = n_false

    @property
    def score_det(self):
        """
        The sum of the scores.

        Returns:
            :obj:`float64`
        """
        return self._score_det

    @score_det.setter
    def score_det(self, score_det):
        self._score_det = score_det

    @property
    def acc_pc(self):
        """
        The average score per match (%).

        Returns:
            :obj:`float64`
        """
        return self._acc_pc

    @acc_pc.setter
    def acc_pc(self, acc_pc):
        self._acc_pc = acc_pc

    @property
    def scores_df(self):
        """
        Dataframe containing the scores.

        Returns:
            :obj:`pandas.DataFrame`
        """
        return self._scores_df

    @scores_df.setter
    def scores_df(self, scores_df):
        if self.detail:
            self._scores_df = scores_df

    @property
    def match_df(self):
        """
        Dataframe of matched sources.

        Returns:
            :obj:`pandas.DataFrame`
        """
        return self._match_df

    @match_df.setter
    def match_df(self, match_df):
        if self.detail:
            self._match_df = match_df
