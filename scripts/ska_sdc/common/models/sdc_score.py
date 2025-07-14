class SdcScore:
    """
    Simple data container class for collating data relating to an SDC score.

    This is created by the SDC Scorer's run method.
    """

    def __init__(self):
        self._value = 0.0

    @property
    def value(self):
        """
        The score for the last run.

        Returns:
            :obj:`float64`
        """
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
