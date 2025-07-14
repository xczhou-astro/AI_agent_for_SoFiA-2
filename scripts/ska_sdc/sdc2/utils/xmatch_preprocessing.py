import inspect
import logging
import time
from importlib import import_module


class XMatchPreprocessing:
    """
    Prepare catalogues for crossmatching.
    """

    def __init__(self, step_names=[]):
        """
        Args:
            step_names (:obj:`list`): Name of the steps to be imported from
                :class:`ska_sdc.sdc2.utils.xmatch_preprocessing_steps`
        """
        self.steps = []
        for step_name in step_names:
            self.steps.append(
                getattr(
                    import_module("ska_sdc.sdc2.utils.xmatch_preprocessing_steps"),
                    step_name,
                )
            )

    def preprocess(self, *args, **kwargs):
        """
        A wrapper function used to sequentially call all other prerequisite
        crossmatching preprocessing functions.

        Returns:
            :class:`pandas.DataFrame`: Preprocessed catalogue.
        """
        start = time.time()

        cat_rtn = kwargs["cat"]
        for step in self.steps:
            cat_rtn = step(*args, **kwargs).execute()

            # Overwrite with catalogue output from preceding step.
            #
            kwargs["cat"] = cat_rtn

        logging.info(
            "[{}.{}] complete in {:.2f}s".format(
                self.__class__.__name__, inspect.stack()[0][3], time.time() - start
            )
        )
        return cat_rtn
