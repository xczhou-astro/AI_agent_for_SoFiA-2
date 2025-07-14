import os

import pandas as pd
import pkg_resources

# Primary beam info data
pb_info_names = ["radius", "pixels", "average", "rms", "ann_sum", "cum_sum"]
pb_info_stream = pkg_resources.resource_stream(
    __name__, os.path.join("beam_info", "PB_I_14.log")
)
pb_info_df = pd.read_csv(
    pb_info_stream,
    header=None,
    names=pb_info_names,
    delim_whitespace=True,
    comment="#",
)

# SDC2 configuration file
SDC2_CONFIG_PATH = pkg_resources.resource_filename(
    __name__, os.path.join("conf", "sdc2", "config.ini")
)
