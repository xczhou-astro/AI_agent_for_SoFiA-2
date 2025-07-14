# Valid catalogue frequencies in SDC1
FREQS = [560, 1400, 9200]

# The submitted/truth catalogue column names
CAT_COLUMNS = [
    "id",
    "ra_core",
    "dec_core",
    "ra_cent",
    "dec_cent",
    "flux",
    "core_frac",
    "b_maj",
    "b_min",
    "pa",
    "size",
    "class",
]

# Image field centre
RA_CENTRE = 0.0
DEC_CENTRE = -30.0

# Training area limits for each frequency value
TRAIN_LIM = {
    9200: {"ra_min": -0.04092, "ra_max": 0.0, "dec_min": -29.9400, "dec_max": -29.9074},
    1400: {"ra_min": -0.2688, "ra_max": 0.0, "dec_min": -29.9400, "dec_max": -29.7265},
    560: {"ra_min": -0.6723, "ra_max": 0.0, "dec_min": -29.9400, "dec_max": -29.4061},
}

# Enum for positional cross-match modes
MODE_CORE, MODE_CENTR, MODE_FULL = range(3)

# Threshold values
multid_thr = 5.0  # number of sigmas to accept a match as a true positive
position_thr = 0.3  # accuracy of position recovery
flux_thr = 0.1  # accuracy of flux recovery
size_thr = 0.3  # accuracy of size recovery
pa_thr = 10.0  # accuracy of position angle recovery
core_thr = 0.05  # accuracy of core fraction recovery
