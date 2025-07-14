from math import pi

# General constants
deg2rad = pi / 180

# Gaussian to Largest Angular Scale (LAS)
# Gaussian has FWHM = 2.355 * sigma
# LAS defined as 5 sigma maximum extent:
gauss_to_las = 5.0 / 2.355
las_to_gauss = 1.0 / gauss_to_las

# Exponential to Gaussian; factor sqrt(2)
expo_to_gauss = 2 ** 0.5
gauss_to_expo = 1 / expo_to_gauss

# LAS to exponential
las_to_expo = las_to_gauss * gauss_to_expo
expo_to_las = 1 / las_to_expo

# Cosmology constants
H_0 = 67.0
Om_m = 0.32
Om_lambda = 0.68
