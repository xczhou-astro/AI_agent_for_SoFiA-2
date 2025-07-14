## Original edited by Yingfeng Liu

# ____________________________________________________________________ #
#                                                                      #
# Usage: physical_parameter_conversion.py <VOTable> [<rel_thresh>      #
#        [<snr_thresh> [<n_pix_thresh>]]]                              #
#                                                                      #
# <VOTable> is the SoFiA 2 catalogue in VOTable (XML) format. The      #
#   catalogue must have been obtained with both physical parameter     #
#   conversion and WCS conversion enabled (SoFiA 2 settings 'parameter #
#   .physical = true' and 'parameter.wcs = true').                     #
#                                                                      #
# <rel_thresh> is an optional reliability threshold. If set, then      #
#   only detections with a reliability >= <rel_thresh> will be added   #
#   to the output catalogue.                                           #
#                                                                      #
# <snr_thresh> is an optional threshold in integrated signal-to-noise  #
#   (SNR) ratio, where SNR is calculated as f_sum / err_f_sum. Only    #
#   detections with SNR >= <snr_thresh> will be added to the output    #
#   catalogue.                                                         #
#                                                                      #
# <n_pix_thresh> is an optional threshold in the total number of       #
#   pixels contained within the source mask (SoFiA parameter n_pix).   #
#   Only detections with n_pix >= <n_pix_thresh> will be added to the  #
#   output catalogue.                                                  #
#                                                                      #
# The purpose of this script is to convert the raw source parameters   #
# from the SoFiA 2 catalogue into the required physical parameters to  #
# be submitted to the SKA Science Data Challenge 2. The catalogue from #
# SoFiA 2 must be in XML format (VOTable), and the output will be      #
# written to standard output in the correct format required by the     #
# SDC2 scoring service.                                                #
# A typical example call might look like this:                         #
#                                                                      #
#   ./physical_parameter_conversion.py sofia_cat.xml > sdc2_cat.dat    #
#                                                                      #
# which would read the SoFiA 2 source catalogue named "sofia_cat.xml"  #
# and direct the output to a file named "sdc2_cat.dat" which can then  #
# be submitted directly to the SDC2 scoring service.                   #
#                                                                      #
# Note that several settings and assumptions, e.g. on the cosmology    #
# used, are hard-coded at the beginning of the main routine below.     #
# Users are advised to review and revise those as needed.              #
# ____________________________________________________________________ #
#                                                                      #

import sys
import math
import astropy.units as au
from astropy.cosmology import FlatLambdaCDM
from astropy.io.votable import parse_single_table
import numpy as np


# ----------------------
# Gaussian deconvolution
# ----------------------
#
# This function deconvolves the major axis, minor axis and position angle of
# one Gaussian with those of another Gaussian. The deconvolved parameters are
# returned. The algorithm is the same as the one used by the ATNF Miriad data
# reduction package.
#
# Arguments:
#
#  a1, b1    Major and minor axis of the Gaussian to be deconvolved. These
#            must be standard deviations (sigma) in radians (rad).
#  pa1       Position angle in radians of the Gaussian to be deconvolved.
#  a2, b2    Same as a1, b1, but for the Gaussian to deconvolve with
#            (e.g. beam size).
#  pa2       Same as pa1. but for the Gaussian to deconvolve with
#            (e.g. beam PA).
#
# Return value:
#
#  (a, b, pa)   Tuple containing the major axis, minor axis and position angle
#               of the deconvolved Gaussian. a and b are standard deviations
#               in radians, while pa is the position angle in radians.


class AstroPyCosmo(FlatLambdaCDM):
    """Cosmology calculator based on `astro.cosmology.FlatLambdaCDM` with default parameters:
        - Hubble constant: H0 = 67.3
        - matter density: Om0 = 0.315
        - baryon density: Ob0 = 0.048825
        - sigma8 = 0.826

    Notes:
        Distances in Mpc.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('H0', 67.3)
        kwargs.setdefault('Om0', 0.315)
        kwargs.setdefault('Ob0', 0.048825)
        kwargs.setdefault('Tcmb0', 2.7255)
        kwargs.setdefault('Neff', 3.05)
        kwargs.setdefault('m_nu', [0.0, 0.0, 0.06] * au.eV)
        super(AstroPyCosmo, self).__init__(*args, **kwargs)


def deconvolve(a1, b1, pa1, a2, b2, pa2):
    alpha = (
        (a1 * math.cos(pa1)) ** 2.0
        + (b1 * math.sin(pa1)) ** 2.0
        - (a2 * math.cos(pa2)) ** 2.0
        - (b2 * math.sin(pa2)) ** 2.0
    )
    beta = (
        (a1 * math.sin(pa1)) ** 2.0
        + (b1 * math.cos(pa1)) ** 2.0
        - (a2 * math.sin(pa2)) ** 2.0
        - (b2 * math.cos(pa2)) ** 2.0
    )
    gamma = 2.0 * (b1 * b1 - a1 * a1) * math.sin(pa1) * math.cos(pa1) - 2.0 * (
        b2 * b2 - a2 * a2
    ) * math.sin(pa2) * math.cos(pa2)

    if alpha < 0 or beta < 0:
        return (0.0, 0.0, 0.0)

    s = alpha + beta
    t = math.sqrt((alpha - beta) * (alpha - beta) + gamma * gamma)

    if s < t:
        return (0.0, 0.0, 0.0)

    a = math.sqrt(0.5 * (s + t))
    b = math.sqrt(0.5 * (s - t))
    if abs(alpha - beta) + abs(gamma) > 0.0:
        pa = 0.5 * math.atan2(-gamma, alpha - beta)
    else:
        pa = 0.0

    return (a, b, pa)


# --------------------
# Conversion functions
# --------------------
#
# Several convenience functions for converting between frequently required
# properties. These should all be self-explanatory.


def deg_to_rad(x):
    return x * math.pi / 180.0


def rad_to_deg(x):
    return x * 180.0 / math.pi


def sigma_to_fwhm(x):
    return x * 2.0 * math.sqrt(2.0 * math.log(2.0))


def fwhm_to_sigma(x):
    return x / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def deg_to_arcsec(x):
    return x * 3600.0


def arcsec_to_deg(x):
    return x / 3600.0


# --------------------------
# Parameter bias corrections
# --------------------------
#
# Noise bias correction for certain parameters as a function of flux.
# The flux is expected to be supplied in units of Jy Hz.
# The corrections were derived by comparing the parameters measured by
# SoFiA with those from the truth catalogue during a test run on the
# 40 GB development data cube.


def corr_flux(flux):
    x = math.log10(flux)
    return (
        -1.95944 * math.pow(x, 6)
        + 25.0718 * math.pow(x, 5)
        - 130.981 * math.pow(x, 4)
        + 357.67 * math.pow(x, 2)
        - 538.804 * math.pow(x, 2)
        + 424.973 * x
        - 136.178
    )


def corr_w20(flux):
    x = math.log10(flux)
    return 0.109584 * math.pow(x, 3) - 1.092518 * math.pow(x, 2) + 3.013832 * x - 1.309686


def corr_size(flux):
    x = math.log10(flux)
    return 0.384714 * math.pow(x, 3) - 2.448307 * math.pow(x, 2) + 4.967467 * x - 2.287058


# ------------
# Main routine
# ------------


def main(
    votable, thresh_rel=0.0, thresh_snr=0.0, thresh_npix=0.0, *,
    no_corr=False, drop_neg_nan=False, skew_fill_cut=False,
    SDC2_cosmo=False,
):
    # Mathematical and physical constants
    const_c = 299792.458       # km/s
    const_f0 = 1.420405752e9  # Hz
    micro = 1e-6
    milli = 1e-3
    kilo = 1e3
    mega = 1e6

    # Global settings
    beam_size = 7.00            # beam FWHM in arcsec
    px_size = 2.80              # pixel size in arcsec
    ch_size = 3e4              # channel size in Hz
    beam_area = math.pi * (beam_size / px_size) * (beam_size / px_size) / (4.0 * math.log(2.0))

    if SDC2_cosmo:
        H0 = 67.00                  # Hubble constant in km/s/Mpc
        Om0 = 0.32                  # cosmic matter density
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)  # set up cosmology
    else:
        cosmo = AstroPyCosmo()  # set up cosmology

    # Default values for unresolved sources
    def_size = 8.5      # mean HI size assumed for point sources (arcsec)
    def_incl = 57.3     # mean disk inclination assumed for point sources (deg)

    # Read SoFiA catalogue (VOTable only)
    table = parse_single_table(votable).to_table()

    if drop_neg_nan:
        table = table[(table['kin_pa'] != -1) & ~table['kin_pa'].mask]
    else:
        table['kin_pa'][(table['kin_pa'] == -1) | table['kin_pa'].mask] = 0.0

    # Extract relevant columns
    freq = table['freq'].data                 # Hz
    flux = table['f_sum'].data                # Jy Hz
    flux_err = table['err_f_sum'].data        # Jy Hz
    ell_maj = table['ell_maj'].data           # px (2 sigma)
    ell_min = table['ell_min'].data           # px (2 sigma)
    ell_pa = table['ell_pa'].data             # deg
    w20 = table['w20'].data                   # Hz
    kin_pa = table['kin_pa'].data             # deg
    ra = table['ra'].data                     # deg
    dec = table['dec'].data                   # deg
    rel = table['rel'].data
    n_pix = table['n_pix'].data
    rms = table['rms'].data
    snr = flux / flux_err

    skew = None
    try:
        skew = table['skew'].data
    except Exception:
        pass
    else:
        fill = table['fill'].data
        std = table['std'].data

    kin_pa += 180.0
    kin_pa[kin_pa>=360.0] -= 360.0

    # Print header row
    sys.stdout.write('id ra dec hi_size line_flux_integral central_freq pa i w20\n')
    source_id_counter = 0

    # Loop over all detections
    for i in range(len(freq)):
        # Check user-defined thresholds
        if snr[i] < thresh_snr or n_pix[i] < thresh_npix or rel[i] < thresh_rel:  # noqa
            continue

        # Hard-coded cuts in skew-n_pix space and fill-snr space
        # These are the primary cuts used to discard false positives
        # due to noise
        if skew is not None and skew_fill_cut:
            if skew[i] < -0.00135 * (n_pix[i] - 942) or fill[i] > 0.18 * snr[i] + 0.17:  # noqa
                continue

        # --------------------------------------------
        # (1) Convert ellipse size to physical HI size
        #     and calculate disc inclination angle
        # --------------------------------------------
        hi_size = def_size
        incl = def_incl

        # Obtain redshift and distances
        z = const_f0 / freq[i] - 1
        dl = cosmo.luminosity_distance(z).value                         # Mpc
        da = cosmo.angular_diameter_distance(z).value                   # Mpc

        # Deconvolve major and minor axis of source
        a1 = deg_to_rad(0.5 * ell_maj[i] * arcsec_to_deg(px_size))      # rad (sigma)
        b1 = deg_to_rad(0.5 * ell_min[i] * arcsec_to_deg(px_size))      # rad (sigma)
        pa1 = deg_to_rad(ell_pa[i])                                     # rad
        a2 = fwhm_to_sigma(deg_to_rad(arcsec_to_deg(beam_size)))        # rad (sigma)
        (a, b, pa) = deconvolve(a1, b1, pa1, a2, a2, 0.0)               # note: beam is symmetric

        if a != 0.0 and b != 0.0:
            # Convert deconvolved major axis to physical size
            sigma_phys = a * da * mega                                  # pc (sigma)

            # Derive HI mass and central mass surface density
            M_HI = 49.7 * dl * dl * flux[i]                             # M_sun
            S0 = M_HI / (2.0 * math.pi * sigma_phys * sigma_phys)       # M_sun / pc^2

            if S0 > 1.0:
                # Source diameter at 1 M_sun/pc^2
                hi_size = math.sqrt(-2.0 * math.log(1.0 / S0)) * sigma_phys      # pc
                hi_size = 2.0 * deg_to_arcsec(rad_to_deg(hi_size * micro / da))  # arcsec
                if not no_corr:
                    hi_size /= corr_size(flux[i])

                # Disc inclination
                incl = rad_to_deg(math.acos(b / a))

        # --------------------------------------------
        # (2) Convert line width to km/s in source RF
        # --------------------------------------------
        w20[i] *= const_c * (1.0 + z) / const_f0  # km/s
        if not no_corr:
            w20[i] /= corr_w20(flux[i])

        # --------------------------------------------
        # (3) Print output row increment counter
        # --------------------------------------------
        write_to_cat = w20[i] > 0.0 and hi_size > 0.0
        if not no_corr:
            write_to_cat = write_to_cat and corr_flux(flux[i]) > 0.0

        if write_to_cat:
            sys.stdout.write(
                '{:d} {:.14f} {:.14f} {:.14f} {:.14f} {:.1f} {:.14f} {:.14f} {:.14f}\n'.format(
                    source_id_counter,
                    ra[i], dec[i], hi_size,
                    flux[i] if no_corr else flux[i] / corr_flux(flux[i]),
                    freq[i], kin_pa[i], incl, w20[i],
                )  # noqa
            )
            source_id_counter += 1


# ------------
# Run main
# ------------

if __name__ == '__main__':
    import argh
    argh.dispatch_command(main, old_name_mapping_policy=False)
