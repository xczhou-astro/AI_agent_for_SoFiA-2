from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import LambdaCDM
from scipy.constants import speed_of_light
from ska_sdc.common.utils.constants import H_0, Om_lambda, Om_m


class XMatchPreprocessingStep:
    """
    Base class for preprocessing steps.
    """

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)


class XMatchPreprocessingStepStub(XMatchPreprocessingStep):
    """
    Stub class for a preprocessing step.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute(self):
        """
        Execute the step.

        Returns:
            :class:`pandas.DataFrame`: Processed catalogue.
        """
        # Logic placeholder
        return self.cat


class ScaleAndCalculateLargestSize(XMatchPreprocessingStep):
    """
    Calculate the additional properties that will be used in crossmatching.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def execute(self):
        """
        Calculate convolved size; this is necessary to control for the potentially
        small Gaussian source sizes, which could yield an unrepresentative
        positional accuracy.

        Thus we calculate the apparent size by convolving with the beam size.

        Returns:
            :class:`pandas.DataFrame`: Processed catalogue.
        """
        # Retrieve values from config:
        rest_freq = self.config.getfloat("cube", "rest_freq")  # Hz

        # beam_size is needed to get convolved sizes
        beam_size = self.config.getfloat("cube", "beam_size")

        field_centre_ra = self.config.getfloat("cube", "field_centre_ra")
        field_centre_dec = self.config.getfloat("cube", "field_centre_dec")

        # TODO PH: Could replace with astropy.cosmology.WMAP9 for simplicity;
        # would slightly change results tho
        cosmo = LambdaCDM(H0=H_0, Om0=Om_m, Ode0=Om_lambda)

        # remove any rows where frequency is outside bandwidth range
        self.cat = self.cat[self.cat["central_freq"] > 0.95e9]
        self.cat = self.cat[self.cat["central_freq"] < 1.15e9]

        # Calculate physical locations (Mpc) for matching
        z = rest_freq / self.cat["central_freq"] - 1

        self.cat["d_a"] = cosmo.angular_diameter_distance(z).value

        sky_coord = SkyCoord(
            ra=self.cat["ra"].values * u.deg,
            dec=self.cat["dec"].values * u.deg,
            frame="fk5",
        )
        sky_coord_centre = SkyCoord(
            ra=field_centre_ra * u.deg,
            dec=field_centre_dec * u.deg,
            frame="fk5",
        )
        ra_offset_deg, dec_offset_deg = sky_coord_centre.spherical_offsets_to(sky_coord)

        self.cat["ra_offset_physical"] = (
            self.cat["d_a"] * ra_offset_deg.to(u.radian).value
        )
        self.cat["dec_offset_physical"] = (
            self.cat["d_a"] * dec_offset_deg.to(u.radian).value
        )

        # We will use a cuboid positional cross-match, so use the greater of the
        # source dimensions

        # Approximate convolved size by adding the beam and source sizes in quadrature
        self.cat["conv_size"] = ((self.cat["hi_size"] ** 2) + (beam_size ** 2)) ** 0.5
        self.cat["physical_conv_size"] = (self.cat["conv_size"] / 206265) * self.cat[
            "d_a"
        ]

        # Calculate line width in frequency (units of Hz)
        spectral_size_velocity_s = self.cat["w20"] * 1000  # m/s
        self.cat["spectral_size"] = spectral_size_velocity_s / (
            (speed_of_light * rest_freq) / (self.cat["central_freq"] ** 2)
        )
        freq_plus_dfreq = self.cat["central_freq"] + self.cat["spectral_size"]
        z_dfreq = rest_freq / freq_plus_dfreq - 1

        da_dfreq = cosmo.angular_diameter_distance(z_dfreq).value
        self.cat["physical_spectral_size"] = self.cat["d_a"] - da_dfreq

        # TODO PH: Clarify comment "not using this, using conv_size"
        self.cat["largest_size"] = self.cat[
            ["physical_conv_size", "physical_spectral_size"]
        ].max(axis=1)

        return self.cat
