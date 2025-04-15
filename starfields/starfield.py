# starfield/starfield.py
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astroquery.gaia import Gaia


class Starfield:
    def __init__(
        self,
        width,
        height,
        pixscale_arcsec,
        zeropoint=20.0,
        saturation_val=65535,
    ):
        """Starfield Class

        Parameters
        ----------
        width : int
            width of image in pix
        height : int
            height of image in pix
        pixscale_arcsec : float
            arcseconds per pixel
        zeropoint : float, optional
            mag-to-flux zeropoint, by default 20.0
        saturation_val : int, optional
            maximum allowed DN/ADU value, by default 65535
        """
        self.width = width
        self.height = height
        self.pixscale = pixscale_arcsec
        self.zeropoint = zeropoint
        self.clip = saturation_val

    def get_stars(
        self, ra=None, dec=None, theta_deg=0.0, fov_radius=1.5, gmag_max=13.0
    ):
        """Grab a catalog from Gaia

        Parameters
        ----------
        ra: float, optional
            ra center of field; if not provided, a random one will be chosen, by default None
        dec: float, optional
            dec center of field; if not provided, a random one will be chosen. Note that valid dec is -30 to 80, by default None
        theta_deg: float, optional
            field rotation in degrees, by default 0.0
        fov_radius: float, optional
            cone (in deg) with which to retrieve stars (ideally a bit bigger than your image), by default 1.5
        gmag_max: float, optional
            faintest gaia g-band magnitude to include in query. This tool is designed for brighter stars, by default 13.0

        Returns
        -------
        catalog: pandas.DataFrame
            dataframe with columns XIMAGE and YIMAGE in addition to ra, dec, and gmag. Sets class attribute as well.
        """
        if ra is None:
            ra = np.random.uniform(0, 360)
        if dec is None:
            dec = np.random.uniform(-30, 80)

        query = f"""
        SELECT ra, dec, phot_g_mean_mag
        FROM gaiadr3.gaia_source
        WHERE phot_g_mean_mag < {gmag_max}
        AND 1 = CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra}, {dec}, {fov_radius})
        )
        """
        job = Gaia.launch_job_async(query)
        df = job.get_results().to_pandas()
        df.rename(columns={"phot_g_mean_mag": "gmag"}, inplace=True)

        self.wcs = self.make_simple_wcs(ra, dec, theta_deg)
        sky_coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg)
        x, y = self.wcs.world_to_pixel(sky_coords)
        df["XIMAGE"] = x
        df["YIMAGE"] = y

        self.catalog = df[
            (x >= 0) & (x < self.width) & (y >= 0) & (y < self.height)
        ].copy()
        return self.catalog

    def make_simple_wcs(
        self, ra_center: float, dec_center: float, theta_deg: float = 0.0
    ):
        """Helper function to generate a WCS

        Parameters
        ----------
        ra_center : float
            field center RA
        dec_center : float
            Field center dec
        theta_deg : float, optional
            field rotation, by default 0.0

        Returns
        -------
        w: astropy.wcs.WCS
            WCS object describing the position on sky.
        """
        w = WCS(naxis=2)
        w.wcs.crval = [ra_center, dec_center]
        w.wcs.crpix = [self.width / 2, self.height / 2]
        scale_deg = self.pixscale / 3600.0
        angle_rad = np.deg2rad(theta_deg)
        w.wcs.cd = np.array(
            [
                [-scale_deg * np.cos(angle_rad), scale_deg * np.sin(angle_rad)],
                [scale_deg * np.sin(angle_rad), scale_deg * np.cos(angle_rad)],
            ]
        )
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        return w

    def vignette_map(self, strength=0.4, power=2.5):
        """Multiplicative flatfield simulator

        Parameters
        ----------
        strength : float, optional
            max difference between edge and center, by default 0.4
        power : float, optional
            rolloff term; bigger means image more flat for more of the image, by default 2.5

        Returns
        -------
        vignette: np.ndarray
            image of width/height set by class with scaling values.
        """
        y, x = np.indices((self.height, self.width))
        cx, cy = self.width / 2, self.height / 2
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        r_norm = r / np.max(r)
        vignette = 1.0 - strength * (r_norm**power)
        return np.clip(vignette, 1.0 - strength, 1.0)

    def simulate_image(
        self,
        fwhm_x: float,
        fwhm_y: float,
        jitter: float = 0.0,
        shift_x: float = 0.0,
        shift_y: float = 0.0,
        offset: float = 200.0,
        cloud_transparency: float = 1.0,
        exptime: float = 1.0,
        gain: float = 1.0,
        read_noise: float = 1.0,
        background_noise: float = 100,
        hot_pixel_frac: float = 1e-5,
        seed: int = None,
        vignette_strength: float = 0.4,
        vignette_power: float = 2.5,
    ):
        """Primary Image Simulation Tool

        Parameters
        ----------
        fwhm_x : float
            pixel fwhm of stars in x
        fwhm_y : _type_
            pixel fwhm of stars in y
        jitter : float, optional
            scintillation term; if > 0, stars will be individually nudged a random direction with sigma = jitter, by default 0.0
        shift_x : float, optional
            global pixel offset for all stars in x, by default 0.0
        shift_y : float, optional
            global pixel offset for all stars in y, by default 0.0
        offset : float, optional
            constant offset/bias level, by default 200.0
        cloud_transparency : float, optional
            simple global factor by which to attenuate star fluxes (but not bias), by default 1.0
        exptime : float, optional
            used with gain to convert model image to e, by default 1.0
        gain : float, optional
            used with exptime to convert model image to e, by default 1.0
        read_noise : float, optional
            base detector noise, by default 1.0
        background_noise : float, optional
            arbitrary noise term, by default 100.0
        hot_pixel_frac : float, optional
            fraction of random pixels to be hot (with values up to saturation), by default 1e-5
        seed : int, optional
            seed for reproducibility, by default None
        vignette_strength : float, optional
            edge-to-center ratio in flatfield, by default 0.4
        vignette_power : float, optional
            rolloff strength of flatfield, by default 2.5

        Returns
        -------
        img: np.ndarray
            simulated image with starfield and noise/detector terms added (or multiplied).
        """
        image = np.zeros((self.height, self.width), dtype=np.float64)
        stamp_size = int(6 * max(fwhm_x, fwhm_y))

        if seed is not None:
            np.random.seed(seed)

        for _, row in self.catalog.iterrows():
            x, y, mag = row["XIMAGE"], row["YIMAGE"], row["gmag"]
            flux = 10 ** (-0.4 * (mag - self.zeropoint))
            x_shifted = x + shift_x
            y_shifted = y + shift_y
            x0, y0 = int(np.floor(x_shifted)), int(np.floor(y_shifted))
            dx = x_shifted - x0 + np.random.normal(0, jitter)
            dy = y_shifted - y0 + np.random.normal(0, jitter)
            stamp = gaussian_stamp_subpixel(
                flux, fwhm_x, fwhm_y, dx, dy, oversample=16, stamp_size=stamp_size
            )
            x1, y1 = x0 - stamp_size, y0 - stamp_size
            x2, y2 = x0 + stamp_size + 1, y0 + stamp_size + 1
            xs1, ys1 = max(0, x1), max(0, y1)
            xs2, ys2 = min(self.width, x2), min(self.height, y2)
            sx1, sy1 = xs1 - x1, ys1 - y1
            sw, sh = xs2 - xs1, ys2 - ys1
            image[ys1 : ys1 + sh, xs1 : xs1 + sw] += stamp[
                sy1 : sy1 + sh, sx1 : sx1 + sw
            ]

        image *= cloud_transparency
        image += offset
        image *= self.vignette_map(vignette_strength, vignette_power)
        image = np.clip(image, 0, self.clip)

        e_image = image * exptime * gain
        noisy_e = np.random.poisson(e_image).astype(np.float64)
        noisy_e += np.random.normal(0, read_noise, size=image.shape)
        noisy_e = np.clip(noisy_e, 0, None)
        noisy_dn = noisy_e / gain
        noisy_dn += np.random.normal(0, background_noise, size=image.shape)
        noisy_dn += np.random.normal(0, 0.2, size=(1, self.width))

        n_hot = int(hot_pixel_frac * image.size)
        ys = np.random.randint(0, self.height, size=n_hot)
        xs = np.random.randint(0, self.width, size=n_hot)
        noisy_dn[ys, xs] += np.random.uniform(3000, self.clip, size=n_hot)

        return np.clip(noisy_dn, 0, self.clip)


def gaussian_stamp_subpixel(
    flux: float,
    fwhm_x: float,
    fwhm_y: float,
    dx: float,
    dy: float,
    oversample: int = 8,
    stamp_size: int = 15,
):
    """Helper function to create cutout stamps of gaussian stars

    Parameters
    ----------
    flux : float
        flux of star
    fwhm_x : float
        fwhm in the x direction (pixel)
    fwhm_y : float
        fhwm in the y direction (pixel)
    dx : float
        offset from center of patch in x
    dy : float
        offset from center of patch in y
    oversample : int, optional
        over sample grid to compute gaussian on before binning to image pixels, by default 8
    stamp_size : int, optional
        size of stamp grid, should be several times the chosen fwhm, by default 15

    Returns
    -------
    psf: np.ndarray
        stamp to be summed into the larger image
    """
    sigma_x = fwhm_x / 2.355
    sigma_y = fwhm_y / 2.355
    grid = np.arange(-stamp_size, stamp_size + 1, 1 / oversample, dtype=np.float64)
    xx, yy = np.meshgrid(grid - dx, grid - dy)
    psf_oversampled = np.exp(-0.5 * ((xx / sigma_x) ** 2 + (yy / sigma_y) ** 2))
    psf_oversampled *= flux / psf_oversampled.sum()
    size = 2 * stamp_size + 1
    shape = (size, oversample, size, oversample)
    return psf_oversampled.reshape(shape).sum(axis=(1, 3))
