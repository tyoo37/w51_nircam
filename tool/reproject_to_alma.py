

from spectral_cube import SpectralCube
from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt

from spectral_cube import SpectralCube
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
import os
from reproject import reproject_interp
import matplotlib.pyplot as plt
from regions import Regions
from astropy.nddata import Cutout2D
from astropy.wcs import WCS


reprojected_image, footprint = reproject_interp(
    (HCN_moment0_image, HCN_moment0_wcs),
    alma_b3_wcs,
    shape_out=alma_b3_image.shape
)