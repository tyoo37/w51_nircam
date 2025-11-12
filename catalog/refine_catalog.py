import glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import numpy as np
from regions import Regions
from astropy.nddata import Cutout2D
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm
from astropy.table import Table


def refine_catalog(filter_name, catalog, output_filename):
    #if nircam:

    roundness1 = catalog['roundness1']
    roundness2 = catalog['roundness2']
    sharpness = catalog['sharpness']
    flux = catalog['flux_fit']
    fluxerr = catalog['flux_err']
    snr = flux/fluxerr
    qfit = catalog['qfit']
    cfit = catalog['cfit']
    skycoord = catalog['skycoord']
    n_match_good = catalog['nmatch_good']

    if filter_name in ['F140M', 'F150W', 'F162M', 'F182M', 'F187N', 'F210M', 'F335M', 'F360M', 'F405N', 'F410M', 'F480M']:
        good_sources = ((roundness1 < 0.8) & (roundness1 > -0.9) & (roundness2 < 0.6) & (roundness2 > -0.6) & (sharpness < 1.2) & (sharpness>0.25) 
    & (snr > 3) & (qfit < 0.33) & (cfit < 0.2) & (cfit > -0.2) & ~((snr < 20) & (flux > 50)))

    #elif miri:
    else:
        good_sources = ((roundness1 < 0.7) & (roundness1 > -0.7) & (roundness2 < 0.5) & (roundness2 > -0.5) & (sharpness < 1.1) & (sharpness > 0.4)
                         & (snr > 3) & (qfit < 0.6) & (cfit < 0.06) & (cfit > -0.06) & (np.sqrt(std_ra**2 + std_dec**2) < 3.5e-6))

    refined_catalog = catalog[good_sources]
    refined_catalog.write(output_filename, format='fits', overwrite=True)
    return refined_catalog
filters = ['F140M', 'F162M', 'F182M', 'F187N', 'F210M', 'F335M', 'F360M', 'F405N', 'F410M', 'F480M', 'F560W', 'F770W', 'F1000W', 'F1280W', 'F2100W']
catdir = '/orange/adamginsburg/jwst/w51/catalogs/'
for filter in filters:
    catalog = 
    refine_catalog(filter, refined_catalog, f"{catdir}/refined_catalog_{filter}.fits")