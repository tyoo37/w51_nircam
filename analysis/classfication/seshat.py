import pandas as pd
from seshat_classifier import seshat
import os
os.environ["CRDS_PATH"] = os.path.expanduser("~/crds_cache")
os.makedirs(os.environ["CRDS_PATH"], exist_ok=True)

os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
os.environ["CRDS_CONTEXT"] = "jwst_1460.pmap"

from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astroquery.svo_fps import SvoFps
from astropy.wcs import WCS
from astropy.io import fits
from dust_extinction.averages import RL85_MWGC, CT06_MWLoc
from dust_extinction.parameter_averages import CCM89

def get_mag(catalog, ww, filtername='f140m' ):
    print(ww.proj_plane_pixel_area())
    
    flux= (catalog['flux_fit_' + filtername] * u.MJy/u.sr * ww.proj_plane_pixel_area()).to(u.Jy)
    eflux_jy = (catalog['flux_err_' + filtername] * u.MJy/u.sr *  ww.proj_plane_pixel_area()).to(u.Jy)

    jfilts = SvoFps.get_filter_list('JWST')
    jfilts.add_index('filterID')
    wav = int(filtername[1:-1])

    zeropoint_ab = 3631 * u.Jy  # Default to AB magnitude zero point
 
    if wav < 500:

        zeropoint_vega = u.Quantity(jfilts.loc[f'JWST/NIRCam.{filtername.upper()}']['ZeroPoint'], u.Jy)
    else:
        zeropoint_vega = u.Quantity(jfilts.loc[f'JWST/MIRI.{filtername.upper()}']['ZeroPoint'], u.Jy)
   
    abmag = -2.5 * np.log10(flux / zeropoint_ab) * u.mag
    abmag_err = 2.5 / np.log(10) * np.abs(eflux_jy / flux) * u.mag

    vegamag = -2.5 * np.log10(flux / zeropoint_vega) * u.mag
    vegamag_err = 2.5 / np.log(10) * np.abs(eflux_jy / flux) * u.mag

    return  vegamag, vegamag_err, abmag, abmag_err


image_filenames ={
    "f140m": "/orange/adamginsburg/jwst/w51/F140M/pipeline/jw06151-o001_t001_nircam_clear-f140m-merged_i2d.fits",
    "f150w": "/orange/adamginsburg/jwst/w51/F150W/pipeline/jw06151-o001_t001_nircam_clear-f150w-merged_i2d.fits",
    "f162m": "/orange/adamginsburg/jwst/w51/F162M/pipeline/jw06151-o001_t001_nircam_clear-f162m-merged_i2d.fits",
    "f182m": "/orange/adamginsburg/jwst/w51/F182M/pipeline/jw06151-o001_t001_nircam_clear-f182m-merged_i2d.fits",
    "f187n": "/orange/adamginsburg/jwst/w51/F187N/pipeline/jw06151-o001_t001_nircam_clear-f187n-merged_i2d.fits",
    "f210m": "/orange/adamginsburg/jwst/w51/F210M/pipeline/jw06151-o001_t001_nircam_clear-f210m-merged_i2d.fits",
    "f335m": "/orange/adamginsburg/jwst/w51/F335M/pipeline/jw06151-o001_t001_nircam_clear-f335m-merged_i2d.fits",
    "f360m": "/orange/adamginsburg/jwst/w51/F360M/pipeline/jw06151-o001_t001_nircam_clear-f360m-merged_i2d.fits",
    "f405n": "/orange/adamginsburg/jwst/w51/F405N/pipeline/jw06151-o001_t001_nircam_clear-f405n-merged_i2d.fits",
    "f410m": "/orange/adamginsburg/jwst/w51/F410M/pipeline/jw06151-o001_t001_nircam_clear-f410m-merged_i2d.fits", # weird, the filename is different from what is downloaded with the STScI pipeline...
    "f480m": "/orange/adamginsburg/jwst/w51/F480M/pipeline/jw06151-o001_t001_nircam_clear-f480m-merged_i2d.fits",
    "f560w": "/orange/adamginsburg/jwst/w51/F560W/pipeline/jw06151-o002_t001_miri_f560w_i2d.fits",
    "f770w": "/orange/adamginsburg/jwst/w51/F770W/pipeline/jw06151-o002_t001_miri_f770w_i2d.fits",
    "f1000w": "/orange/adamginsburg/jwst/w51/F1000W/pipeline/jw06151-o002_t001_miri_f1000w_i2d.fits",
    "f1280w": "/orange/adamginsburg/jwst/w51/F1280W/pipeline/jw06151-o002_t001_miri_f1280w_i2d.fits",
    "f2100w": "/orange/adamginsburg/jwst/w51/F2100W/pipeline/jw06151-o002_t001_miri_f2100w_i2d.fits",
    
}
catalog = Table.read('/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/final_catalog_new.fits')
nmatch = catalog['nmatch_bands']
catalog = catalog[nmatch>3]
print(catalog.colnames)


f140m_header = fits.getheader(image_filenames['f140m'], ext=('SCI', 1))
f162m_header = fits.getheader(image_filenames['f162m'], ext=('SCI', 1))
f182m_header = fits.getheader(image_filenames['f182m'], ext=('SCI', 1))
f210m_header = fits.getheader(image_filenames['f210m'], ext=('SCI', 1))
f335m_header = fits.getheader(image_filenames['f335m'], ext=('SCI', 1))
f360m_header = fits.getheader(image_filenames['f360m'], ext=('SCI', 1))
f405n_header = fits.getheader(image_filenames['f405n'], ext=('SCI', 1))
f410m_header = fits.getheader(image_filenames['f410m'], ext=('SCI', 1))
f480m_header = fits.getheader(image_filenames['f480m'], ext=('SCI', 1))
f560w_header = fits.getheader(image_filenames['f560w'], ext=('SCI', 1))
f770w_header = fits.getheader(image_filenames['f770w'], ext=('SCI', 1))
f1000w_header = fits.getheader(image_filenames['f1000w'], ext=('SCI', 1))
f1280w_header = fits.getheader(image_filenames['f1280w'], ext=('SCI', 1))
f2100w_header = fits.getheader(image_filenames['f2100w'], ext=('SCI', 1))

print(f140m_header)

f140m_mag = get_mag(catalog, WCS(f140m_header), filtername='f140m')
f162m_mag = get_mag(catalog, WCS(f162m_header), filtername='f162m')
f182m_mag = get_mag(catalog, WCS(f182m_header), filtername='f182m')
f210m_mag = get_mag(catalog, WCS(f210m_header), filtername='f210m')
f335m_mag = get_mag(catalog, WCS(f335m_header), filtername='f335m')
f360m_mag = get_mag(catalog, WCS(f360m_header), filtername='f360m')
f405n_mag = get_mag(catalog, WCS(f405n_header), filtername='f405n')
f410m_mag = get_mag(catalog, WCS(f410m_header), filtername='f410m')
f480m_mag = get_mag(catalog, WCS(f480m_header), filtername='f480m')
f560w_mag = get_mag(catalog, WCS(f560w_header), filtername='f560w')
f770w_mag = get_mag(catalog, WCS(f770w_header), filtername='f770w')
f1000w_mag = get_mag(catalog, WCS(f1000w_header), filtername='f1000w')
f1280w_mag = get_mag(catalog, WCS(f1280w_header), filtername='f1280w')
f2100w_mag = get_mag(catalog, WCS(f2100w_header), filtername='f2100w')
refined_catalog = pd.DataFrame({
    'f140m': f140m_mag[0],
    'f162m': f162m_mag[0],
    'f182m': f182m_mag[0],
    'f210m': f210m_mag[0],
    'f335m': f335m_mag[0],
    'f360m': f360m_mag[0],
    'f410m': f410m_mag[0],
    'f480m': f480m_mag[0],
    'f560w': f560w_mag[0],
    'f770w': f770w_mag[0],
    'f1000w': f1000w_mag[0],
    'f1280w': f1280w_mag[0],
    'f2100w': f2100w_mag[0],
    'e_f140m': f140m_mag[1],
    'e_f162m': f162m_mag[1],
    'e_f182m': f182m_mag[1],
    'e_f210m': f210m_mag[1],
    'e_f335m': f335m_mag[1],
    'e_f360m': f360m_mag[1],
    'e_f410m': f410m_mag[1],
    'e_f480m': f480m_mag[1],
    'e_f560w': f560w_mag[1],
    'e_f770w': f770w_mag[1],
    'e_f1000w': f1000w_mag[1],
    'e_f1280w': f1280w_mag[1],
    'e_f2100w': f2100w_mag[1],
})


classes = ['YSO', 'FS', 'Gal']
filters = [
    'f140m', 'f162m', 'f182m', 'f210m', 'f335m', 'f360m',
    'f410m', 'f480m', 'f560w', 'f770w', 'f1000w', 'f1280w', 'f2100w'
]

# refine refined_catalog to only include sources with finite f162m, f210m, f360m, f480m magnitudes
# split dataset into 5 chunks and run seshat on each chunk to avoid memory issues
catalog_subset1 = refined_catalog.iloc[:len(refined_catalog)//3].copy()
catalog_subset2 = refined_catalog.iloc[len(refined_catalog)//3:2*len(refined_catalog)//3].copy()
catalog_subset3 = refined_catalog.iloc[2*len(refined_catalog)//3:].copy()

# check that at least one element of the subcatalog has finite values for all filters
# if one filter has no finite values, we should remove it from the columns used for classification. Seshat requires at least one row to be complete in all used filters.
for filt in filters:
    if not (catalog_subset1[filt].notna().any()):
        catalog_subset1.drop(columns=[filt, f"e_{filt}"], inplace=True)
    if not (catalog_subset2[filt].notna().any()):
        catalog_subset2.drop(columns=[filt, f"e_{filt}"], inplace=True)
    if not (catalog_subset3[filt].notna().any()):
        catalog_subset3.drop(columns=[filt, f"e_{filt}"], inplace=True)

    
"""
# Keep only filters that actually exist and have some finite data
usable_filters = []
for f in filters:
    mag_ok = f in catalog_subset5.columns and catalog_subset5[f].notna().any()
    err_ok = f"e_{f}" in catalog_subset5.columns and catalog_subset5[f"e_{f}"].notna().any()
    if mag_ok and err_ok:
        usable_filters.append(f)

# Seshat needs at least one row complete in all used filters
catalog_subset5_clean = catalog_subset5.dropna(subset=usable_filters + [f"e_{f}" for f in usable_filters]).copy()

if len(catalog_subset5_clean) == 0:
    raise ValueError("No rows in catalog_subset5 are complete in the filters required by Seshat.")
"""


my_catalog_classified1 = seshat.classify(real = catalog_subset1, classes = classes, cosmological = False, return_test=False, threads = 1)
#add the classified catalogs together
catalog_subset1['class_seshat'] = my_catalog_classified1['Predicted_Class']
catalog_subset1['prob_YSO'] = my_catalog_classified1['Prob YSO']
catalog_subset1.to_pickle('/orange/adamginsburg/jwst/w51/catalogs/seshat_classified_catalog_subset1.pkl')

my_catalog_classified2 = seshat.classify(real = catalog_subset2, classes = classes, cosmological = False, return_test=False, threads = 1)
catalog_subset2['class_seshat'] = my_catalog_classified2['Predicted_Class']
catalog_subset2['prob_YSO'] = my_catalog_classified2['Prob YSO']
catalog_subset2.to_pickle('/orange/adamginsburg/jwst/w51/catalogs/seshat_classified_catalog_subset2.pkl')

my_catalog_classified3 = seshat.classify(real = catalog_subset3, classes = classes, cosmological = False, return_test=False, threads = 1)
catalog_subset3['class_seshat'] = my_catalog_classified3['Predicted_Class']
catalog_subset3['prob_YSO'] = my_catalog_classified3['Prob YSO']
catalog_subset3.to_pickle('/orange/adamginsburg/jwst/w51/catalogs/seshat_classified_catalog_subset3.pkl')
"""
my_catalog_classified4 = seshat.classify(real = catalog_subset4, classes = classes, cosmological = False, return_test=False, threads = 1)
catalog_subset4['class_seshat'] = my_catalog_classified4['Predicted_Class']
catalog_subset4['prob_YSO'] = my_catalog_classified4['Prob YSO']
catalog_subset4.to_pickle('/orange/adamginsburg/jwst/w51/catalogs/seshat_classified_catalog_subset4.pkl')


catalog_subset5['class_seshat'] = my_catalog_classified5['Predicted_Class']
catalog_subset5['prob_YSO'] = my_catalog_classified5['Prob YSO']
catalog_subset5.to_pickle('/orange/adamginsburg/jwst/w51/catalogs/seshat_classified_catalog_subset5.pkl')
"""