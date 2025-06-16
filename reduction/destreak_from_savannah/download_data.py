import os
import astroquery
from astroquery.mast import Observations

mast_dir = 'mast:jwst/product' # Download from MAST
data_dir = 'data'  # save downloaded data
os.makedirs(data_dir, exist_ok=True)

# JWST images to be analyzed

image_files = [
    "jw01345001001_10201_00001_nrca3_rate.fits",  # sparse
    "jw01345001001_10201_00001_nrca3_cal.fits",
    "jw01074001001_02101_00001_nrca1_rate.fits",  # crowded
    "jw01074001001_02101_00001_nrca1_cal.fits",
    "jw01074001001_02101_00001_nrca1_i2d.fits",
    "jw02107025001_02101_00001_nrcb2_rate.fits",  # extended
    "jw02107025001_02101_00001_nrcb2_cal.fits",
    "jw01484177001_04201_00001_nrs1_rate.fits", ## problem?
    "jw01484177001_04201_00001_nrs2_rate.fits",## problem?
    "jw01335004001_03101_00002_nrs1_rate.fits",
    "jw01335004001_03101_00002_nrs2_rate.fits",
    "jw01345061001_07101_00003_nrs1_rate.fits",
    "jw01345061001_07101_00003_nrs2_rate.fits",
    "jw01501001001_26101_00001_nis_rate.fits"
]

for image_file in image_files:
    # Download file (if not already downloaded)
    
    #for product in =['_rate.fits','_uncal.fits']:
    for product in ['_rate.fits']:
        get_file = image_file.replace('_rate.fits',product)
    
    mast_path  = os.path.join(mast_dir, get_file)
    local_path = os.path.join(data_dir, get_file)
    if os.path.exists(image_file) == False:
        Observations.download_file(mast_path, local_path=local_path)