from astropy.io import fits
import numpy as np
from astropy.visualization import simple_norm
import pylab as plt
from astropy import wcs
import os
from reproject import reproject_interp
import PIL
#import pyavm
import shutil

from astropy.wcs import WCS
import reproject



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

image_sub_filenames = {
    "f182m-f187n": "/orange/adamginsburg/jwst/w51/filter_subtractions/f182m_minus_f187n.fits",
    "f187n-f182m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f187n_minus_f182m.fits",
    "f405n-f410m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f405n_minus_f410m.fits",
    "f410m-f405n": "/orange/adamginsburg/jwst/w51/filter_subtractions/f410m_minus_f405n.fits",

}





new_basepath = '/orange/adamginsburg/jwst/w51/data_reprojected/'
repr140_image_filenames = {x: y.replace("i2d", "i2d_reprj_f140") for x,y in image_filenames.items()}
repr140_image_filenames = {x: (new_basepath+os.path.basename(y)) for x,y in repr140_image_filenames.items()}
repr140_image_sub_filenames = {x: y.replace("i2d", "i2d_reprj_f140") for x,y in image_sub_filenames.items()}
repr140_image_sub_filenames = {x: (new_basepath+os.path.basename(y)) for x,y in repr140_image_sub_filenames.items()}

tgt_header = fits.getheader(image_filenames['f140m'], ext=('SCI', 1))
"""
for filtername in ['f162m']:
    if not os.path.exists(repr140_image_filenames[filtername]):
        print(f"Reprojecting {filtername} {image_filenames[filtername]} to {repr140_image_filenames[filtername]}")
        result,_ = reproject_interp(image_filenames[filtername], tgt_header, hdu_in='SCI')
        hdu = fits.PrimaryHDU(data=result, header=tgt_header)
        hdu.writeto(repr140_image_filenames[filtername], overwrite=True)
"""
for filtername in image_filenames:
    print(f"Reprojecting {filtername} {image_filenames[filtername]} to {repr140_image_filenames[filtername]}")
    result,_ = reproject_interp(image_filenames[filtername], tgt_header, hdu_in='SCI')
    hdu = fits.PrimaryHDU(data=result, header=tgt_header)
    hdu.writeto(repr140_image_filenames[filtername], overwrite=True)
"""
for filtername in image_sub_filenames:
    if not os.path.exists(repr140_image_sub_filenames[filtername]):
        print(f"Reprojecting {filtername} {image_sub_filenames[filtername]} to {repr140_image_sub_filenames[filtername]}")
        result,_ = reproject_interp(image_sub_filenames[filtername], tgt_header, hdu_in='SCI')
        hdu = fits.PrimaryHDU(data=result, header=tgt_header)
        hdu.writeto(repr140_image_sub_filenames[filtername], overwrite=True)
"""

import reproject
from astropy.coordinates import FK5
def expand_wcs_to_cover_all(reference_wcs, input_hdus):

# Expand a WCS (keeping its orientation & pixel scale) to cover
# the footprints of all given HDUs.

    # Get pixel scale (deg/pix)
    cdelt = np.abs(reference_wcs.wcs.cdelt)

    # Convert all images' corners to world coords, then to pixel coords in ref WCS
    all_pixels = []
    for hdu in input_hdus:
        wcs_in = WCS(hdu.header)
        ny, nx = hdu.data.shape
        corners_pix = np.array([[0,0],[nx,0],[0,ny],[nx,ny]])
        corners_world = wcs_in.all_pix2world(corners_pix, 0)
        corners_in_refpix = reference_wcs.all_world2pix(corners_world, 0)
        all_pixels.append(corners_in_refpix)

    all_pixels = np.vstack(all_pixels)

    # Find bounding box in reference pixel coordinates
    xmin, ymin = np.floor(all_pixels.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(all_pixels.max(axis=0)).astype(int)

    # Compute new shape
    new_nx = xmax - xmin
    new_ny = ymax - ymin

    # Shift CRPIX so that old reference origin moves accordingly
    new_wcs = reference_wcs.deepcopy()
    new_wcs.wcs.crpix -= [xmin, ymin]

    return new_wcs, (new_ny, new_nx)

# --- Usage ---

new_basepath = '/orange/adamginsburg/jwst/w51/data_reprojected_140_expanded/'
repr140_image_filenames = {x: y.replace("i2d", "i2d_reprj_f140") for x,y in image_filenames.items()}
repr140_image_filenames = {x: (new_basepath+os.path.basename(y)) for x,y in repr140_image_filenames.items()}
if not os.path.exists(new_basepath):
    os.makedirs(new_basepath)



filters = [filt for filt in image_filenames.keys()]
files = [image_filenames[filt] for filt in filters]
hdus = [fits.open(f)['SCI'] for f in files]

expanded_wcs, shape_out = expand_wcs_to_cover_all(WCS(hdus[0].header), hdus)
for i, (filt, filename) in enumerate(image_filenames.items()):
    reproj1, _ = reproject_interp((hdus[i].data, WCS(hdus[i].header)), expanded_wcs, shape_out=shape_out)
    hdu1_new = fits.PrimaryHDU(data=reproj1, header=expanded_wcs.to_header())
    hdu1_new.writeto(repr140_image_filenames[filt], overwrite=True)

h2k_gtc_image = fits.getdata('/orange/adamginsburg/w51/TaehwaYoo/gtc/adendawson/real_reduction/reduced_images/H2_minus_K.fits')
h2k_gtc_wcs = WCS(fits.getheader('/orange/adamginsburg/w51/TaehwaYoo/gtc/adendawson/real_reduction/reduced_images/H2_minus_K.fits'))
h2k_gtc_reproj_image, _ = reproject_interp((h2k_gtc_image, h2k_gtc_wcs), expanded_wcs, shape_out=shape_out)
h2k_gtc_hdu = fits.PrimaryHDU(data=h2k_gtc_reproj_image, header=expanded_wcs.to_header())
h2k_gtc_hdu.writeto(new_basepath + 'H2_minus_K_reprj_f140.fits', overwrite=True)
"""