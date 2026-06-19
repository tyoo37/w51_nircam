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
import sys
sys.path.append('/home/t.yoo/Paths')
import Paths.Paths as paths
Path = paths.filepaths()
jwst_image_filenames ={
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

jwst_image_sub_filenames = {
    "f182m-f187n": "/orange/adamginsburg/jwst/w51/filter_subtractions/f182m_minus_f187n.fits",
    "f187n-f182m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f187n_minus_f182m.fits",
    "f405n-f410m": "/orange/adamginsburg/jwst/w51/filter_subtractions/f405n_minus_f410m.fits",
    "f410m-f405n": "/orange/adamginsburg/jwst/w51/filter_subtractions/f410m_minus_f405n.fits",

}

alma_image_dir = Path.w51n_b3_tt0


def project_jwst_to_alma_expanded(wcs_A, wcs_B, data_A, data_B, save_path):
    # -------------------------------
    # 1. Create output WCS
    # -------------------------------

    wcs_out = WCS(naxis=2)

    # Use ALMA projection
    wcs_out.wcs.ctype = wcs_B.wcs.ctype

    # Use ALMA sky orientation
    wcs_out.wcs.pc = wcs_B.wcs.pc.copy()

    # Use JWST pixel size
    wcs_out.wcs.cdelt = wcs_A.wcs.cdelt.copy()

    # Use JWST reference sky position
    wcs_out.wcs.crval = wcs_A.wcs.crval.copy()

    wcs_out.wcs.cunit = list(wcs_A.wcs.cunit)

    # -------------------------------
    # 2. Find JWST footprint
    #    in the new coordinate system
    # -------------------------------

    ny, nx = data_A.shape

    corners_x = np.array([0, nx-1, nx-1, 0])
    corners_y = np.array([0, 0, ny-1, ny-1])


    # JWST pixels -> sky
    ra, dec = wcs_A.pixel_to_world_values(
        corners_x,
        corners_y
    )


    # sky -> new pixels
    x_new, y_new = wcs_out.world_to_pixel_values(
        ra,
        dec
    )


    # -------------------------------
    # 3. Determine output size
    # -------------------------------

    xmin = np.floor(x_new.min())
    xmax = np.ceil(x_new.max())

    ymin = np.floor(y_new.min())
    ymax = np.ceil(y_new.max())


    nx_out = int(xmax - xmin + 1)
    ny_out = int(ymax - ymin + 1)


    print("Output size:", nx_out, ny_out)


    # Shift reference pixel
    wcs_out.wcs.crpix = [
        -xmin,
        -ymin
    ]


    # -------------------------------
    # 4. Reproject JWST
    # -------------------------------

    jwst_reproj, footprint = reproject_interp(
        (data_A, wcs_A),
        wcs_out,
        shape_out=(ny_out, ny_out)
    )


    # save
    fits.PrimaryHDU(
        jwst_reproj,
        header=wcs_out.to_header()
    ).writeto(
        save_path,
        overwrite=True
    )
"""
for filter_name, jwst_image_filename in jwst_image_filenames.items():
    print(f"Processing {filter_name}...")
   
    jwst_data = fits.getdata(jwst_image_filenames[filter_name], ext=('SCI', 1))
    if jwst_data is None:
        print(jwst_image_filenames[filter_name])
        
        raise ValueError(f"Data is None for {filter_name}")
    jwst_hdr = fits.getheader(jwst_image_filenames[filter_name], ext=('SCI', 1))
    jwst_wcs = WCS(jwst_hdr, naxis=2)

    alma_image_filename = f"{alma_image_dir}"
    alma_hdu = fits.open(alma_image_filename)[0]
    alma_data = alma_hdu.data
    alma_wcs = WCS(alma_hdu.header, naxis=2)

    save_path = f"/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/reproject_to_alma_extended/{filter_name}_reprojected_to_alma.fits"

    project_jwst_to_alma_expanded(
        wcs_A=jwst_wcs,
        wcs_B=alma_wcs,
        data_A=jwst_data,
        data_B=alma_data,
        save_path=save_path
    )
"""
vla_image_filenames = {
 "vla_22GHz": "/orange/adamginsburg/w51/TaehwaYoo/vla/2016paper/W51-K-B.S1-ICLN.DAVID-MEH.fits",
    "vla_14GHz": "/orange/adamginsburg/w51/TaehwaYoo/vla/2016paper/W51Ku_C_Aarray_continuum_2048_high_uniform.clean.image.fits",
    "vla_8GHz": "/orange/adamginsburg/w51/TaehwaYoo/vla/2016paper/W51-X-ABCD-S1.VTESS.VTC.DAVID-MEH.fits",
    "vla_5GHz": "/orange/adamginsburg/w51/TaehwaYoo/vla/2016paper/W51-CBAND-feathered.fits"
}
for filter_name, vla_image_filename in vla_image_filenames.items():
    print(f"Processing {filter_name}...")
   
    vla_data = fits.getdata(vla_image_filenames[filter_name])
    if vla_data is None:
        print(vla_image_filenames[filter_name])
        
        raise ValueError(f"Data is None for {filter_name}")
    vla_hdr = fits.getheader(vla_image_filenames[filter_name])
    vla_wcs = WCS(vla_hdr, naxis=2)

    alma_image_filename = f"{alma_image_dir}"
    alma_hdu = fits.open(alma_image_filename)[0]
    alma_data = alma_hdu.data
    alma_wcs = WCS(alma_hdu.header, naxis=2)

    save_path = f"/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/reproject_to_alma_extended/{filter_name}_reprojected_to_alma.fits"

    project_jwst_to_alma_expanded(
        wcs_A=vla_wcs,
        wcs_B=alma_wcs,
        data_A=vla_data[0][0],
        data_B=alma_data,
        save_path=save_path
    )
