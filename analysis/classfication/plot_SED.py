from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.table import Table
from astropy.io import fits
from astropy.visualization import simple_norm
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import re
from astropy.coordinates import SkyCoord
import sys
sys.path.append('/home/t.yoo/Paths')
import Paths.Paths as paths
import matplotlib as mpl
import matplotlib.patches as patches
Path = paths.filepaths()

import warnings
from astropy.wcs import FITSFixedWarning
from astroquery.svo_fps import SvoFps

warnings.filterwarnings("ignore", category=FITSFixedWarning)

plt.rcParams['axes.labelsize']=20
plt.rcParams['xtick.labelsize']=15
plt.rcParams['ytick.labelsize']=15
image_filenames ={
    "f140m": "/orange/adamginsburg/jwst/w51/F140M/pipeline/jw06151-o001_t001_nircam_clear-f140m-merged_i2d.fits",
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
    "f1500w": "/orange/adamginsburg/jwst/w51/F1500W/pipeline/jw06151-o002_t001_miri_f1500w_i2d.fits",
    "f2100w": "/orange/adamginsburg/jwst/w51/F2100W/pipeline/jw06151-o002_t001_miri_f2100w_i2d.fits",
    "w51e_1.3mm": Path.w51e_b6_tt0,
    "w51e_3mm": Path.w51e_b3_tt0,
    "w51n_1.3mm": Path.w51n_b6_tt0,
    "w51n_3mm": Path.w51n_b3_tt0,
    "vla_22GHz": "/orange/adamginsburg/w51/TaehwaYoo/vla/2016paper/W51-K-B.S1-ICLN.DAVID-MEH.fits",
    "vla_14GHz": "/orange/adamginsburg/w51/TaehwaYoo/vla/2016paper/W51Ku_C_Aarray_continuum_2048_high_uniform.clean.image.fits",
    "vla_8GHz": "/orange/adamginsburg/w51/TaehwaYoo/vla/2016paper/W51-X-ABCD-S1.VTESS.VTC.DAVID-MEH.fits",
    "vla_5GHz": "/orange/adamginsburg/w51/TaehwaYoo/vla/2016paper/W51-CBAND-feathered.fits"
}
catalogs_filters = {"f140m_nrca": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f140m_nrca_indivexp_merged_dao_after_merger_combined_with_satstars.fits',
                   "f162m_nrca": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f162m_nrca_indivexp_merged_dao_after_merger_combined_with_satstars.fits',
                   "f182m_nrca": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f182m_nrca_indivexp_merged_dao_after_merger_combined_with_satstars.fits',
                   "f187n_nrca": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f187n_nrca_indivexp_merged_dao_after_merger_combined_with_satstars.fits',
                   "f210m_nrca": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f210m_nrca_indivexp_merged_dao_after_merger_combined_with_satstars.fits',
                   "f335m_nrca": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f335m_nrca_indivexp_merged_dao_after_merger_combined_with_satstars.fits',
                   "f360m_nrca": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f360m_nrca_indivexp_merged_dao_after_merger_combined_with_satstars.fits',
                   "f405n_nrca": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f405n_nrca_indivexp_merged_dao_after_merger_combined_with_satstars.fits',
                   "f410m_nrca": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f410m_nrca_indivexp_merged_dao_after_merger_combined_with_satstars.fits',
                   "f480m_nrca": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f480m_nrca_indivexp_merged_dao_after_merger_combined_with_satstars.fits',
                   "f140m_nrcb": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f140m_nrcb_indivexp_merged_dao_after_merger_combined_with_satstars.fits',
                   "f162m_nrcb": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f162m_nrcb_indivexp_merged_dao_after_merger_combined_with_satstars.fits',
                  "f182m_nrcb": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f182m_nrcb_indivexp_merged_dao_after_merger_combined_with_satstars.fits',
                   "f187n_nrcb": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f187n_nrcb_indivexp_merged_dao_after_merger_combined_with_satstars.fits',
                   "f210m_nrcb": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f210m_nrcb_indivexp_merged_dao_after_merger_combined_with_satstars.fits',
                   "f335m_nrcb": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f335m_nrcb_indivexp_merged_dao_after_merger_combined_with_satstars.fits',
                   "f360m_nrcb": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f360m_nrcb_indivexp_merged_dao_after_merger_combined_with_satstars.fits',
                   "f405n_nrcb": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f405n_nrcb_indivexp_merged_dao_after_merger_combined_with_satstars.fits',
                   "f410m_nrcb": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f410m_nrcb_indivexp_merged_dao_after_merger_combined_with_satstars.fits',
                   "f480m_nrcb": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f480m_nrcb_indivexp_merged_dao_after_merger_combined_with_satstars.fits',
                   "f560w": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f560w_mirimage_indivexp_merged_dao_after_merger_combined_with_satstars_fixed.fits',
                   "f770w": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f770w_mirimage_indivexp_merged_dao_after_merger_combined_with_satstars_fixed.fits',
                   "f1000w": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f1000w_mirimage_indivexp_merged_dao_after_merger_combined_with_satstars_fixed.fits',
                   "f1280w": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f1280w_mirimage_indivexp_merged_dao_after_merger_combined_with_satstars_fixed.fits',
                   "f2100w": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f2100w_mirimage_indivexp_merged_dao_after_merger_combined_with_satstars_fixed.fits',
                   "vla": '/home/t.yoo/w51/w51_nircam/analysis/vla.fits',
                  }
                  
catalogs_filters_filtered = {"f140m_nrca": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f140m_nrca_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_a.fits',
                   "f162m_nrca": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f162m_nrca_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_a.fits',
                   "f182m_nrca": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f182m_nrca_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_a.fits',
                   "f187n_nrca": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f187n_nrca_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_a.fits',
                   "f210m_nrca": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f210m_nrca_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_a.fits',
                   "f335m_nrca": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f335m_nrca_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_a.fits',
                   "f360m_nrca": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f360m_nrca_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_a.fits',
                   "f405n_nrca": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f405n_nrca_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_a.fits',
                   "f410m_nrca": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f410m_nrca_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_a.fits',
                   "f480m_nrca": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f480m_nrca_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_a.fits',
                   "f140m_nrcb": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f140m_nrcb_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_a.fits',
                   "f162m_nrcb": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f162m_nrcb_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_a.fits',
                  "f182m_nrcb": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f182m_nrcb_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_a.fits',
                   "f187n_nrcb": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f187n_nrcb_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_a.fits',
                   "f210m_nrcb": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f210m_nrcb_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_a.fits',
                   "f335m_nrcb": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f335m_nrcb_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_a.fits',
                   "f360m_nrcb": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f360m_nrcb_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_a.fits',
                   "f405n_nrcb": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f405n_nrcb_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_a.fits',
                   "f410m_nrcb": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f410m_nrcb_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_a.fits',
                   "f480m_nrcb": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f480m_nrcb_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_a.fits',
                   "f560w": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f560w_mirimage_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_b_fixed.fits',
                   "f770w": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f770w_mirimage_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_b_fixed.fits',
                   "f1000w": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f1000w_mirimage_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_b_fixed.fits',
                   "f1280w": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f1280w_mirimage_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_b_fixed.fits',
                   "f2100w": '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/f2100w_mirimage_indivexp_merged_dao_after_merger_combined_with_satstars_nmatch_cut_grade_b_fixed.fits',
                   "vla": '/home/t.yoo/w51/w51_nircam/analysis/vla.fits',
                  }
reprojected_dir = '/orange/adamginsburg/jwst/w51/reproject_to_alma_extended/'

def get_mag(flux, ww, filtername='f140m' ):
    print(ww.proj_plane_pixel_area())
    
    #flux= (flux * u.MJy/u.sr * ww.proj_plane_pixel_area()).to(u.Jy)
   # eflux_jy = (catalog['flux_err_' + filtername] * u.MJy/u.sr *  ww.proj_plane_pixel_area()).to(u.Jy)

    jfilts = SvoFps.get_filter_list('JWST')
    jfilts.add_index('filterID')
    wav = int(filtername[1:-1])

    zeropoint_ab = 3631 * u.Jy  # Default to AB magnitude zero point
 
    if wav < 500:

        zeropoint_vega = u.Quantity(jfilts.loc[f'JWST/NIRCam.{filtername.upper()}']['ZeroPoint'], u.Jy)
    else:
        zeropoint_vega = u.Quantity(jfilts.loc[f'JWST/MIRI.{filtername.upper()}']['ZeroPoint'], u.Jy)
   
 #   abmag = -2.5 * np.log10(flux / zeropoint_ab) * u.mag
#    abmag_err = 2.5 / np.log(10) * np.abs(eflux_jy / flux) * u.mag

    vegamag = -2.5 * np.log10(flux / zeropoint_vega) 
 #   vegamag_err = 2.5 / np.log(10) * np.abs(eflux_jy / flux) * u.mag

    return  vegamag


catalog = Table.read('/orange/adamginsburg/jwst/w51/catalogs/final_nircam_miri_indivexp_merged_dao_refined_after_sat.fits')
catalog = Table.read('/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/final_catalog_new.fits')

def plot_SED(sky_coordinates, label, 
            jwst_catalog='/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/final_catalog_new.fits', 
            alma_catalog=['/blue/adamginsburg/t.yoo/from_red/w51/w51_frag_new/dendro/tables/dendro_w51e_master.fits',
                        '/blue/adamginsburg/t.yoo/from_red/w51/w51_frag_new/dendro/tables/dendro_w51n_master.fits'], 
            vla_catalog='/home/t.yoo/w51/w51_nircam/analysis/vla_updated.fits', 
            cutout_size=2*u.arcsec, alma_region='w51e', show=False, matching_rad = 0.1*u.arcsec, 
            SEDsavedir='/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/plots/seds/'):
  
    # VLA catalog refinement
    vla_tab = Table.read(vla_catalog)
    vla_ra = vla_tab['GRAdeg']
    vla_dec = vla_tab['GDEdeg']
    finite_idx = np.isfinite(vla_ra) & np.isfinite(vla_dec)
    vla_tab_updated = vla_tab[finite_idx]

    jwst_tab = Table.read(jwst_catalog)
    alma_tab = Table.read(alma_catalog[0]) if alma_region == 'w51e' else Table.read(alma_catalog[1])

    skycoord_jwst = SkyCoord(ra=jwst_tab['skycoord_ref'].ra, dec=jwst_tab['skycoord_ref'].dec)
    skycoord_alma = SkyCoord(ra=alma_tab['ra']*u.deg, dec=alma_tab['dec']*u.deg)
    skycoord_vla = SkyCoord(ra=vla_tab_updated['GRAdeg'], dec=vla_tab_updated['GDEdeg'])

    idx_jwst, d2d_jwst, _ = sky_coordinates.match_to_catalog_sky(skycoord_jwst, nthneighbor=1)
    idx_vla, d2d_vla, _ = sky_coordinates.match_to_catalog_sky(skycoord_vla, nthneighbor=1)
    idx_alma, d2d_alma, _ = sky_coordinates.match_to_catalog_sky(skycoord_alma, nthneighbor=1)

    ismatched_jwst = np.sum(d2d_jwst < matching_rad, axis=0) > 0
    if ismatched_jwst:
        row_jwst = jwst_tab[idx_jwst]
    else:
        row_jwst = None
    
    ismatched_vla = np.sum(d2d_vla < matching_rad, axis=0) > 0
    if ismatched_vla:
        row_vla = vla_tab_updated[idx_vla]
    else:
        row_vla = None

    ismatched_alma = np.sum(d2d_alma < matching_rad, axis=0) > 0
    if ismatched_alma:
        row_alma = alma_tab[idx_alma]
    else:
        row_alma = None


    fig = plt.figure(figsize=(22, 10))
    gs = GridSpec(5,11, figure=fig, wspace=0, hspace=0)
    ax_f140m = fig.add_subplot(gs[0,0])
    ax_f162m = fig.add_subplot(gs[0,1])
    ax_f182m = fig.add_subplot(gs[0,2])
    ax_f187n = fig.add_subplot(gs[0,3])
    ax_f210m = fig.add_subplot(gs[0,4])
    ax_f335m = fig.add_subplot(gs[0,5])
    ax_f360m = fig.add_subplot(gs[0,6])
    ax_f405n = fig.add_subplot(gs[0,7])
    ax_f410m = fig.add_subplot(gs[0,8])
    ax_f480m = fig.add_subplot(gs[0,9])
    ax_f560w = fig.add_subplot(gs[0,10])
    ax_f770w = fig.add_subplot(gs[1,0])
    ax_f1000w = fig.add_subplot(gs[1,1])
    ax_f1280w = fig.add_subplot(gs[1,2])
    ax_f2100w = fig.add_subplot(gs[1,3])
    ax_b6 = fig.add_subplot(gs[1,4])
    ax_b3 = fig.add_subplot(gs[1,5])
    ax_vla_22GHz = fig.add_subplot(gs[1,6])
    ax_vla_14GHz = fig.add_subplot(gs[1,7])
    ax_vla_8GHz = fig.add_subplot(gs[1,8])
    ax_vla_5GHz = fig.add_subplot(gs[1,9])
    ax_images = [ax_f140m, ax_f162m, ax_f182m, ax_f187n, ax_f210m, ax_f335m, ax_f360m, ax_f405n,
                 ax_f410m, ax_f480m, ax_f560w, ax_f770w, ax_f1000w, ax_f1280w, ax_f2100w, ax_b6, ax_b3, ax_vla_22GHz, ax_vla_14GHz, ax_vla_8GHz, ax_vla_5GHz]
    ax_main = fig.add_subplot(gs[2:5, :])
    filter_names = ["f140m", "f162m", "f182m", "f187n", "f210m", "f335m", "f360m", "f405n",
                    "f410m", "f480m", "f560w", "f770w", "f1000w", "f1280w", "f2100w", "1.3mm", "3mm",  "vla_22GHz", "vla_14GHz", "vla_8GHz", "vla_5GHz", ]
    #skycoords = SkyCoord(ra=catalog['skycoord_ref'].ra[idx], dec=catalog['skycoord_ref'].dec[idx])
 
    for i, ax in enumerate(ax_images):  
       
        img_b3 = image_filenames[f'{alma_region}_3mm']
        header_b3 = fits.open(img_b3)[0].header
        pixel_scale_b3 = WCS(header_b3, naxis=2).proj_plane_pixel_scales()[0]
        img_b6 = image_filenames[f'{alma_region}_1.3mm']
        header_b6 = fits.open(img_b6)[0].header
        pixel_scale_b6 = WCS(header_b6, naxis=2).proj_plane_pixel_scales()[0]

        if filter_names[i] in ["1.3mm", "3mm"]: 
            if filter_names[i] == "1.3mm":
                band = 'b6'
            elif filter_names[i] == "3mm":
                band = 'b3'
            img_filename = image_filenames[f"{alma_region}_{filter_names[i]}"]
            print('alma image filename:', img_filename)
            img = fits.open(img_filename)[0].data[0][0]
            header = fits.open(img_filename)[0].header
            wcs = WCS(header, naxis=2)
        elif filter_names[i] in ['f140m', 'f162m', 'f182m', 'f187n', 'f210m', 'f335m', 'f360m', 'f405n', 'f410m', 'f480m', 'f560w', 'f770w', 'f1000w', 'f1280w', 'f2100w']:
            #filt}_reprojected_to_alma_w51n_b6.fits
            img_filename = reprojected_dir + f"{filter_names[i]}_reprojected_to_alma.fits"
            print('JWST image filename:', img_filename)
            img = fits.open(img_filename)[0].data
            header = fits.open(img_filename)[0].header
            wcs = WCS(header, naxis=2)
        else:
            
            image_filename = reprojected_dir + f"{filter_names[i]}_reprojected_to_alma.fits"
            img = fits.open(image_filename)[0].data
            if not len(img.shape) == 2:
                img = img[0][0]
            header = fits.open(image_filename)[0].header
            wcs = WCS(header, naxis=2)
          
        # plot cutout images

        try:
        
            cutout = Cutout2D(img, sky_coordinates, (cutout_size, cutout_size), wcs=wcs)
            norm = simple_norm(cutout.data, 'sqrt', percent=99.5)
            ax.imshow(cutout.data, norm=norm, origin='lower', cmap='inferno')
            if filter_names[i] == "1.3mm":
                pixel_scale = pixel_scale_b6
            else:
                pixel_scale = pixel_scale_b3
            print('pixel_scale:', pixel_scale)
                
            circle = patches.Circle((cutout.data.shape[1]/2, cutout.data.shape[0]/2), radius=(0.1*u.arcsec/pixel_scale).to(u.deg/u.deg).value, edgecolor='cyan', facecolor='none', lw=2)
            ax.add_patch(circle)
            print(filter_names[i])
            ax.text(0.1, 0.9, filter_names[i].upper(), transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
            ax.axis('off')
            ax.set_xlim(0, cutout.data.shape[1])
            ax.set_ylim(0, cutout.data.shape[0])
        except Exception as e:
            print(img.shape)
            pixcoord = sky_coordinates.to_pixel(wcs)
            print('pixcoord:', pixcoord)
            print(f"Could not create cutout for filter {filter_names[i]}: {e}")
            # imshow empty 2d array
            ax.imshow(np.zeros((100, 100)), origin='lower', cmap='inferno')
            ax.text(0.1, 0.9, filter_names[i].upper(), transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
            ax.axis('off')
            ax.set_xlim(0, cutout.data.shape[1])
            ax.set_ylim(0, cutout.data.shape[0])
                 
        # plot catalog sources in the cutout
            
        if not filter_names[i] in ['1.3mm', '3mm']:
            if filter_names[i] in ['f140m', 'f162m', 'f182m', 'f187n', 'f210m', 'f335m', 'f360m', 'f405n', 'f410m', 'f480m']:
                cat_nrca = catalogs_filters_filtered[f'{filter_names[i]}_nrca']
                skycoord_nrca = Table.read(cat_nrca)['skycoord']
                print('catalog filename cat_nrca:', cat_nrca)
                pixcoord_nrca = skycoord_nrca.to_pixel(cutout.wcs)
                cat_nrcb = catalogs_filters_filtered[f'{filter_names[i]}_nrcb']
                print('catalog filename cat_nrcb:', cat_nrcb)

                skycoord_nrcb = Table.read(cat_nrcb)['skycoord']
                pixcoord_nrcb = skycoord_nrcb.to_pixel(cutout.wcs)
                ax.scatter(pixcoord_nrca[0], pixcoord_nrca[1], facecolor='none', color='blue', s=10)
                ax.scatter(pixcoord_nrcb[0], pixcoord_nrcb[1], facecolor='none', color='red', s=10)
                
            
            elif filter_names[i] in ['f560w', 'f770w', 'f1000w', 'f1280w', 'f2100w']:
                cat_miri = catalogs_filters_filtered[f'{filter_names[i]}']
                print('catalog filename cat_miri:', cat_miri)
                skycoord_miri = Table.read(cat_miri)['skycoord']
                pixcoord_miri = skycoord_miri.to_pixel(cutout.wcs)
                ax.scatter(pixcoord_miri[0], pixcoord_miri[1], facecolor='none', color='green', s=10)

            else:
                
                cat_vla = Table.read(catalogs_filters_filtered['vla'])
                print('catalog filename cat_vla:', catalogs_filters_filtered['vla'])
                ra = cat_vla['GRAdeg']
                dec = cat_vla['GDEdeg']
                freq = cat_vla['Freq']
                freq_selected = filter_names[i].split('_')[1]
                if freq_selected.endswith('GHz'):
                    freq_selected = freq_selected[:-3]  # Remove 'GHz' from the end
                freq_idx = np.where(np.isclose(freq, float(freq_selected), rtol=0.1))[0]
                if len(freq_idx) > 0:
                    ra = ra[freq_idx]
                    dec = dec[freq_idx]
                    skycoord_vla = SkyCoord(ra=ra, dec=dec)
                    pixcoord_vla = skycoord_vla.to_pixel(cutout.wcs)
                    ax.scatter(pixcoord_vla[0], pixcoord_vla[1], facecolor='none', color='magenta', s=10)

        # limit xlim ylim as same as cutout size
       
    # plot SED
    colors = mpl.cm.viridis(np.linspace(0, 1, len(filter_names)))
    fluxarr = []
    mag_f140m = None
    mag_f210m = None
    mag_f360m = None
    mag_f480m = None
    for i, filter_name in enumerate(filter_names):
        flux_jy = np.nan * u.Jy
        
        if filter_name == '1.3mm':
            wav = 1300
            if row_alma is not None:
                flux_jy = row_alma['flux_b6'] * u.Jy if not hasattr(row_alma['flux_b6'], 'unit') else row_alma['flux_b6']
        elif filter_name == '3mm':
            wav = 3000
            if row_alma is not None:
                flux_jy = row_alma['flux_b3'] * u.Jy if not hasattr(row_alma['flux_b3'], 'unit') else row_alma['flux_b3']
        elif filter_name in ["vla_5GHz", "vla_8GHz", "vla_14GHz", "vla_22GHz"]:
            if row_vla is not None:
                freq_selected = filter_name.split('_')[1]
                if freq_selected.endswith('GHz'):
                    freq_selected = freq_selected[:-3]  # Remove 'GHz' from the end        
                print('freq_selected:', freq_selected)
                freq = row_vla['Freq']
                freq_idx = np.where(np.isclose(freq, float(freq_selected), rtol=0.1))[0]
                if len(freq_idx) > 0:
                    #flux_jy = row_vla['FluxPk'][freq_idx] * u.Jy if not hasattr(row_vla['FluxPk'], 'unit') else row_vla['FluxPk'][freq_idx]
                    # get flux_jy with the most recent obs_date
                    obs_date = row_vla['Obs_date'][freq_idx]
                    if isinstance(obs_date, np.ma.MaskedArray):
                        obs_date = obs_date.data
                    if len(obs_date) > 1:
                        latest_idx = np.argmax(obs_date)
                        flux_jy = row_vla['FluxPk'][freq_idx][latest_idx] * u.Jy if not hasattr(row_vla['FluxPk'][freq_idx][latest_idx], 'unit') else row_vla['FluxPk'][freq_idx][latest_idx]
                        wav = (3e8 / (freq[freq_idx][latest_idx] * 1e9)) * 1e6 # convert frequency in GHz to wavelength in micron
                        print('wav in vla', wav)
                    else:
                        flux_jy = row_vla['FluxPk'][freq_idx] * u.Jy if not hasattr(row_vla['FluxPk'][freq_idx], 'unit') else row_vla['FluxPk'][freq_idx]
                        wav = (3e8 / (freq[freq_idx] * 1e9)) * 1e6 # convert frequency in GHz to wavelength in micron
                        print('wav in vla', wav)
                else:
                    print(f"No matching frequency found in VLA catalog for {filter_name}")
                    flux_jy = np.nan * u.Jy
            else:
                flux_jy = np.nan * u.Jy

        else:
            print(filter_name)
            wcs_filter = WCS(fits.getheader(image_filenames[filter_name], ext=('SCI', 1)), naxis=2)
            wav = int(filter_name[1:-1])/100
            flux_jy = (row_jwst['flux_fit_' + filter_name] * u.MJy / u.sr * wcs_filter.proj_plane_pixel_area()).to(u.Jy) # convert surface brightness to flux density in Jy
        

        print('flux_jy:', flux_jy)

        if np.isnan(flux_jy) or flux_jy <= 0 or np.ma.is_masked(flux_jy):
            if filter_names[i] in ['f140m', 'f162m', 'f182m', 'f187n', 'f210m', 'f335m', 'f360m', 'f405n', 'f410m', 'f480m']:
                print(f"Flux for filter {filter_name} is NaN or non-positive, checking catalogs for upper limits...")
                cat_nrca = catalogs_filters_filtered[f'{filter_names[i]}_nrca']
                skycoord_nrca = Table.read(cat_nrca)['skycoord']
                cat_nrcb = catalogs_filters_filtered[f'{filter_names[i]}_nrcb']
                skycoord_nrcb = Table.read(cat_nrcb)['skycoord']
                # get the sources that are within 0.1 arcsec from the target source in the catalog, and use their fluxes as upper limits
                idx_nrca = skycoord_nrca.separation(sky_coordinates) < 0.1*u.arcsec
                idx_nrcb = skycoord_nrcb.separation(sky_coordinates) < 0.1*u.arcsec
                if np.any(idx_nrca):
                    flux_nrca = Table.read(cat_nrca)['flux_fit'][idx_nrca]
                    flux_nrca = flux_nrca[~np.isnan(flux_nrca)]
                    if len(flux_nrca) > 0:
                        print('flux_nrca:', flux_nrca)
                        flux_jy = np.max(flux_nrca* u.MJy / u.sr * wcs_filter.proj_plane_pixel_area()).to(u.Jy)
                if np.any(idx_nrcb):
                    flux_nrcb = Table.read(cat_nrcb)['flux_fit'][idx_nrcb]
                    flux_nrcb = flux_nrcb[~np.isnan(flux_nrcb)]
                    if len(flux_nrcb) > 0:
                        print('flux_nrcb:', flux_nrcb)
                        flux_jy = np.max(flux_nrcb* u.MJy / u.sr * wcs_filter.proj_plane_pixel_area()).to(u.Jy)
            elif filter_names[i] in ['f560w', 'f770w', 'f1000w', 'f1280w', 'f2100w']:
                cat_miri = catalogs_filters[f'{filter_names[i]}']
                skycoord_miri = Table.read(cat_miri)['skycoord']
                idx_miri = skycoord_miri.separation(sky_coordinates) < 0.1*u.arcsec
                if np.any(idx_miri):
                    flux_miri = Table.read(cat_miri)['flux_fit'][idx_miri]
                    flux_miri = flux_miri[~np.isnan(flux_miri)]
                    if len(flux_miri) > 0:
                        flux_jy = np.max(flux_miri* u.MJy / u.sr * wcs_filter.proj_plane_pixel_area()).to(u.Jy)
            """
            if np.isnan(flux_jy) or flux_jy <= 0  or np.ma.is_masked(flux_jy): # if the flux is still NaN or non-positive, get the pixel value within the area of FWHM at the center of the cutout as upper limit
                # get the pixel value within the area of FWHM at the center of the cutout
                aperture_reg = CirclePixelregion(center=(cutout.data.shape[1]/2, cutout.data.shape[0]/2), radius=(fwhm_pix/2))
                mask_cutout = aperture_reg.to_mask(method='center')
                cutout_masked = mask_cutout.multiply(cutout.data)
                flux = (np.sum(cutout_masked)*4* u.MJy / u.sr * wcs_filter.proj_plane_pixel_area()).to(u.Jy)
                is_upperlimit = True
            """        
            
        """        
        if np.isfinite(flux_jy) and filter_names[i] in ['f182m']:
            #flux_f182m = (flux* u.MJy / u.sr * wcs_filter.proj_plane_pixel_area()).to(u.Jy)
            hdr_f182m = fits.getheader(image_filenames['f182m'], ext=('SCI', 1))
            wcs_f182m = WCS(hdr_f182m, naxis=2)
            mag_f182m = get_mag(flux_jy, wcs_f182m, filtername='f182m')
        elif np.isfinite(flux_jy) and filter_names[i] in ['f210m']:
            #flux_f210m = (flux_jy* u.MJy / u.sr * wcs_filter.proj_plane_pixel_area()).to(u.Jy)
            hdr_f210m = fits.getheader(image_filenames['f210m'], ext=('SCI', 1))
            wcs_f210m = WCS(hdr_f210m, naxis=2)
            mag_f210m = get_mag(flux_jy, wcs_f210m, filtername='f210m')
        elif np.isfinite(flux_jy) and filter_names[i] in ['f360m']:
            #flux_f360m = (flux_jy* u.MJy / u.sr * wcs_filter.proj_plane_pixel_area()).to(u.Jy)
            hdr_f360m = fits.getheader(image_filenames['f360m'], ext=('SCI', 1))
            wcs_f360m = WCS(hdr_f360m, naxis=2)
            mag_f360m = get_mag(flux_jy, wcs_f360m, filtername='f360m')
        elif np.isfinite(flux_jy) and filter_names[i] in ['f480m']:
            #flux_f480 = (flux_jy* u.MJy / u.sr * wcs_filter.proj_plane_pixel_area()).to(u.Jy)
            hdr_f480m = fits.getheader(image_filenames['f480m'], ext=('SCI', 1))
            wcs_f480m = WCS(hdr_f480m, naxis=2)
            mag_f480m = get_mag(flux_jy, wcs_f480m, filtername='f480m')
        """
        print(f"Filter: {filter_name}, Wavelength: {wav} micron, Flux: {flux_jy}")

        flux_value = flux_jy.to_value(u.Jy) if hasattr(flux_jy, 'to_value') else float(flux_jy)
#        if is_upperlimit:
 #           ax_main.errorbar(wav , flux_value, yerr=flux_value*0.5, uplims=True, color=colors[i], marker='o', markersize=20, label=filter_name.upper())
        
        ax_main.plot(wav , flux_value, color = colors[i], marker='o', markersize=20, 
            label=filter_name.upper())
        
        ax_main.vlines(wav, ymin=1e-10, ymax=1e10, colors=colors[i], linestyles='dashed', alpha=0.5)
        fluxarr.append(flux_value)
    print('fluxarr',fluxarr)
    fluxarr = np.array(fluxarr, dtype=float)
    ax_main.set_xscale('log')
    ax_main.set_yscale('log')
    ax_main.set_xlabel('Wavelength (micron)')
    ax_main.set_ylabel('Flux (Jy)')
    
    ax_main.text(0.7, 0.9, f'SED for Source {label}', transform=ax_main.transAxes, fontsize=26)
    #if np.isfinite(mag_f182m) and np.isfinite(mag_f210m) and np.isfinite(mag_f360m) and np.isfinite(mag_f480m):
    #    ax_main.text(0.7,0.8, f'f182m-210m: {(mag_f182m-mag_f210m).value:.2f}, f360m-480m: {(mag_f360m-mag_f480m).value:.2f} mag', transform=ax_main.transAxes, fontsize=14, bbox=dict(facecolor='white', alpha=0.7))
    #ax_main.legend(fontsize=12, ncol=4, bbox_to_anchor=(0.55, 0, 0.2,0.4))
    ax_main.set_ylim(1e-8, 1e4)
    ax_main.set_xlim(1, 7e4)

    
    plt.tight_layout()
    plt.savefig(f'{SEDsavedir}/{alma_region}_source_{label}_SED.png')
    if show:
        plt.show()
    plt.close()
    """
    if np.isfinite(mag_f182m) and np.isfinite(mag_f210m) and np.isfinite(mag_f360m) and np.isfinite(mag_f480m):
        return mag_f182m, mag_f210m, mag_f360m, mag_f480m, label
    else:
        return None, None, None, None, label

    """



