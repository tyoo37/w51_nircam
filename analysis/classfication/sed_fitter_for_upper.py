import sys
sys.path.append('/blue/adamginsburg/t.yoo/SPICY_ALMAIMF')
import imp, table_loading
imp.reload(table_loading)
from table_loading import fit_a_source, geometries
from astropy.table import Table
from plot_fit import plot_fit, datafunction
import numpy as np
from astropy import constants
from astropy import units as u
from astropy.modeling import models, fitting

from photutils.aperture import SkyCircularAperture
from photutils.aperture import aperture_photometry
from photutils.aperture import SkyCircularAnnulus
from photutils.aperture import ApertureStats
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
#from filtering import get_filtername, get_fwhm
import plot_fit
from astroquery.svo_fps import SvoFps
from dust_extinction.averages import RL85_MWGC, CT06_MWLoc
import plot_fit
imp.reload(plot_fit)
from plot_fit import plot_fit, datafunction
from scipy.ndimage import binary_dilation
from matplotlib.lines import Line2D
import os
import glob

catalog = Table.read('/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/final_catalog_new_.fits')
nmatch = catalog['nmatch_bands']
catalog = catalog[nmatch>3]

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
    "f2100w": "/orange/adamginsburg/jwst/w51/F2100W/pipeline/jw06151-o002_t001_miri_f2100w_i2d.fits",
    
}


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

def get_mag(catalog, ww, filtername='f140m' ):
    #print(ww.proj_plane_pixel_area())
    
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

f187n_header = fits.getheader(image_filenames['f187n'], ext=('SCI', 1))
f187n_mag = get_mag(catalog, WCS(f187n_header), filtername='f187n')

f162 = f162m_mag[0].to_value(u.mag)
f210 = f210m_mag[0].to_value(u.mag)
f360 = f360m_mag[0].to_value(u.mag)
f480 = f480m_mag[0].to_value(u.mag)

num_sources = len(catalog)

from dust_extinction.averages import RL85_MWGC, CT06_MWGC
from dust_extinction.parameter_averages import CCM89
import astropy.constants as const

ext = CT06_MWLoc()

mags = {
    'f140m': f140m_mag,
    'f162m': f162m_mag,
    'f182m': f182m_mag,
   # 'f187n': f187n_mag,
    'f210m': f210m_mag,
    'f335m': f335m_mag,
    'f360m': f360m_mag,
    #'f405n': f405n_mag,
    'f410m': f410m_mag,
    'f480m': f480m_mag,
    'f560w': f560w_mag,
    'f770w': f770w_mag,
    'f1000w': f1000w_mag,
    'f1280w': f1280w_mag,
    'f2100w': f2100w_mag,
}


def get_av(color, mag, ext=CT06_MWLoc(), color1=('f140m', 'f162m'), mag1=('f140m',), isochrone_line_params=None):
    # Get the slope and intercept of the isochrone line
    m_iso, b_iso = isochrone_line_params
    print('color1:', color1, 'mag1:', mag1)

    # Get the extinction vector components
    w_color1_1 = int(color1[0][1:-1])/100*u.um
    w_color1_2 = int(color1[1][1:-1])/100*u.um
    w_mag1 = int(mag1[0][1:-1])/100*u.um
    print('w_color1_1:', w_color1_1, 'w_color1_2:', w_color1_2, 'w_mag1:', w_mag1)

    e_color1_1 = ext(1/w_color1_1)
    e_color1_2 = ext(1/w_color1_2)
    e_mag1 = ext(1/w_mag1)

    # get the contact to the isochrone line along the extinction vector
    # the extinction vector can be represented as a line with slope m_ext = e_mag1 / (e_color1_1 - e_color1_2) and passing through the point (color, mag)
    m_ext = e_mag1 / (e_color1_1 - e_color1_2)
    b_ext = mag - m_ext * color

    # find the intersection of the isochrone line and the extinction vector
    x_int = (b_ext - b_iso) / (m_iso - m_ext)
    y_int = m_iso * x_int + b_iso

    # calculate A_V as the distance from (color, mag) to (x_int, y_int) along the extinction vector
    av = (mag - y_int) / ext(1/w_mag1)
    av[av < 0] = 0
    return av
parsec_av0 = Table.read('/home/t.yoo/parsec_av0.txt', format='ascii')
parsec_av0.pprint(max_width=-1, max_lines=-1)   
m_init = parsec_av0['col4']  
parsec_av0 = parsec_av0[m_init<300]              
logage_av0 = parsec_av0['col3']
mass_av0 = parsec_av0['col6']
#print(np.min(m_init), np.max(m_init))
#print(np.min(mass_av0), np.max(mass_av0))
F140Mmag_av0 = parsec_av0['col39'] + 5*np.log10(5400)-5
F162Mmag_av0 = parsec_av0['col40']+ 5*np.log10(5400)-5
F182Mmag_av0 = parsec_av0['col41']+ 5*np.log10(5400)-5
F210Mmag_av0 = parsec_av0['col42']+ 5*np.log10(5400)-5
F335Mmag_av0 = parsec_av0['col45']+ 5*np.log10(5400)-5
F360Mmag_av0 = parsec_av0['col46']+ 5*np.log10(5400)-5
F410Mmag_av0 = parsec_av0['col47']+ 5*np.log10(5400)-5
F480Mmag_av0 = parsec_av0['col50']+ 5*np.log10(5400)-5
age_idx = logage_av0 == 6
idx_01msun = np.argmin(np.abs(mass_av0[age_idx]-0.1))
idx_20msun = np.argmin(np.abs(mass_av0[age_idx]-20))
isochrone_x_start = F162Mmag_av0[age_idx][idx_01msun] - F210Mmag_av0[age_idx][idx_01msun]
isochrone_x_end = F162Mmag_av0[age_idx][idx_20msun] - F210Mmag_av0[age_idx][idx_20msun]
isocrhone_y_start = F162Mmag_av0[age_idx][idx_01msun]
isocrhone_y_end = F162Mmag_av0[age_idx][idx_20msun]
color_slope = (isocrhone_y_end - isocrhone_y_start) / (isochrone_x_end - isochrone_x_start)
color_intercept = isocrhone_y_start - color_slope * isochrone_x_start
av_estimates = get_av(color=f162-f210, mag=f162, ext=CT06_MWLoc(), color1=('f162m', 'f210m'), mag1=('f162m',), isochrone_line_params=(color_slope, color_intercept))


from sedfitter.extinction import Extinction
from dust_extinction.shapes import P92

from astropy.wcs import WCS
from regions import Regions
from itertools import chain
from astropy import stats
import pickle
import math

w51e_b6_noise_region = '/orange/adamginsburg/w51/TaehwaYoo/regions/w51e_b6_std_sky_new.reg'
w51e_b3_noise_region = '/orange/adamginsburg/w51/TaehwaYoo/regions/w51e_b3_std_sky_new.reg'
w51n_b6_noise_region = '/orange/adamginsburg/w51/TaehwaYoo/regions/w51n_b6_std_sky_new.reg'
w51n_b3_noise_region = '/orange/adamginsburg/w51/TaehwaYoo/regions/w51n_b3_std_sky_new.reg'

read_w51e_b3_noise = Regions.read(w51e_b3_noise_region,format='ds9')
read_w51e_b6_noise = Regions.read(w51e_b6_noise_region,format='ds9')
read_w51n_b3_noise = Regions.read(w51n_b3_noise_region,format='ds9')
read_w51n_b6_noise = Regions.read(w51n_b6_noise_region,format='ds9')



def get_noise(fitsdata, noiseregion):
    
    image = fitsdata[0].data
    if len(image.shape)!=2:
        image = image[0][0] 
    hdrNB = fitsdata[0].header
    wcsNB = WCS(hdrNB,naxis=2)
    container = []
    for reg in noiseregion:
        pix_reg = reg.to_pixel(wcsNB)
        noisemask = pix_reg.to_mask()
        noiseim = noisemask.cutout(image)
        container.append(noiseim.flatten())
    noiseim = list(chain.from_iterable(container))
    std = stats.mad_std(noiseim,ignore_nan=True)

    return std
def add_alma(skycoord):
    from radio_beam import Beam
    from photutils.aperture import SkyEllipticalAperture
    from photutils.aperture import aperture_photometry
    from astropy.wcs import WCS

    w51e_b3_file = '/orange/adamginsburg/w51/TaehwaYoo/2017.1.00293.S_W51_B3_LB/may2021_successful_imaging/w51e2.spw0thru19.14500.robust0.thr0.075mJy.mfs.I.startmod.selfcal7.image.tt0.pbcor.fits'
    w51n_b3_file = '/orange/adamginsburg/w51/TaehwaYoo/2017.1.00293.S_W51_B3_LB/may2021_successful_imaging/w51n.spw0thru19.14500.robust0.thr0.075mJy.mfs.I.startmod.selfcal7.image.tt0.pbcor.fits'
    w51n_b6_file = '/orange/adamginsburg/w51/TaehwaYoo/w51n_b6_imaging_2025/w51n.spw0thru19.14500.robust0.thr0.1mJy.mfs.I.startmod.selfcal7.image.tt0.pbcor.fits'
    w51e_b6_file = '/orange/adamginsburg/w51/TaehwaYoo/w51e_b6_imaging_2025/w51e2.spw0thru19.14500.robust0.thr0.1mJy.mfs.I.startmod.selfcal7.image.tt0.pbcor.fits'

    w51e_alma_cat_file = '/orange/adamginsburg/w51/TaehwaYoo/dendro_w51e_master.fits'
    w51n_alma_cat_file = '/orange/adamginsburg/w51/TaehwaYoo/dendro_w51n_master.fits'
    w51e_b3_fits = fits.open(w51e_b3_file)
    w51n_b3_fits = fits.open(w51n_b3_file)
    w51n_b6_fits = fits.open(w51n_b6_file)
    w51e_b6_fits = fits.open(w51e_b6_file)
    
    w51e_b3_hdr = w51e_b3_fits[0].header
    w51n_b3_hdr = w51n_b3_fits[0].header
    w51n_b6_hdr = w51n_b6_fits[0].header
    w51e_b6_hdr = w51e_b6_fits[0].header

    w51e_b3_wcs = WCS(w51e_b3_hdr)
    w51n_b3_wcs = WCS(w51n_b3_hdr)
    w51n_b6_wcs = WCS(w51n_b6_hdr)
    w51e_b6_wcs = WCS(w51e_b6_hdr)
    
    w51e_b3_img = w51e_b3_fits[0].data[0][0]
    w51n_b3_img = w51n_b3_fits[0].data[0][0]
    w51n_b6_img = w51n_b6_fits[0].data[0][0]
    w51e_b6_img = w51e_b6_fits[0].data[0][0]

    w51e_alma_cat = Table.read(w51e_alma_cat_file)
    w51n_alma_cat = Table.read(w51n_alma_cat_file)

    # check whether skycoord is in the fov of the image
    fluxes = []
    flux_errs = []
    valids = []
    for alma_cat, b3_fits, b6_fits, b3_noise, b6_noise in zip([w51e_alma_cat, w51n_alma_cat], [w51e_b3_fits, w51n_b3_fits], [w51e_b6_fits, w51n_b6_fits], [read_w51e_b3_noise, read_w51n_b3_noise], [read_w51e_b6_noise, read_w51n_b6_noise]):
       # print(alma_cat.colnames)
        for band in ['b6', 'b3']:
            skycoord_from_almacat = SkyCoord(ra=alma_cat[f'{band}_xsky']*u.deg, dec=alma_cat[f'{band}_ysky']*u.deg)
            separation = skycoord.separation(skycoord_from_almacat)
            ismatched = np.min(separation) < 0.1*u.arcsec


            if ismatched:
                flux = alma_cat[f'flux_{band}'][np.argmin(separation)] * 1e3
                fluxerr = alma_cat[f'flux_err_{band}'][np.argmin(separation)] * 1e3
                print(f'matched with ALMA with flux: in {band} ', flux)

                valid = 1

            else:
                
                if band == 'b6':
                    b6_hdr = b6_fits[0].header
                    b6_img = b6_fits[0].data[0][0]
                    wcs_alma = WCS(b6_hdr, naxis=2)
                    img_alma = b6_img
                else:
                    b3_hdr = b3_fits[0].header
                    b3_img = b3_fits[0].data[0][0]
                    wcs_alma = WCS(b3_hdr, naxis=2)
                    img_alma = b3_img

                cel_wcs = wcs_alma.celestial
                pixcoord_alma = cel_wcs.world_to_pixel(skycoord)

                if (
                    pixcoord_alma[0] > 0 and pixcoord_alma[1] > 0
                    and pixcoord_alma[0] < img_alma.shape[1]
                    and pixcoord_alma[1] < img_alma.shape[0]
                ):
                    if band == 'b6':
                        beam_alma = Beam.from_fits_header(b6_hdr)
                        noiseregion = b6_noise
                        global_std = get_noise(b6_fits, noiseregion)

                    else:
                        beam_alma = Beam.from_fits_header(b3_hdr)
                        noiseregion = b3_noise
                        global_std = get_noise(b3_fits, noiseregion)
                    aperture = SkyEllipticalAperture(
                        positions=skycoord,
                        a=beam_alma.major,
                        b=beam_alma.minor,
                        theta=beam_alma.pa,
                    )
                    phot_table = aperture_photometry(img_alma, aperture,  wcs=wcs_alma)
                    #print(phot_table.colnames)
                    if phot_table['aperture_sum'][0] < 0:
                        print('negative flux in aperture photometry, setting flux to NaN')

                        area_pix = (beam_alma.sr / wcs_alma.proj_plane_pixel_area()).to(u.dimensionless_unscaled).value
                        print('area_pix:', area_pix, 'global_std:', global_std)
                        flux = phot_table['aperture_sum'] * 1e3 - global_std * area_pix * 1e3
                        print('aperture_sum:', phot_table['aperture_sum'], 'flux:', flux)
                        valid=3
                        flux= global_std * 1e3 * 3 # 3sigma
                        fluxerr=0.997
                    else:
                       
                        area_pix = (beam_alma.sr / wcs_alma.proj_plane_pixel_area()).to(u.dimensionless_unscaled).value
                        print('area_pix:', area_pix, 'global_std:', global_std)
                        flux = phot_table['aperture_sum'] * 1e3 - global_std * area_pix * 1e3
                        print('aperture_sum:', phot_table['aperture_sum'], 'flux:', flux)
                        fluxerr= global_std * np.sqrt(area_pix) * 1e3
                        valid = 3
                        if flux < 0:
                            print('negative flux after background subtraction, setting flux to NaN')
                            flux = global_std * 1e3 * 3
                            fluxerr = 0.997
                            valid = 3
                        print(f'inside fov of {band}, flux: {flux}')
                else:
                    flux = np.nan
                    fluxerr = np.nan
                    valid = 0
                    print(f'outside fov of {band}')
                    print('pixcoord_alma:', pixcoord_alma, 'img_alma.shape:', img_alma.shape)                
              
            if isinstance(flux, u.Quantity):
                flux = flux.value
            if isinstance(fluxerr, u.Quantity):
                fluxerr = fluxerr.value
            if isinstance(flux, np.ndarray):
                flux = flux.item()
            if isinstance(fluxerr, np.ndarray):
                fluxerr = fluxerr.item()
            fluxes.append(flux)
            flux_errs.append(fluxerr)
            valids.append(valid)
    b6_flux = [fluxes[0], fluxes[2]]
    b3_flux = [fluxes[1], fluxes[3]]   
    b3_valid = [valids[1], valids[3]]
    b6_valid = [valids[0], valids[2]]
    print('b3_flux:', b3_flux, 'b6_flux:', b6_flux, 'b3_valid:', b3_valid, 'b6_valid:', b6_valid)
    if np.all(np.isnan(b3_flux)):
        b3_flux_final = np.nan
        b3_fluxerr_final = np.nan
        b3_valid_final = 0
    else:
        b3_flux_final = np.nanmean(b3_flux)
        b3_fluxerr_final = np.sqrt(np.nansum(np.array([flux_errs[1], flux_errs[3]])**2)) / len([flux_errs[1], flux_errs[3]])
        b3_valid_final = 3
    if np.all(np.isnan(b6_flux)):
        b6_flux_final = np.nan
        b6_fluxerr_final = np.nan
        b6_valid_final = 0
    else:
        b6_flux_final = np.nanmean(b6_flux)
        b6_fluxerr_final = np.sqrt(np.nansum(np.array([flux_errs[0], flux_errs[2]])**2)) / len([flux_errs[0], flux_errs[2]])
        b6_valid_final = 3
    b3_valid_final = int(np.nanmax(b3_valid))
    b6_valid_final = int(np.nanmax(b6_valid))
    return b3_flux_final, b3_fluxerr_final, b6_flux_final, b6_fluxerr_final, b3_valid_final, b6_valid_final

def make_extinction():
    ext = P92()

    guyver2009_avtocol = (2.21e21 * u.cm**-2 * (1.34 * u.Da)).to(u.g / u.cm**2)
    ext_wav = np.sort((np.geomspace(0.001, 1000, 10000) / u.um).to(u.um, u.spectral()))
    ext_vals = ext(ext_wav)

    extinction = Extinction()
    extinction.wav = ext_wav
    extinction.chi = ext_vals / guyver2009_avtocol
    return extinction

extinction = make_extinction()


imgs ={filt: fits.getdata(image_filenames[filt], ext=('SCI', 1)) for filt in image_filenames.keys()} 
wcss = {filt: WCS(fits.getheader(image_filenames[filt], ext=('SCI', 1))) for filt in image_filenames.keys()}
max_fluxs = {filt: np.nanmax(catalog[f'flux_fit_{filt}']) for filt in image_filenames.keys()}
filter_names = list(mags.keys())


line_slope = (27 - 7) / (1.3 - 0.4)
line_intercept = 7 - line_slope * 0.4


upper_idx = np.where((f360-f480 > 0.15 + (1.5/6)*(f162-f210)) & (f162 - (line_slope * (f162 - f210) + line_intercept) < 0))[0]
lower_idx = np.where((f360-f480 <= 0.15 + (1.5/6)*(f162-f210)) & (f162 - (line_slope * (f162 - f210) + line_intercept) < 0))[0]


def get_valid(catalog_row, filters, imgs, wcss, max_fluxs, masks):
    valid = np.ones(len(filters), dtype=int)
    fluxarr = []

    for ii, filt in enumerate(filters):
        raw_flux = catalog_row[f'flux_fit_{filt}']
        

        flux = (raw_flux * u.MJy / u.sr * wcss[filt].proj_plane_pixel_area()).to(u.mJy).value

        if not np.ma.is_masked(raw_flux) and  np.isfinite(flux):
            print(f'flux is not masked for flux_fit_{filt}')
            valid[ii] = 1
            fluxarr.append(flux)
        else:
            print(f'flux is masked for flux_fit_{filt}')
            skycoord = catalog_row[f'skycoord_{filt}']
            pixcoord = wcss[filt].world_to_pixel(skycoord)
            central_pix = imgs[filt][int(pixcoord[1]), int(pixcoord[0])]

            if np.isnan(central_pix): #saturated
                valid[ii] = 2
                flux = max_fluxs[filt] * 1e3
            else: # nondetection
                valid[ii] = 3
                aperture = SkyCircularAperture(skycoord, r=fwhms[ii] * u.arcsec)
                aperture_area = (
                    fwhms[ii] * u.arcsec * fwhms[ii] * u.arcsec / wcss[filt].proj_plane_pixel_area()
                ).to_value(u.dimensionless_unscaled)
                annulus = SkyCircularAnnulus(
                    skycoord,
                    r_in=fwhms[ii] * 1.5 * u.arcsec,
                    r_out=fwhms[ii] * 2 * u.arcsec,
                )
                annulus_stats = ApertureStats(imgs[filt], annulus, wcs=wcss[filt], mask=masks[filt])
                bkg_mean = annulus_stats.mean
                phot_table = aperture_photometry(imgs[filt], aperture, wcs=wcss[filt])

                # 3sigma local background as flux upper limit for non detection
                flux = (
                    3* ( bkg_mean * aperture_area)
                    * u.MJy / u.sr * wcss[filt].proj_plane_pixel_area()
                ).to(u.mJy).value
                
                if flux < 0:
                    flux =0
                    valid[ii] = 0

                if np.isnan(flux):
                    valid[ii] = 2
                    flux = max_fluxs[filt] * 1e3

            fluxarr.append(flux)

    return valid.tolist(), np.array(fluxarr)

fwhms = []
fwhms_pix = []
wavelengths = []
for ii, filt in enumerate(filter_names):
    header = fits.getheader(image_filenames[filt], ext=('SCI', 1))
    if filt in ['f140m', 'f162m', 'f182m', 'f187n', 'f210m', 'f335m', 'f360m', 'f405n', 'f410m', 'f480m']:
        instrument_replacement = 'NIRCam'
    elif filt in ['f560w', 'f770w', 'f1000w', 'f1280w', 'f2100w']:
        instrument_replacement = 'MIRI'
    fwhm_tbl = Table.read('/orange/adamginsburg/w51/jwst/reduction/fwhm_table.ecsv')
    row = fwhm_tbl[fwhm_tbl['Filter'] == filt.upper()]
    fwhm = fwhm_arcsec = float(row['PSF FWHM (arcsec)'][0])
    pixel_scale = wcss[filt].proj_plane_pixel_scales()[0]
    fwhm_pix = fwhm_arcsec / pixel_scale.to_value(u.arcsec)
    #fwhm, fwhm_pix = get_fwhm(header, instrument_replacement=instrument_replacement)
    fwhms.append(fwhm_arcsec)
    fwhms_pix.append(fwhm_pix)
    wav = int(filt[1:-1])/100 #um
    wavelengths.append(wav)
fwhms = np.array(fwhms)
wavelengths = np.array(wavelengths)
masks = {}
for ii, filt in enumerate(filter_names):
    mask = np.isnan(imgs[filt])
    skycoord = catalog[f'skycoord_{filt}']
    pixcoord = wcss[filt].world_to_pixel(skycoord)

    mask[pixcoord[1].astype(int), pixcoord[0].astype(int)] = True
    R = int(np.ceil(fwhms_pix[ii]/2))
    yy, xx = np.ogrid[-R:R+1, -R:R+1]
    footprint = xx**2 + yy**2 <= R**2
    mask = binary_dilation(mask, structure=footprint)

    masks[filt] = mask





model_ids_upper={}
best_models_ids_upper={}
best_idx_upper={}

all_gs = glob.glob('/blue/adamginsburg/richardson.t/research/flux/r+24_models-1.2/s*')
all_gs = [g.split('/')[-1] for g in all_gs if 'ipynb' not in g]
for geo in all_gs:
    model_ids_upper[geo] = []
    best_models_ids_upper[geo] = []
    best_idx_upper[geo] = []

n_chunks = 20
chunk_size = math.ceil(len(upper_idx) / n_chunks)
for ii, idx in enumerate(upper_idx):
    if True:
        chunk_id = ii // chunk_size
        if chunk_id >= n_chunks:
            chunk_id = n_chunks - 1

        task_id = os.getenv("SLURM_ARRAY_TASK_ID")
        if task_id is not None and int(task_id) != chunk_id:
            continue

        for filt in filter_names:
            print(f'filt: {filt}, flux_fit: {catalog[idx][f"flux_fit_{filt}"]}, flux_err: {catalog[idx][f"flux_err_{filt}"]}')
        #fluxarr = np.array([catalog[idx][f'flux_fit_{filt}'] for filt in filter_names])
        skycoord = catalog[idx]['skycoord_ref']
        print('skycoord:', skycoord)
        pixcoord_f560w = wcss['f560w'].world_to_pixel(skycoord)
        if pixcoord_f560w[0] < 0 or pixcoord_f560w[1] < 0 or pixcoord_f560w[0] >= imgs['f560w'].shape[1] or pixcoord_f560w[1] >= imgs['f560w'].shape[0]:
            filternames_use = ['f140m', 'f162m', 'f182m', 'f210m', 'f335m', 'f360m', 'f410m', 'f480m']
            imgs_use = {filt: imgs[filt] for filt in filternames_use}
            wcss_use = {filt: wcss[filt] for filt in filternames_use}
            max_fluxs_use = {filt: max_fluxs[filt] for filt in filternames_use}
            masks_use = {filt: masks[filt] for filt in filternames_use}
            wavelengths_use = [wavelengths[filter_names.index(filt)] for filt in filternames_use]
            fwhms_use = [fwhms[filter_names.index(filt)] for filt in filternames_use]
        else:
            filternames_use = filter_names
            imgs_use = imgs
            wcss_use = wcss
            max_fluxs_use = max_fluxs
            masks_use = masks
            wavelengths_use = wavelengths
            fwhms_use = fwhms
        # check the unit of flux_err in the catalog, it is in MJy/sr, so we need to convert it to mJy
        #print('wcss[filt].proj_plane_pixel_area():', wcss['f140m'].proj_plane_pixel_area())
        #print('catalog[idx][f"flux_err_{filt}"]:', catalog[idx][f'flux_err_f140m'])
        fluxerr = u.Quantity([(catalog[idx][f'flux_err_{filt}'] * u.MJy / u.sr * wcss[filt].proj_plane_pixel_area().to(u.sr)).to(u.mJy) for filt in filternames_use]) # Jy -> mJy
        

        valid, fluxarr = get_valid(catalog[idx], filternames_use, imgs_use, wcss_use, max_fluxs_use, masks_use)
        #print('fluxarr', fluxarr)
        print('valid before filtering:', valid)
        valid = np.asarray(valid, dtype=int)
        valid_idx = valid != 0

        fwhms_use = np.asarray(fwhms_use)[valid_idx]
        wavelengths_use = np.asarray(wavelengths_use)[valid_idx]
        fluxarr = np.asarray(fluxarr)[valid_idx]
        fluxerr = np.asarray(fluxerr)[valid_idx]
        valid = valid[valid_idx]
        valid = valid.tolist()
        print('valid after filtering:', valid)
        tab_upper = Table()
        tab_upper['aperture'] = fwhms_use
        tab_upper['wavelength'] = wavelengths_use
        tab_upper['flux'] = fluxarr
        tab_upper['eflux'] = fluxerr
        tab_upper['aperture'].unit = u.arcsec
        tab_upper['wavelength'].unit = u.um
        tab_upper['flux'].unit = u.mJy
        tab_upper['eflux'].unit = u.mJy
        b3_flux, b3_fluxerr, b6_flux, b6_fluxerr, b3_valid, b6_valid = add_alma(catalog[idx]['skycoord_ref'])
        print('b3_flux, b3_fluxerr, b6_flux, b6_fluxerr:', b3_flux, b3_fluxerr, b6_flux, b6_fluxerr)
        print('b3_valid', b3_valid)
        print('b6_valid', b6_valid)
        if not np.isnan(b6_flux):
            tab_upper.add_row([fwhms[-1], 1300, b6_flux, b6_fluxerr])
            valid.append(b6_valid)
        if not np.isnan(b3_flux):
            tab_upper.add_row([fwhms[-1], 3000, b3_flux, b3_fluxerr])   
            valid.append(b3_valid)
        print('flux', tab_upper['flux'])
        print('wavelength', tab_upper['wavelength'])
        print('valid', valid)
        print('av_estimates[idx]:', av_estimates[idx])


        #print('av_estimates[idx]:', av_estimates[idx])
        fits_dict = {geom:
            fit_a_source(data=tab_upper['flux'].quantity,
                error=tab_upper['eflux'].quantity,
                valid=valid, aperture_size=tab_upper['aperture'].quantity,
                filters=tab_upper['wavelength'].quantity,
                    av_range=[av_estimates[idx], 100],
                    distance_range=[5300,5500]*u.pc,
                        geometry=geom,
                        stash_to_mmap=True,
                        robitaille_modeldir='/blue/adamginsburg/richardson.t/research/flux/r+24_models-1.2',
                        extinction=extinction
                    )
            for geom in geometries}#['spubhmi']} #geometries}

        minminchi2 = min([np.nanmin(fits_dict[geom].chi2) for geom in fits_dict])

        okgeo = []

        chi2limit = minminchi2*3 if minminchi2 > 3 else 9
        deltachi2limit = chi2limit - minminchi2

        #if np.isnan(minchi2):
        #    raise ValueError(f"minchi2 is NaN for source {idx}, check the input data and fitting process.")
        for geom in geometries:
          
            #print(f"Delta-chi^2 = {deltachi2limit} for min chi2 = {minchi2}")
            if np.nanmin(fits_dict[geom].chi2) < chi2limit:
                okgeo.append(geom)
        if len(okgeo)>0:
            for geom in okgeo:
                fitinfo = fits_dict[geom]

                selection = fitinfo.chi2 < chi2limit
                model_ids_upper[geom].extend(fitinfo.model_id[selection])

                minchi2 = np.nanmin(fits_dict[geom].chi2)
                if minchi2 == minminchi2:
                    best_models_ids_upper[geom].append(fits_dict[geom].model_id[np.nanargmin(fits_dict[geom].chi2)])
                    best_idx_upper[geom].append(idx)

# save model_ids_upper to a file
with open(f"/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/sed_fitting/model_ids_upper_{os.getenv('SLURM_ARRAY_TASK_ID')}.pkl", "wb") as f:
    pickle.dump(model_ids_upper, f)
with open(f"/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/sed_fitting/best_idx_upper_{os.getenv('SLURM_ARRAY_TASK_ID')}.pkl", "wb") as f:
    pickle.dump(best_idx_upper, f)
with open(f"/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/sed_fitting/best_models_ids_upper_{os.getenv('SLURM_ARRAY_TASK_ID')}.pkl", "wb") as f:
    pickle.dump(best_models_ids_upper, f)