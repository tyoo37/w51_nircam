import glob
import pickle
from collections import defaultdict
base = "/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/sed_fitting"

import os
from datetime import datetime
import numpy as np
from collections import OrderedDict
from tqdm.auto import tqdm
all_gs = glob.glob('/blue/adamginsburg/richardson.t/research/flux/r+24_models-1.2/s*')
all_gs = [g.split('/')[-1] for g in all_gs if 'ipynb' not in g]
# Cache pars tables for each geometry
pars_cache = {}
param_names = set()


def merge_geom_pickles(pattern):
    merged = {}
    files = sorted(glob.glob(pattern))
    print(f"Found {len(files)} files for pattern: {pattern}")
    
    for fn in files:
        #print last modfied date and time
        print(f"Processing file: {fn}, last modified date and time: {datetime.fromtimestamp(os.path.getmtime(fn))}")
        with open(fn, "rb") as f:
            d = pickle.load(f)

        for idx, entry in d.items():
            """
            if not isinstance(entry, dict):
                continue
            if 'model_id' not in entry or 'chi2' not in entry:
                continue
            if len(entry['model_id']) == 0 or len(entry['chi2']) == 0:
                continue
            """
            merged[idx] = entry
         

    return merged
combined_upper_dict = merge_geom_pickles(f"{base}/model_dict_upper_*.pkl")
combined_lower_dict = merge_geom_pickles(f"{base}/model_dict_lower_*.pkl")


with open(f"{base}/model_dict_upper_0.pkl", "rb") as f:
    d = pickle.load(f)
    print(len(d))
with open(f"{base}/model_dict_upper_1.pkl", "rb") as f:
    d = pickle.load(f)
    print(len(d))

all_idx_upper = [idx for idx, entry in combined_upper_dict.items()]
all_idx_lower = [idx for idx, entry in combined_lower_dict.items()]
all_upper_rows = [
    {'source_idx': src_idx, 'model_id': model_id, 'chi2': chi2}
    for src_idx, entry in combined_upper_dict.items()
    for model_id, chi2 in zip(entry['model_id'], entry['chi2'])
]

all_model_names_upper = [
    model_name
    for entry in combined_upper_dict.values()
    for model_name in entry['model_name']
]
all_model_ids_upper = [
    model_id
    for entry in combined_upper_dict.values()
    for model_id in entry['model_id']
]

all_chi2_upper = [
    chi2
    for entry in combined_upper_dict.values()
    for chi2 in entry['chi2']
]

all_stage_upper = [
    stage   
for entry in combined_upper_dict.values()
    for stage in entry['stage']
]

all_class_upper = [
    class_
    for entry in combined_upper_dict.values()
    for class_ in entry['class']
]

all_geo_upper = [
    geo
    for entry in combined_upper_dict.values()
    for geo in entry['geom']
]

all_lower_rows = [
    {'source_idx': src_idx, 'model_id': model_id, 'chi2': chi2}
    for src_idx, entry in combined_lower_dict.items()
    for model_id, chi2 in zip(entry['model_id'], entry['chi2'])
]
all_model_ids_lower = [
    model_id
    for entry in combined_lower_dict.values()
    for model_id in entry['model_id']
]
all_chi2_lower = [
    chi2
    for entry in combined_lower_dict.values()
    for chi2 in entry['chi2']
]

all_geo_lower = [
    geo
    for entry in combined_lower_dict.values()
    for geo in entry['geom']
]

all_stage_lower = [
    stage
    for entry in combined_lower_dict.values()
    for stage in entry['stage']
]
all_class_lower = [
    class_
    for entry in combined_lower_dict.values()
    for class_ in entry['class']    
]
all_model_names_lower = [
    model_name
    for entry in combined_lower_dict.values()
    for model_name in entry['model_name']
]
# Automatically collect all available parameters from each geometry's pars table



for g in tqdm(all_gs):
    geo = g.split('/')[-1]
    try:
        pars_tab = Table.read(f'/blue/adamginsburg/richardson.t/research/flux/pars/{geo}_augmented.fits')
        print(pars_tab.colnames)
        pars_cache[geo] = pars_tab
        # collect parameter names (skip MODEL_NAME)
        for col in pars_tab.colnames:
            if col == 'MODEL_NAME':
                continue
            param_names.add(col)
    except Exception as e:
        pars_cache[geo] = None

param_names = sorted(param_names)

# Build per-parameter arrays for upper and lower selections (aligned with all_geo_upper / all_geo_lower)
params_upper = OrderedDict([(p, []) for p in param_names])
params_lower = OrderedDict([(p, []) for p in param_names])

def _extract_value(pars, pname, idx):
    if pars is None or pname not in pars.colnames:
        return np.nan
    try:
        if pars[pname].ndim == 1:
            v = pars[pname][idx].item()
        elif pars[pname].ndim ==2:
            v = pars[pname][idx,3]
        # strip astropy Quantity
        if hasattr(v, 'value'):
            return float(v.value)
        return float(v)
    except Exception:
        return np.nan

for geo, model_name in zip(all_geo_upper, all_model_names_upper):
    pars = pars_cache.get(geo)
    if pars is None:
        for p in param_names:
            params_upper[p].append(np.nan)
        continue
    match_idx = np.where(pars['MODEL_NAME'] == model_name)[0]
    mi = match_idx[0] if match_idx.size>0 else None
    for p in param_names:
        if mi is None:
            params_upper[p].append(np.nan)
        else:
            params_upper[p].append(_extract_value(pars, p, mi))
for geo, model_name in zip(all_geo_lower, all_model_names_lower):
    pars = pars_cache.get(geo)
    if pars is None:
        for p in param_names:
            params_lower[p].append(np.nan)
        continue
    match_idx = np.where(pars['MODEL_NAME'] == model_name)[0]
    mi = match_idx[0] if match_idx.size>0 else None
    for p in param_names:
        if mi is None:
            params_lower[p].append(np.nan)
        else:
            params_lower[p].append(_extract_value(pars, p, mi))






from sedfitter.extinction import Extinction
from dust_extinction.shapes import P92
from dust_extinction.averages import RL85_MWGC, CT06_MWLoc
import astropy.units as u


all_distances_lower = [
    dist
    for entry in combined_lower_dict.values()
    for dist in entry['distance']
]

all_distances_upper = [
    dist
    for entry in combined_upper_dict.values()
    for dist in entry['distance']
]

all_avs_lower = [
    av
    for entry in combined_lower_dict.values()
    for av in entry['av']
]

all_avs_upper = [
    av
    for entry in combined_upper_dict.values()
    for av in entry['av']
]
def make_extinction():
    """
    Best-fit parameters:
    BKG_amp = 250
    BKG_lambda = 0.02
    BKG_b = 407.27
    BKG_n = 1.5907
    SIL1_amp = 0.00194345
    SIL1_lambda = 9.8
    SIL1_b = -1.97467
    SIL1_n = 1.5
    SIL2_amp = 0.0346621
    SIL2_lambda = 18.6
    SIL2_b = -1.43583
    SIL2_n = 1.5
    """
    best_params = {'BKG_amp': 250, 'BKG_lambda': 0.02, 'BKG_b': 407.27, 'BKG_n': 1.5907, 'SIL1_amp': 0.00194345, 'SIL1_lambda': 9.8, 'SIL1_b': -1.97467, 'SIL1_n': 1.5, 'SIL2_amp': 0.0346621, 'SIL2_lambda': 18.6, 'SIL2_b': -1.43583, 'SIL2_n': 1.5}
    ext_p92 = P92(**best_params)
    ext_ct06 = CT06_MWLoc()
    guyver2009_avtocol = (2.21e21 * u.cm**-2 * (1.34 * u.Da)).to(u.g / u.cm**2)
    ext_wav = np.sort((np.geomspace(0.001, 1000, 10000) / u.um).to(u.um, u.spectral()))
    wav_ct06 = np.geomspace(1.28, 25, 1000)*u.um  # Wavelength range from 0.1 to 3 microns
    ext_val_p92 = ext_p92(ext_wav)
    ext_val_ct06 = ext_ct06(wav_ct06)
    wav_short = np.where(ext_wav<1.28*u.um)[0]
    wav_mid = np.where((ext_wav>=1.28*u.um) & (ext_wav<=26.5*u.um))[0]
    wav_long = np.where(ext_wav>26.5*u.um)[0]
     
    ext_val = np.zeros_like(ext_wav.value)
    ext_val[wav_mid] = ext_ct06((1.0 / ext_wav[wav_mid]).to(1 / u.um))
    slope = (ext_val[wav_mid][0] - ext_val[wav_mid][1]) / (ext_wav[wav_mid][0].value - ext_wav[wav_mid][1].value)
    intercept = ext_val_ct06[0] - slope * wav_ct06[0].value

    ext_val[wav_short] = slope * ext_wav[wav_short].value + intercept
    ext_val[wav_long] = ext_val_p92[wav_long]

    extinction = Extinction()
    extinction.wav = ext_wav
    extinction.chi = ext_val / guyver2009_avtocol
    return extinction

extinction = make_extinction()

from sedfitter.sed import SEDCube
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astroquery.svo_fps import SvoFps
import matplotlib.pyplot as plt
import warnings
from astropy.wcs import FITSFixedWarning

warnings.filterwarnings(
    "ignore",
    category=FITSFixedWarning,
    module=r"astropy\.wcs\.wcs"
)
def get_line_excess(sed, sed_wav, narrow, wide):
    # --- Convert catalog MJy/sr → Jy ---
    #sed = u.Quantity(sed).to_value(u.Jy)
    sed_wav = u.Quantity(sed_wav).to_value(u.AA)
    sed = u.Quantity(sed).to_value(u.Jy)
   
    ww_narrow = WCS(fits.getheader(image_filenames[narrow], ext=('SCI', 1)))
    ww_wide = WCS(fits.getheader(image_filenames[wide], ext=('SCI', 1)))



    # --- Load filter curves ---
    tab_n = SvoFps.get_transmission_data(f'JWST/NIRCAM.{narrow}')
    tab_w = SvoFps.get_transmission_data(f'JWST/NIRCAM.{wide}')

    wave_n = tab_n['Wavelength'] * u.AA
    trans_n = tab_n['Transmission']

    wave_w = tab_w['Wavelength'] * u.AA
    trans_w = tab_w['Transmission']
   
    # --- Interpolate narrow onto wide grid ---
    trans_n_interp = np.interp(sed_wav, wave_n.value, trans_n)
    trans_w_interp = np.interp(sed_wav, wave_w.value, trans_w)



    data_narrow = np.sum(sed * trans_n_interp) / np.sum(trans_n_interp)
    data_wide = np.sum(sed * trans_w_interp) / np.sum(trans_w_interp)
 

    # --- Fractional overlap (proper integral) ---
    num = np.trapezoid(
    (trans_w_interp / trans_w_interp.max()) * (trans_n_interp / trans_n_interp.max()),
    sed_wav
    )
    den = np.trapezoid(trans_w_interp / trans_w_interp.max(), sed_wav)
    f = num / den

    if np.isclose(f, 1.0):
        return None, None

    # --- Continuum & line (still in Jy = F_nu) ---
    cont_nu = (data_wide - data_narrow * f) / (1 - f)
    line_nu = data_narrow - cont_nu

    width_narrow = np.trapezoid(trans_n_interp / trans_n_interp.max(), sed_wav) * u.AA

    # --- Equivalent Width ---
    with np.errstate(divide='ignore', invalid='ignore'):
        ew = (line_nu / cont_nu) * width_narrow

    return line_nu



image_filenames ={
    "f140m": "/orange/adamginsburg/jwst/w51/images/jw06151-o001_t001_nircam_clear-f140m-merged_i2d.fits",
    "f162m": "/orange/adamginsburg/jwst/w51/images/jw06151-o001_t001_nircam_clear-f162m-merged_i2d.fits",
    "f182m": "/orange/adamginsburg/jwst/w51/images/jw06151-o001_t001_nircam_clear-f182m-merged_i2d.fits",
    "f187n": "/orange/adamginsburg/jwst/w51/images/jw06151-o001_t001_nircam_clear-f187n-merged_i2d.fits",
    "f210m": "/orange/adamginsburg/jwst/w51/images/jw06151-o001_t001_nircam_clear-f210m-merged_i2d.fits",
    "f335m": "/orange/adamginsburg/jwst/w51/images/jw06151-o001_t001_nircam_clear-f335m-merged_i2d.fits",
    "f360m": "/orange/adamginsburg/jwst/w51/images/jw06151-o001_t001_nircam_clear-f360m-merged_i2d.fits",
    "f405n": "/orange/adamginsburg/jwst/w51/images/jw06151-o001_t001_nircam_clear-f405n-merged_i2d.fits",
    "f410m": "/orange/adamginsburg/jwst/w51/images/jw06151-o001_t001_nircam_clear-f410m-merged_i2d.fits", # weird, the filename is different from what is downloaded with the STScI pipeline...
    "f480m": "/orange/adamginsburg/jwst/w51/images/jw06151-o001_t001_nircam_clear-f480m-merged_i2d.fits",
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

filter_names = list(image_filenames.keys())
# remove f187n and f405n from the list
filter_names.remove('f187n')
filter_names.remove('f405n')    

fwhms_arcsec = []
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
    fwhms_arcsec.append(fwhm_arcsec)
fwhms_au = np.array(fwhms_arcsec)*5400
robitaille_modeldir='/blue/adamginsburg/richardson.t/research/flux/r+24_models-1.2'
paa_average_upper = []
paa_std_upper = []
bra_average_upper = []
bra_std_upper = []


catalog = Table.read('/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/catalogs/final_catalog_new_cfit_cut.fits')
nmatch = catalog['nmatch_bands']
catalog = catalog[nmatch>3]


aperture_grid = np.logspace(2,6,20)


cat_wav = np.array([1.4, 1.62, 1.82, 2.10, 3.35, 3.60, 4.10, 4.80, 5.60, 7.70, 10.00, 12.80, 21.00])*u.um

plot =False

bra_excess = []
paa_excess = []
for ii in range(len(all_idx_upper)):
    if True:
        cat_idx = all_idx_upper[ii]
        i10_arr = np.arange(ii*10, (ii+1)*10)
        if plot:
            fig = plt.figure(figsize=(18,6))
            ax = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
        for jj, i10 in enumerate(i10_arr):
            
            geo = all_geo_upper[i10]
            model_id = all_model_ids_upper[i10]
            chi2 = all_chi2_upper[i10]
            model_dir = f'{robitaille_modeldir}/{geo}'
            sedcube = SEDCube.read(f"{model_dir}/flux.fits",)
            distance = all_distances_upper[i10]
            av = all_avs_upper[i10]
            modelname = all_model_names_upper[i10]
            sed = sedcube.get_sed(modelname)
            #apnum = np.argmin(np.abs(default_aperture - sedcube.apertures))
            # https://github.com/astrofrog/sedfitter/blob/41dee15bdd069132b7c2fc0f71c4e2741194c83e/sedfitter/sed/sed.py#L64
            #av_scale = fitinfo.av[index]
            distance_scale = (1 / distance)**2
            apertures_au = np.array(fwhms_arcsec)*distance*1e3
            sed_wav_inside = sed.wav[(sed.wav > cat_wav.min()) & (sed.wav < cat_wav.max())]
            sed_flux_inside = sed.flux[:,(sed.wav > cat_wav.min()) & (sed.wav < cat_wav.max())]
            sed_wav_inside = sed_wav_inside[::-1]
            sed_flux_inside = sed_flux_inside[:,::-1]
            print('sed_wav_inside', sed_wav_inside)
            apertures_au_interp = np.interp(sed_wav_inside, cat_wav[::-1], apertures_au)
            av_scale_conv = np.array([
                10**(av * extinction.get_av(wavelength).item())
            
                for wavelength in sed_wav_inside
            ])
            # interpolate sed_flux based on the aperture size
            print('aperture_grid.shape', aperture_grid.shape)
            print('sed_flux_inside.shape', sed_flux_inside.shape)
            sed_flux_interp = interp1d(
                aperture_grid,
                np.log10(sed_flux_inside.value.T),
                axis=1,
                kind='linear'
            )

            # apertures_au_interp: (42 wavelengths,)
            # output: (42 wavelengths,)
            sed_flux_interp_apertures = np.diagonal(
                sed_flux_interp(apertures_au_interp)
            )

            extincted_sed_for_plot = (
                10**sed_flux_interp_apertures
                * 1e-3 * u.Jy
                * distance_scale
                * av_scale_conv
            )

            deextincted_sed = (
                10**sed_flux_interp_apertures
                * 1e-3 * u.Jy
                * distance_scale
            )

            print(extincted_sed_for_plot.shape)

            #sed_flux_interp_paa = interp1d(np.log10(np.ones_like(apertures_au_interp)*apertures_au_interp[2]), np.log10(sed_flux_inside.value), axis=0, kind='linear')
            #sed_flux_interp_bra = interp1d(np.log10(np.ones_like(apertures_au_interp)*apertures_au_interp[6]), np.log10(sed_flux_inside.value), axis=0, kind='linear')
            deextincted_sed_paa = 10**np.diagonal(sed_flux_interp(np.ones_like(apertures_au_interp)*fwhms_au[2])) * 1e-3 *u.Jy * distance_scale
            deextincted_sed_bra = 10**np.diagonal(sed_flux_interp(np.ones_like(apertures_au_interp)*fwhms_au[6] ))* 1e-3 *u.Jy * distance_scale
      
            paa_line= get_line_excess(deextincted_sed_paa, sed_wav_inside, 'f187n', 'f182m')
            bra_line = get_line_excess(deextincted_sed_bra, sed_wav_inside, 'f405n', 'f410m')
        
            gradient_paa = np.gradient(deextincted_sed_paa, sed_wav_inside)[np.argmin(np.abs(sed_wav_inside.value-1.87))]
            gradient_bra = np.gradient(deextincted_sed_bra, sed_wav_inside)[np.argmin(np.abs(sed_wav_inside.value-4.05))]
            print('gradient_paa:', gradient_paa)
            print('gradient_bra:', gradient_bra)
            print('paa_line:', paa_line)
            print('bra_line:', bra_line)
            print('chi2:', chi2)
            if plot:
                flux_obs = []
                for kk, filt in enumerate(filter_names):
                    ww = WCS(fits.getheader(image_filenames[filt], ext=('SCI', 1)))
                    flux_jy = (catalog[f'flux_fit_{filt.lower()}'][cat_idx] * u.MJy/u.sr * ww.proj_plane_pixel_area()).to(u.Jy)
                    ax.scatter(cat_wav[kk].value, flux_jy.value, marker='o', c='k')
                    flux_obs.append(flux_jy.value)
                if jj==0:
                    alpha=1
                else:
                    alpha = 1-0.08*jj
                ax.plot(sed_wav_inside, extincted_sed_for_plot, alpha=alpha, color=plt.get_cmap('rainbow_r')(jj / (len(i10_arr) - 1)))
            
            
                ax.plot(sed_wav_inside, deextincted_sed, alpha=alpha, color=plt.get_cmap('rainbow_r')(jj / (len(i10_arr) - 1)), ls='dashed')
                ax.axvline(1.87, color='r', linestyle='--', label='Pa-alpha')
                ax.axvline(4.05, color='g', linestyle='--', label='Br-alpha')
                ax.set_xlabel('Wavelength (micron)')
                ax.set_ylabel('Flux (Jy)')
                ax.set_xscale('log')
                ax.set_yscale('log')
               
                ax2.scatter(gradient_paa, paa_line, label='Pa-alpha', color=plt.get_cmap('rainbow_r')(jj / (len(i10_arr) - 1)), marker='o')
                ax2.scatter(gradient_bra, bra_line, label='Br-alpha', color=plt.get_cmap('rainbow_r')(jj / (len(i10_arr) - 1)), marker='x')
                ax2.set_xlabel('Gradient of Deextincted SED (Jy/micron)')
                ax2.set_ylabel('Line excess flux (Jy)')

                ax3.scatter(gradient_paa, paa_line, label='Pa-alpha', color=plt.get_cmap('rainbow_r')(jj / (len(i10_arr) - 1)), marker='o')
                ax3.scatter(gradient_bra, bra_line, label='Br-alpha', color=plt.get_cmap('rainbow_r')(jj / (len(i10_arr) - 1)), marker='x')
                ax3.set_xlabel('Gradient of Deextincted SED (Jy/micron)')
                ax3.set_ylabel('Line excess flux (Jy)')
                ax3.set_xscale('log')
                ax3.set_yscale('log')
               

            ax.set_xlim(1.4, 5)
            #ax.set_ylim(extincted_sed[np.argmin(np.abs(sedcube.wav.value-1))].value*0.1, extincted_sed[np.argmin(np.abs(sedcube.wav.value-25))].value*10)
            paa_excess.append(paa_line)
            bra_excess.append(bra_line)
            """
            # for every 10 models, save their average and standard deviation of the line fluxes 
            if ii % 10 == 0:
                paa_excess = []
                bra_excess = []
                paa_lines = []
                paa_ews = []
                bra_lines = []
                bra_ews = []
            else:
                paa_lines.append(paa_line)
               
                bra_lines.append(bra_line)
            
            
            if ii % 10 == 9:
                paa_average_upper.append(np.mean(paa_lines))
                paa_std_upper.append(np.std(paa_lines))
                bra_average_upper.append(np.mean(bra_lines))
                bra_std_upper.append(np.std(bra_lines))
            """
# write the paa_excess and bra_excess to a pickle file
with open(f"{base}/paa_bra_excess_upper.pkl", "wb") as f:
    pickle.dump({'paa_excess': paa_excess, 'bra_excess': bra_excess}, f)


bra_excess = []
paa_excess = []
for ii in range(len(all_idx_lower)):
    if True:
        cat_idx = all_idx_lower[ii]
        i10_arr = np.arange(ii*10, (ii+1)*10)
        if plot:
            fig = plt.figure(figsize=(18,6))
            ax = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)
        for jj, i10 in enumerate(i10_arr):
            
            geo = all_geo_lower[i10]
            model_id = all_model_ids_lower[i10]
            chi2 = all_chi2_lower[i10]
            model_dir = f'{robitaille_modeldir}/{geo}'
            sedcube = SEDCube.read(f"{model_dir}/flux.fits",)
            distance = all_distances_lower[i10]
            av = all_avs_lower[i10]
            modelname = all_model_names_lower[i10]
            sed = sedcube.get_sed(modelname)
            #apnum = np.argmin(np.abs(default_aperture - sedcube.apertures))
            # https://github.com/astrofrog/sedfitter/blob/41dee15bdd069132b7c2fc0f71c4e2741194c83e/sedfitter/sed/sed.py#L64
            #av_scale = fitinfo.av[index]
            distance_scale = (1 / distance)**2
            apertures_au = np.array(fwhms_arcsec)*distance*1e3
            sed_wav_inside = sed.wav[(sed.wav > cat_wav.min()) & (sed.wav < cat_wav.max())]
            sed_flux_inside = sed.flux[:,(sed.wav > cat_wav.min()) & (sed.wav < cat_wav.max())]
            sed_wav_inside = sed_wav_inside[::-1]
            sed_flux_inside = sed_flux_inside[:,::-1]
            print('sed_wav_inside', sed_wav_inside)
            apertures_au_interp = np.interp(sed_wav_inside, cat_wav[::-1], apertures_au)
            av_scale_conv = np.array([
                10**(av * extinction.get_av(wavelength).item())
            
                for wavelength in sed_wav_inside
            ])
            # interpolate sed_flux based on the aperture size
            print('aperture_grid.shape', aperture_grid.shape)
            print('sed_flux_inside.shape', sed_flux_inside.shape)
            sed_flux_interp = interp1d(
                aperture_grid,
                np.log10(sed_flux_inside.value.T),
                axis=1,
                kind='linear'
            )

            # apertures_au_interp: (42 wavelengths,)
            # output: (42 wavelengths,)
            sed_flux_interp_apertures = np.diagonal(
                sed_flux_interp(apertures_au_interp)
            )

            extincted_sed_for_plot = (
                10**sed_flux_interp_apertures
                * 1e-3 * u.Jy
                * distance_scale
                * av_scale_conv
            )

            deextincted_sed = (
                10**sed_flux_interp_apertures
                * 1e-3 * u.Jy
                * distance_scale
            )

            print(extincted_sed_for_plot.shape)

            #sed_flux_interp_paa = interp1d(np.log10(np.ones_like(apertures_au_interp)*apertures_au_interp[2]), np.log10(sed_flux_inside.value), axis=0, kind='linear')
            #sed_flux_interp_bra = interp1d(np.log10(np.ones_like(apertures_au_interp)*apertures_au_interp[6]), np.log10(sed_flux_inside.value), axis=0, kind='linear')
            deextincted_sed_paa = 10**np.diagonal(sed_flux_interp(np.ones_like(apertures_au_interp)*fwhms_au[2])) * 1e-3 *u.Jy * distance_scale
            deextincted_sed_bra = 10**np.diagonal(sed_flux_interp(np.ones_like(apertures_au_interp)*fwhms_au[6] ))* 1e-3 *u.Jy * distance_scale
      
            paa_line= get_line_excess(deextincted_sed_paa, sed_wav_inside, 'f187n', 'f182m')
            bra_line = get_line_excess(deextincted_sed_bra, sed_wav_inside, 'f405n', 'f410m')
        
            gradient_paa = np.gradient(deextincted_sed_paa, sed_wav_inside)[np.argmin(np.abs(sed_wav_inside.value-1.87))]
            gradient_bra = np.gradient(deextincted_sed_bra, sed_wav_inside)[np.argmin(np.abs(sed_wav_inside.value-4.05))]
            print('gradient_paa:', gradient_paa)
            print('gradient_bra:', gradient_bra)
            print('paa_line:', paa_line)
            print('bra_line:', bra_line)
            print('chi2:', chi2)
            if plot:
                flux_obs = []
                for kk, filt in enumerate(filter_names):
                    ww = WCS(fits.getheader(image_filenames[filt], ext=('SCI', 1)))
                    flux_jy = (catalog[f'flux_fit_{filt.lower()}'][cat_idx] * u.MJy/u.sr * ww.proj_plane_pixel_area()).to(u.Jy)
                    ax.scatter(cat_wav[kk].value, flux_jy.value, marker='o', c='k')
                    flux_obs.append(flux_jy.value)
                if jj==0:
                    alpha=1
                else:
                    alpha = 1-0.08*jj
                ax.plot(sed_wav_inside, extincted_sed_for_plot, alpha=alpha, color=plt.get_cmap('rainbow_r')(jj / (len(i10_arr) - 1)))
            
            
                ax.plot(sed_wav_inside, deextincted_sed, alpha=alpha, color=plt.get_cmap('rainbow_r')(jj / (len(i10_arr) - 1)), ls='dashed')
                ax.axvline(1.87, color='r', linestyle='--', label='Pa-alpha')
                ax.axvline(4.05, color='g', linestyle='--', label='Br-alpha')
                ax.set_xlabel('Wavelength (micron)')
                ax.set_ylabel('Flux (Jy)')
                ax.set_xscale('log')
                ax.set_yscale('log')
               
                ax2.scatter(gradient_paa, paa_line, label='Pa-alpha', color=plt.get_cmap('rainbow_r')(jj / (len(i10_arr) - 1)), marker='o')
                ax2.scatter(gradient_bra, bra_line, label='Br-alpha', color=plt.get_cmap('rainbow_r')(jj / (len(i10_arr) - 1)), marker='x')
                ax2.set_xlabel('Gradient of Deextincted SED (Jy/micron)')
                ax2.set_ylabel('Line excess flux (Jy)')

                ax3.scatter(gradient_paa, paa_line, label='Pa-alpha', color=plt.get_cmap('rainbow_r')(jj / (len(i10_arr) - 1)), marker='o')
                ax3.scatter(gradient_bra, bra_line, label='Br-alpha', color=plt.get_cmap('rainbow_r')(jj / (len(i10_arr) - 1)), marker='x')
                ax3.set_xlabel('Gradient of Deextincted SED (Jy/micron)')
                ax3.set_ylabel('Line excess flux (Jy)')
                ax3.set_xscale('log')
                ax3.set_yscale('log')
               

            ax.set_xlim(1.4, 5)
            #ax.set_ylim(extincted_sed[np.argmin(np.abs(sedcube.wav.value-1))].value*0.1, extincted_sed[np.argmin(np.abs(sedcube.wav.value-25))].value*10)
            paa_excess.append(paa_line)
            bra_excess.append(bra_line)
            """
            # for every 10 models, save their average and standard deviation of the line fluxes 
            if ii % 10 == 0:
                paa_excess = []
                bra_excess = []
                paa_lines = []
                paa_ews = []
                bra_lines = []
                bra_ews = []
            else:
                paa_lines.append(paa_line)
               
                bra_lines.append(bra_line)
            
            
            if ii % 10 == 9:
                paa_average_upper.append(np.mean(paa_lines))
                paa_std_upper.append(np.std(paa_lines))
                bra_average_upper.append(np.mean(bra_lines))
                bra_std_upper.append(np.std(bra_lines))
            """
# write the paa_excess and bra_excess to a pickle file
with open(f"{base}/paa_bra_excess_lower.pkl", "wb") as f:
    pickle.dump({'paa_excess': paa_excess, 'bra_excess': bra_excess}, f)
