"""NIRSpec MOS S/N and saturation vs NIRCam magnitude (F140M, F162M, F210M, F410M) with pandeia.

A flat-f_nu point source is stepped in brightness; for each run, the S/N per pixel and the number of
groups before saturation are read at the pivot wavelength of every NIRCam filter that falls in the
grating's range (F140M/F162M -> G140H/F100LP, F210M -> G235H/F170LP, F410M -> G395H/F290LP), and the
flux density is converted to a Vega magnitude with the SVO NIRCam zero points (as in get_mag in
nirspec_mos_target_selection.ipynb). Both S/N and saturation are local to that wavelength, so the
results do not depend on the SED shape elsewhere in the spectrum.

Same MOS setup as pandeia_nirspec_mos_etc.py: 1x3 shutter, 3 nods (nexp=3), nint=1, full frame.

    source /orange/adamginsburg/jwst/pandeia/env.sh
    /blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python pandeia_nirspec_mos_mag_grid.py NPROC [readout1,readout2,...]
"""
import os
import sys
import itertools
import warnings
from multiprocessing import Pool

import numpy as np

from pandeia_nirspec_mos_etc import GRATINGS, NEXP, w51_background

HERE = os.path.dirname(os.path.abspath(__file__))

# SVO JWST/NIRCam pivot wavelength (um) and Vega zero point (Jy), and the grating covering it
FILTERS = {
    'F140M': (1.40532, 1288.649, 'g140h_f100lp'),
    'F162M': (1.62725, 1023.039, 'g140h_f100lp'),
    'F210M': (2.09545, 688.0325, 'g235h_f170lp'),
    'F410M': (4.08224, 208.7505, 'g395h_f290lp'),
}
MAG_GRATINGS = sorted({g for _, _, g in FILTERS.values()})
READOUTS = ['nrsirs2rapid', 'nrsirs2']
NGROUPS = list(range(5, 21))
# flat f_nu flux densities (mJy), 0.5 mag steps from 100 mJy to ~3e-4 mJy
FLUX_MJY = 100.0 * 10**(-0.4 * 0.5 * np.arange(28))
WINDOW = 0.01  # um, half width of the wavelength window around each pivot


def build_calc(grating, readout, ngroup, bg, flux_mjy):
    from pandeia.engine.calc_utils import build_default_calc
    disperser, filt, ref_wave = GRATINGS[grating]
    c = build_default_calc('jwst', 'nirspec', 'mos')
    c['configuration']['instrument'].update(disperser=disperser, filter=filt)
    c['configuration']['detector'].update(readout_pattern=readout, ngroup=ngroup, nint=1,
                                          nexp=NEXP, subarray='full')
    c['strategy']['reference_wavelength'] = ref_wave
    spec = c['scene'][0]['spectrum']
    spec['sed'] = dict(sed_type='flat', unit='fnu', z=0.0)
    spec['normalization'] = dict(type='at_lambda', norm_wave=2.0, norm_flux=float(flux_mjy),
                                 norm_fluxunit='mjy', norm_waveunit='microns')
    if bg == 'minzodi':
        c['background'] = 'minzodi'
        c['background_level'] = 'benchmark'
    else:
        c['background'] = w51_background(bg)
        c.pop('background_level', None)
    return c


def run_one(args):
    grating, readout, ngroup, bg, flux_mjy = args
    from pandeia.engine.perform_calculation import perform_calculation
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = perform_calculation(build_calc(grating, readout, ngroup, bg, flux_mjy))
    wave, sn = np.asarray(r['1d']['sn'])
    _, npart = np.asarray(r['1d']['n_partial_saturated'])
    _, nfull = np.asarray(r['1d']['n_full_saturated'])
    ngmap = np.asarray(r['2d']['ngroups_map'])  # groups before saturation, per detector pixel
    rows = []
    for band, (pivot, zp_jy, g) in FILTERS.items():
        if g != grating:
            continue
        sel = np.abs(wave - pivot) < WINDOW
        good = sel & np.isfinite(sn)
        rows.append(dict(filter=band, grating=grating, readout=readout, ngroup=ngroup, nint=1,
                         nexp=NEXP, bg=bg, flux_mjy=float(flux_mjy),
                         mag=float(-2.5 * np.log10(flux_mjy * 1e-3 / zp_jy)),
                         t_exp_total=float(r['scalar']['total_exposure_time']),
                         sn=float(np.median(sn[good])) if good.any() else np.nan,
                         sat_ngroups_local=int(ngmap[:, sel].min()),
                         n_partial_local=int(npart[sel].max()),
                         n_full_local=int(nfull[sel].max()),
                         sat_ngroups_global=int(r['scalar']['sat_ngroups'])))
    return rows


def main(nproc, readouts=None):
    from astropy.table import Table
    # default: READOUTS -> pandeia_nirspec_mos_mag_grid.ecsv; otherwise one file per readout set
    suffix = '' if readouts is None else '_' + '_'.join(readouts)
    readouts = READOUTS if readouts is None else readouts
    grid = list(itertools.product(MAG_GRATINGS, readouts, NGROUPS, ['minzodi', 'p50', 'p84'], FLUX_MJY))
    print(f'{len(grid)} calculations on {nproc} processes', flush=True)
    rows = []
    with Pool(nproc) as pool:
        for i, res in enumerate(pool.imap(run_one, grid, chunksize=4)):
            rows.extend(res)
            if i % 500 == 0:
                print(i, res[0], flush=True)
    Table(rows=rows).write(os.path.join(HERE, f'pandeia_nirspec_mos_mag_grid{suffix}.ecsv'), overwrite=True)
    print('done', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1, sys.argv[2].split(',') if len(sys.argv) > 2 else None)
