"""NIRSpec MOS exposure-time grid with pandeia for the W51 Cycle 6 proposal.

For each mock/percentile SED (wavelength in um, flux density in mJy, as written by
nirspec_mos_target_selection.ipynb), run pandeia over the gratings in the APT MSA plans
(left_pa20_shut3_config12.json / right_PA10_shut3_config12 (Obs 2).json), the four MOS
full-frame readout patterns, and a grid of ngroup / nint, for three backgrounds:
minzodi (benchmark), and the 50th / 84th percentile diffuse W51 surface brightness measured
from the NIRCam medium-band mosaics inside the two MSA fields.

Setup: point source centred in a 1x3 shutter slitlet, 3-shutter nod (nexp=3),
msafullapphot extraction with nod background subtraction (pandeia default MOS strategy).

Environment (pandeia.engine 2026.7 lives in the python313 env):
    source /orange/adamginsburg/jwst/pandeia/env.sh
    /blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python pandeia_nirspec_mos_etc.py NPROC [_nint1_long]
"""
import os
import sys
import itertools
import warnings
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

SED_FILES = {
    'upper_veryred_bright': 'upper_branch_veryred_sed_bright.dat',
    'upper_veryred_faint': 'upper_branch_veryred_sed_faint.dat',
    'parsec_3msun_av60': 'parsec_nircam_3msun_av60.dat',
    'parsec_50msun_av12': 'parsec_nircam_50msun_av12.dat',
}

# grating/filter combinations in the APT MSA plans, with the S/N reference wavelength
GRATINGS = {
    'g140h_f070lp': ('g140h', 'f070lp', 1.10),
    'g140h_f100lp': ('g140h', 'f100lp', 1.62),
    'g235h_f170lp': ('g235h', 'f170lp', 2.10),
    'g395h_f290lp': ('g395h', 'f290lp', 4.10),
}

READOUTS = ['nrsrapid', 'nrs', 'nrsirs2rapid', 'nrsirs2']
NGROUPS = [2, 3, 4, 6, 8, 12, 16, 20]
NINTS = [1, 2, 4]
NGROUPS_LONG = [30, 40, 60, 80, 100]
NEXP = 3  # 3-shutter nod

# NIRCam band wavelengths at which the SEDs are measured; S/N is also reported at these
SN_WAVES = [1.40, 1.62, 1.82, 2.10, 3.35, 3.60, 4.10, 4.80]

# median / 84th percentile surface brightness (MJy/sr) of the NIRCam medium-band mosaics
# inside 3.4'x3.4' boxes at the two MSA pointings (all pixels, so stars are included but
# contribute little to the percentiles)
BG_WAVE = np.array([1.40, 1.62, 1.82, 2.10, 3.35, 3.60, 4.10, 4.80])
BG_P50 = np.array([0.740, 1.013, 1.881, 1.815, 13.01, 9.372, 11.72, 13.62])
BG_P84 = np.array([0.999, 1.583, 4.409, 3.907, 30.05, 22.47, 32.68, 36.04])


def loglog_extend(w, f, wgrid):
    """Log-log interpolation of the photometric SED onto wgrid, extended beyond the first/last
    points as a power law through the two outermost points on each side."""
    lw, lf, lg = np.log10(w), np.log10(f), np.log10(wgrid)
    out = np.interp(lg, lw, lf)
    lo, hi = lg < lw[0], lg > lw[-1]
    out[lo] = lf[0] + (lf[1] - lf[0]) / (lw[1] - lw[0]) * (lg[lo] - lw[0])
    out[hi] = lf[-1] + (lf[-1] - lf[-2]) / (lw[-1] - lw[-2]) * (lg[hi] - lw[-1])
    return 10**out


def load_sed(name):
    w, f = np.loadtxt(os.path.join(HERE, SED_FILES[name])).T
    wgrid = np.logspace(np.log10(0.6), np.log10(5.6), 5000)
    return wgrid, loglog_extend(w, f, wgrid), (w.min(), w.max())


def w51_background(level):
    """W51 diffuse background in MJy/sr, log-log interpolated between the measured bands and
    held constant outside 1.4-4.8 um."""
    s = {'p50': BG_P50, 'p84': BG_P84}[level]
    w = np.logspace(np.log10(0.5), np.log10(6.0), 3000)
    return [w.tolist(), (10**np.interp(np.log10(w), np.log10(BG_WAVE), np.log10(s))).tolist()]


def build_calc(sed_name, grating, readout, ngroup, nint, bg):
    from pandeia.engine.calc_utils import build_default_calc
    disperser, filt, ref_wave = GRATINGS[grating]
    c = build_default_calc('jwst', 'nirspec', 'mos')
    c['configuration']['instrument'].update(disperser=disperser, filter=filt)
    c['configuration']['detector'].update(readout_pattern=readout, ngroup=ngroup, nint=nint,
                                          nexp=NEXP, subarray='full')
    c['strategy']['reference_wavelength'] = ref_wave
    wgrid, fgrid, _ = load_sed(sed_name)
    spec = c['scene'][0]['spectrum']
    spec['sed'] = dict(sed_type='input', spectrum=[wgrid.tolist(), fgrid.tolist()])
    spec['normalization'] = dict(type='none')
    if bg == 'minzodi':
        c['background'] = 'minzodi'
        c['background_level'] = 'benchmark'
    else:
        c['background'] = w51_background(bg)
        c.pop('background_level', None)
    return c


def run_one(args):
    sed_name, grating, readout, ngroup, nint, bg = args
    from pandeia.engine.perform_calculation import perform_calculation
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = perform_calculation(build_calc(sed_name, grating, readout, ngroup, nint, bg))
    sc = r['scalar']
    wave, sn = np.asarray(r['1d']['sn'])
    _, npart = np.asarray(r['1d']['n_partial_saturated'])
    _, nfull = np.asarray(r['1d']['n_full_saturated'])
    wmin_sed, wmax_sed = load_sed(sed_name)[2]
    good = np.isfinite(sn) & (sn > 0)
    # restrict the S/N summary to the wavelengths covered by actual SED points (no extrapolation)
    insed = good & (wave >= wmin_sed) & (wave <= wmax_sed)
    row = dict(sed=sed_name, grating=grating, readout=readout, ngroup=ngroup, nint=nint,
               nexp=NEXP, bg=bg,
               t_exp_total=float(sc['total_exposure_time']),
               t_exp_per_nod=float(sc['exposure_time']),
               sn_ref=float(sc['sn']), ref_wave=float(sc['reference_wavelength']),
               sn_med_all=float(np.median(sn[good])) if good.any() else np.nan,
               sn_med_sed=float(np.median(sn[insed])) if insed.any() else np.nan,
               sn_p10_sed=float(np.percentile(sn[insed], 10)) if insed.any() else np.nan,
               frac_wave_in_sed=float(insed.sum() / max(good.sum(), 1)),
               fraction_saturation=float(sc['fraction_saturation']),
               sat_ngroups=int(sc['sat_ngroups']),
               brightest_pixel_eps=float(sc['brightest_pixel']),
               frac_wave_partial_sat=float(np.mean(npart > 0)),
               frac_wave_full_sat=float(np.mean(nfull > 0)),
               warnings=';'.join(sorted(r.get('warnings', {}).keys())))
    for w0 in SN_WAVES:
        sel = good & (np.abs(wave - w0) < 0.01)
        row[f'sn_{w0:.2f}'] = float(np.median(sn[sel])) if sel.any() else np.nan
    return row, wave.astype(np.float32), np.where(np.isfinite(sn), sn, 0).astype(np.float32)


def main(nproc, suffix=''):
    from astropy.table import Table
    if suffix == '_nint1_long':
        # nint=1 only: longer ramps for the faint SEDs
        grid = list(itertools.product(SED_FILES, GRATINGS, READOUTS, NGROUPS_LONG, [1],
                                      ['minzodi', 'p50', 'p84']))
    else:
        grid = list(itertools.product(SED_FILES, GRATINGS, READOUTS, NGROUPS, NINTS,
                                      ['minzodi', 'p50', 'p84']))
    print(f'{len(grid)} calculations on {nproc} processes', flush=True)
    rows, spectra = [], {}
    with Pool(nproc) as pool:
        for i, (row, wave, sn) in enumerate(pool.imap(run_one, grid, chunksize=4)):
            rows.append(row)
            key = '|'.join(str(row[k]) for k in ('sed', 'grating', 'readout', 'ngroup', 'nint', 'bg'))
            spectra[key + '|wave'] = wave
            spectra[key + '|sn'] = sn
            if i % 200 == 0:
                print(i, key, f"sn_ref={row['sn_ref']:.1f} fsat={row['fraction_saturation']:.2f}",
                      flush=True)
    tab = Table(rows=rows)
    tab.write(os.path.join(HERE, f'pandeia_nirspec_mos_grid{suffix}.ecsv'), overwrite=True)
    np.savez_compressed(os.path.join(HERE, f'pandeia_nirspec_mos_grid{suffix}_1d_sn.npz'), **spectra)
    print('done', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1, sys.argv[2] if len(sys.argv) > 2 else '')
