# flake8: noqa: E262
import os
import numpy as np
import subprocess, glob
import datetime
import shutil
from astropy.table import Table
from tqdm.auto import tqdm
from astropy.io import fits
from filtering import get_fwhm, get_filtername
#\from brick2221.analysis.analysis_setup import basepath
basepath = '/orange/adamginsburg/w51/TaehwaYoo/jwst_w51/'

dolphot_binpath = os.getenv('DOLPHOT_BINPATH',
                            '/orange/adamginsburg/software/dolphot2.0/bin/')

default_params = {
    'Align': 0,             ### Align = 1: Align images to reference? Allowed values are 0 (no), 1 (x/y offsets only), 2 (x/y offsets plus scale difference), 3 (x/y offsets plus distortion), and 4 (full third-order polynomial fit).
    'UseWCS': 2,            ### UseWCS = 0: Use WCS header information for alignment? Allowed values are 0 (no), 1 (use to estimate shift, scale, and rotation), or 2 (use to estimate a full distortion solution); JWST requires UseWCS=2. UseWCS=2 should also be set when running JWST and HST data together, even though it is not optimal for HST photometry alone.
    'AlignTol': 0,          ### aligntol = 0: Tolerance on initial alignment solution. If greater than zero, DOLPHOT will search for matches out to the specified distance (in image pixels) from the initial guess. Must be a non-negative value.
    'Rotate': 0,            ### Rotate = 0: Correct for rotation in alignment? Allowed values are 0 (no) and 1 (yes).
    'img_shift': (0, 0),    ### The offset of a science image relative to reference image. This value can be an initial guess that is later adjusted by DOLPHOT. Values are x and y on the image minus x and y on the reference image. Note that this parameter should not be set for the reference image.
    'img_xform': (1, 0, 0), ### The scale ratio, cubic distortion, and rotation of a science image relative to the reference image. This value can be an initial guess that is later adjusted by DOLPHOT. This parameter should not be set for the reference image, only for science images (i.e., img1_xform to imgN_xform).

    # Combination Parameters
    # NOT A REAL PARAMETER 'CombineType': 'median',         # Use median for combining images
    # NOT A REAL PARAMETER 'CombineSigmaClip': 3.0,          # 3-sigma clipping to remove outliers
    # NOT A REAL PARAMETER 'CombineReject': 'crreject',     # Use cosmic-ray rejection during combination
    # NOT A REAL PARAMETER 'CombineScale': 'yes',           # Scale images before combining based on exposure time
    # NOT A REAL PARAMETER 'CombineWeight': 'noise',        # Weight by noise level during combination

    ####### Star-finding parameters #######
    'SecondPass': 5,      ### SecondPass = 1: Number of additional passes when finding stars to locate stars in the wings of brighter stars. Must be a non-negative value.
    'RCentroid': 1,       ### RCentroid = 1: The centroid used for obtaining initial positions of stars is a square of size 2RCentroid + 1 on each side
    'SearchMode': 1,      ### SearchMode = 1: Sets the minimization used to determine star positions. Allowed values are 0 (chi divided by SNR) and 1 (chi only). A value of one appears safe for all applications. A value of zero has been seen to fail if images of very different exposure times are used together.
    'SigFind': 2.,        ### SigFind = 2.5: Sigma detection threshold. Stars above this limit will be kept in the photometry until the final output.
    'SigFindMult': 0.85,  ### SigFindMult = 0.85: Multiple for sigma detection threshold in initial finding algorithm. This should be close to one for larger PSFs, and as low as 0.75 for badly undersampled PSFs.
    'SigFinal': 2.,       ### SigFinal = 3.5: Sigma threshold for a star to be listed in the final photometry list. To get all stars, set SigFinal equal to SigFind.
    'PosStep': 0.25,      ### PosStep = 0.25: Typical stepsize in x and y during photometry iterations. Should be set to a factor of a few smaller than the PSF FHWM.
    'RCombine': 1.5,      ### RCombine = 2.0: Minimum separation of two stars (they will be combined if they become closer than this value). This value can generally be about 2/3 of the PSF FWHM, but setting below 1.4 will not always be effective.
    'FSat': 0.999,        ### FSat = 0.999: Fraction of nominal saturation for which pixels are considered saturated.

    ####### Photometry #######
    'PSFPhot': 1,         ### PSFPhot = 1: Type of photometry to be run. Options are 0 (aperture), 1 (standard PSF-fit), 2 (PSF-fit weighted for central pixels). Option 1 is suggested for most photometric needs, but option 0 can provide superior photometry if the signal-to-noise is high and the field is uncrowded.
    'PSFPhotIt': 2,       ### PSFPhotIT = 2: Number of iterations on the PSF photometry solution, if PSFPhot is 1 or 2. This will refine the noise estimates on the pixels based on the model fit.
    'FitSky': 2,          ### FitSky = 1: Sky-fitting setting. Options are 0 (use
                        ### the sky map from calcsky), 1 (fit the sky normally prior to each photometry
                        ### measurement), 2 (fit the sky inside the PSF region but outside the photometry
                        ### aperture), 3 (fit the sky within the photometry aperture as a 2-parameter PSF
                        ### fit), and 4 (fit the sky within the photometry aperture as a 4-parameter PSF
                        ### fit). Options 1 and 3 are the suggested settings. Option 0 should be used only
                        ### if the field is very uncrowded; option 2 can be used in extremely crowded
                        ### fields; option 4 can help in fields with strong background gradients. FitSky=4
                        ### is unstable and not recommended except in extreme gradients. In general,
                        ### FitSky=2 provides the most robust results across a wide range of crowding.
    'SkipSky': 1,         ### SkipSky = 1: Sampling of sky annulus; set to a number higher than 1 to gain speed at the expense of precision. This is only used if FitSky is set to 1. In general, this should never be larger than the FWHM of the PSF.
    'SkySig': 2.25,       ### SkySig = 2.25: Sigma rejection threshold for sky fit; only used if FitSky is set to 1.
    'MaxIT': 25,          ### MaxIT = 25: Maximum number of photometry iterations.
    'NoiseMult': 0.10,    ### NoiseMult = 0.05: To allow for imperfect PSFs, the noise is increased by this value times the star brightness in the pixel when computing chi values.
    'SigPSF': 5.0,        ### SigPSF = 10.0: Minimum signal-to-noise for a PSF solution to be attempted on a star. Fainter detections are assigned type 2.
    'CombineChi': 0,      ### CombineChi also affects the combined photometry blocks. If set to zero (default), photometry will be combined weighted by 1/sigma^2 to maximize signal to noise. If set to one, weights will be 1/sigma^2*max(1, chi^2) to reduce the impact of epochs with bad photometry

    ####### Other #######
    'Force1':  1,         ### Force1 = 0: Force all objects to be of class 1 or 2 (i.e., stars)? Allowed values are 0 (no) and 1 (yes). For crowded stellar fields, this should be set to 1 and the χ and sharpness values used to discard extended objects.
    'ApCor': 1,           ### ApCor = 1: Make aperture corrections? Allowed values are 0 (no) and 1 (yes). Default aperture corrections always have the potential for error, so it is strongly recommended that you manually examine the raw output from this process.
    'PSFres': 1,          ### PSFres = 1: Solve for PSF residual image? Allowed values are 0 (no) and 1 (yes). Turning this feature off can create nonlinearities in the photometry unless PSFphot is also set to zero.
    'FlagMask': 0,        ### FlagMask = 4: FlagMask is a bitwise mask that determines what error flags will not be accepted when producing the combined photometry blocks for each filter. A value of zero allows photometry with an error flag less than eight to be used. Adding one eliminates stars close to the chip edge, adding two eliminates stars with too many bad pixels, and adding four eliminates stars with saturated cores.
    'InterpPSFlib': 1,    ### If InterpPSFlib is set to 0, the PSF library will use the nearest X,Y position where a precalculated PSF is available rather than interpolating. The impact is approx. 1percent on the PSF shape but some speed improvement.
    'MIRIvega': 1,        ### If MIRIvega is set to 0, calibrated fluxes will be provided in units of Jy, and instrummental magnitudes in units of ABmag. Otherwise, when set to 1, Vega magnitudes are used and calibrated fluxes are scaled to for a zeroth magnitude source.
    ####### Camera Specific #######
    #'img_rsky': (0,0),     ### img_rsky (int int): Inner and outer radii for computing sky values, if FitSky=1 is being used. Also used in a few places if using FitSky = 2, 3, or 4, so should always be set. The inner radius (first number) should be outside the bulk of the light from the star; the outer (second) should be sufficiently large to compute an accurate sky.
    #'img_rsky2': (0,0),    ### img_rsky2 (int int)*: The annulus setting when using FitSky=2.
    #'img_RSF': (0,0),      ### img_RSF: The size of the PSF used for star subtraction, as well as the size of the PSF residual calculated if PSFRes is set to 1
    'img_apsky': (20,35),    ### img_apsky (int int): Set the inner and outer radii of the annulus used for calculating sky values for aperture corrections.
}


def write_params(filelist, filtername, suffix=''):
    params = default_params.copy()
    
    long = int(filtername[1]) > 2

    jj = 0
    for ii, fn in enumerate(filelist):

        header = fits.getheader(fn)
        fwhm_as, fwhm = get_fwhm(header)
        this_filtername = get_filtername(header)
        if this_filtername == filtername:

            # the example gives Nimg=60 and goes from 0 to 60, so there are 61 files...
            # unclear if img0 is special
            jj += 1

            perim_params = {
                # ?? 'img{ii}_rsky': (fwhm, 4*fwhm),
                f'img{jj}_rsky0': fwhm,
                f'img{jj}_rsky1': fwhm*4,
                # Error: RPSF needs to be at least RAper+2 if using FitSky=2
                f'img{jj}_rpsf': min([fwhm*6, 24]), # Size of the PSF used for star subtraction.  Cannot be > 24
                f'img{jj}_apsky': (3*fwhm, 4*fwhm),
                f'img{jj}_shift': (0, 0),
                f'img{jj}_xform': (1, 0, 0),
                f'img{jj}_file': os.path.splitext(os.path.basename(fn))[0],
            }

            if long:
                perim_params.update({
                    f'img{jj}_raper': 3.0,
                    f'img{jj}_rchi': 2.0,
                    f'img{jj}_rsky2': (4, 10),
                })
            else:
                perim_params.update({
                    f'img{jj}_raper': 2.0,
                    f'img{jj}_rchi': 1.5,
                    f'img{jj}_rsky2': (3, 10),
                })

            params.update(perim_params)

    params['Nimg'] = jj

    keys = list(params.keys())
    keys.remove('Nimg')
    keys.insert(0, 'Nimg')
    params = {key:params[key] for key in keys}

    paramfn = f'params_{filtername}{suffix}.txt'
    with open(paramfn, 'w') as fh:
        for key, val in params.items():
            if isinstance(val, tuple):
                val = ' '.join(str(v) for v in val)
            fh.write(f"{key} = {val}\n")

    return paramfn


def run_masking(filelist):
    for fn in tqdm(filelist, desc='nircammask'):
        if 'DQ' in fits.open(fn) or 'WHT' in fits.open(fn):
            # -etctime used to be required, now it's -noetctime to turn off
            subprocess.check_call(f"{dolphot_binpath}/nircammask {fn}".split())
            dd = fits.getdata(fn)
            if dd.min() == dd.max():
                raise ValueError("masking deleted the data instead of masking it.")
            assert np.nanmax(dd) > np.nanmin(dd), fn

def run_calcsky(filelist):
    for fn in tqdm(filelist, desc='calcsky'):
        if True: #not os.path.exists(f'{os.path.splitext(fn)[0]}.sky.fits'):
            subprocess.check_call(f"{dolphot_binpath}/calcsky {os.path.splitext(fn)[0]} 10 25 2 2.25 2.0".split())

            dd = fits.getdata(fn)
            if dd.min() == dd.max():
                raise ValueError("sky subtraction deleted data.")
            assert np.nanmax(dd) > np.nanmin(dd), fn

def assemble_data(filtername, return_full_filelist=True):

    dolpath = f'{basepath}/dolphot'
    if not os.path.exists(dolpath):
        os.mkdir(dolpath)

    #filelist = sorted(glob.glob(f'{basepath}/{filtername}/pipeline/*_cal.fits'))
    filelist = sorted(glob.glob(f'{basepath}/{filtername}/pipeline/*_destreak_1m1overf.fits'))
    print(f"Found {len(filelist)} matches for filter {filtername}")
    cped_filelist = []
    for fn in tqdm(filelist, desc='copy'):
        if True:#not os.path.exists(f'{dolpath}/{os.path.basename(fn)}'):
            shutil.copy(fn, dolpath)
            #print(f"Copied {fn} to {dolpath}")
            cped_filelist.append(f'{dolpath}/{os.path.basename(fn)}')

    if return_full_filelist:
        #full_filelist = glob.glob(f'{dolpath}/*_cal.fits')
        full_filelist = glob.glob(f'{dolpath}/*_destreak_1m1overf.fits')
        return full_filelist
    else:
        return cped_filelist


def run_dolphot(paramfn):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dolpath = f'{basepath}/dolphot'
    #subprocess.check_call(f"{dolphot_binpath}/dolphot jwst_dolph_{timestamp}.phot -p{paramfn} > {dolpath}/dolphot_{timestamp}.log".split())
    subprocess.check_call(f"{dolphot_binpath}/dolphot jwst_dolph_{timestamp}.phot -p{paramfn}".split())


def read_dolphot(basename):

    colnames = "Extension Chip X Y ChiFit SNRfit Sharp Round PA Crowd ObjectType PassNo TotalCounts TotalSky NormalizedCountRate eNormalizedCountRate InstrumentalVegaMag TransformedUBVRIMagnitude MagnitudeUncertainty Chi SNR Sharpness Roundness Crowding PhotometryQualityFlag".split()
    with open(basename, 'r') as fh:
        lines = [x.split()[:len(colnames)] for x in fh.readlines()]
    tbl = Table(rows=lines, names=colnames, dtype=[int]*2 + [float]*6 + [int, float, int, int] + [float]*12 + [int])

    return tbl


def main():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--filternames", dest="filternames",
                    default='F140M,F150W,F182M,F187N,F210M,F335M,F360M,F405N,F410M,F480M',
                    help="filter name list", metavar="filternames")
    (options, args) = parser.parse_args()

    filternames = options.filternames.split(",")


    for filtername in filternames:
        filelist = assemble_data(filtername)

        for fn in tqdm(filelist, desc='precheck'):
            dd = fits.getdata(fn)
            assert np.nanmax(dd) > np.nanmin(dd), fn

        paramfn = write_params(filelist, filtername)

        run_masking(filelist)

        for fn in tqdm(filelist, desc='check-postmask'):
            dd = fits.getdata(fn)
            assert np.nanmax(dd) > np.nanmin(dd), fn

        run_calcsky(filelist)

        for fn in tqdm(filelist, desc='check-prephot'):
            dd = fits.getdata(fn)
            assert np.nanmax(dd) > np.nanmin(dd), fn

        run_dolphot(paramfn)


if __name__ == "__main__":
    main()

"""
sbatch --job-name=webb-cat-dolph --output=web-cat-dolph-%j.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=8 --nodes=1 --mem=256gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python310/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/dolphot.py "
"""