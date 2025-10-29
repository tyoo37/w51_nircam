import glob
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import numpy as np
from regions import Regions
from astropy.nddata import Cutout2D
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, Column
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import QTable
import stpsf as webbpsf
import crowdsource
from crowdsource import crowdsource_base
import stpsf
from stpsf.utils import to_griddedpsfmodel
from filtering import get_filtername, get_fwhm
from astropy import wcs
from astropy.wcs import WCS
import os
from astropy import log
import urllib3
import functools
import requests
from astropy.visualization import simple_norm
from filtering import get_filtername, get_fwhm
from astropy.coordinates import SkyCoord
from regions import PixCoord
try:
    # version >=1.7.0, doesn't work: the PSF is broken (https://github.com/astropy/photutils/issues/1580?)
    from photutils.psf import PSFPhotometry, IterativePSFPhotometry, SourceGrouper, make_psf_model_image
except:
    # version 1.6.0, which works
    from photutils.psf import BasicPSFPhotometry as PSFPhotometry, IterativelySubtractedPSFPhotometry as IterativePSFPhotometry, DAOGroup as SourceGrouper
try:
    from photutils.background import MMMBackground, MADStdBackgroundRMS, MedianBackground, Background2D, LocalBackground
except:
    from photutils.background import MMMBackground, MADStdBackgroundRMS, MedianBackground, Background2D
    from photutils.background import MMMBackground as LocalBackground
#from photutils.detection.core import _findobj, findobj
from astropy.nddata import NDData

def daofind_metrics_at_positions(data, x,y, fwhm=3.0, threshold=0):
    """
    Compute DAO-like shape parameters (sharpness, roundness1, roundness2)
    for known positions.

    - data : 2D numpy array (full image)
    - positions : iterable of (x, y) pairs (pixel coordinates)
    - fwhm : float, approximate FWHM in pixels
    - threshold : not used by fallback, kept for API compatibility
    Returns: astropy.table.Table with columns 'sharpness', 'roundness1', 'roundness2'
    """
    # Try to use photutils' DAOFindProperties if available
    try:
        from photutils.detection.daofinder import DAOFindProperties  # may fail on newer photutils
        use_upstream = True
    except Exception:
        DAOFindProperties = None
        use_upstream = False

    

    # integer center for patch
    x0 = int(round(x))
    y0 = int(round(y))

    half = max(6, int(2 * fwhm))   # small neighborhood (>=6)
    x1 = x0 - half
    x2 = x0 + half + 1
    y1 = y0 - half
    y2 = y0 + half + 1

    # bounds test
    if x1 < 0 or y1 < 0 or x2 > data.shape[1] or y2 > data.shape[0]:
        
        return np.nan, np.nan, np.nan

    sub = np.array(data[y1:y2, x1:x2], dtype=float)

    # Try upstream implementation if available
    if use_upstream:
        try:
            p = DAOFindProperties(sub, fwhm=fwhm)
            sharpness = getattr(p, "sharpness", np.nan)
            roundness1 = getattr(p, "roundness1", np.nan)
            roundness2 = getattr(p, "roundness2", np.nan)
            return np.nan, np.nan, np.nan
        except Exception:
            # fall back to local method
            pass

    # Local fallback: background estimate, peak value, and second moments
    ny, nx = sub.shape
    cy = ny // 2
    cx = nx // 2
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - cx, yy - cy)

    inner = (r > 1.0) & (r <= max(3.0, fwhm))
    if np.any(inner):
        local_bg = np.nanmedian(sub[inner])
    else:
        flat = sub.flatten()
        if flat.size > 1:
            flat = np.delete(flat, cy * nx + cx)
            local_bg = np.nanmedian(flat)
        else:
            local_bg = 0.0

    center_val = sub[cy, cx]
    denom = max(abs(local_bg), 1e-6)
    sharpness = (center_val - local_bg) / denom

    # subtract background and clamp
    img = sub - local_bg
    img[~np.isfinite(img)] = 0.0
    img[img < 0] = 0.0

    s = img.sum()
    if s <= 0 or not np.isfinite(s):
        return sharpness, np.nan, np.nan

    x_coords = xx.astype(float)
    y_coords = yy.astype(float)
    xc = (img * x_coords).sum() / s
    yc = (img * y_coords).sum() / s

    mu_xx = (img * (x_coords - xc) ** 2).sum() / s
    mu_yy = (img * (y_coords - yc) ** 2).sum() / s
    mu_xy = (img * (x_coords - xc) * (y_coords - yc)).sum() / s

    denom2 = (mu_xx + mu_yy)
    if denom2 == 0 or not np.isfinite(denom2):
        roundness1 = np.nan
        roundness2 = np.nan
    else:
        roundness1 = (mu_xx - mu_yy) / denom2
        roundness2 = 2.0 * mu_xy / denom2

        

    return sharpness, roundness1, roundness2


class WrappedPSFModel(crowdsource.psf.SimplePSF):
    """
    wrapper for photutils GriddedPSFModel
    """
    def __init__(self, psfgridmodel, stampsz=19):
        self.psfgridmodel = psfgridmodel
        self.default_stampsz = stampsz

    def __call__(self, col, row, stampsz=None, deriv=False):

        if stampsz is None:
            stampsz = self.default_stampsz

        parshape = numpy.broadcast(col, row).shape
        tparshape = parshape if len(parshape) > 0 else (1,)

        # numpy uses row, column notation
        rows, cols = np.indices((stampsz, stampsz)) - (np.array([stampsz, stampsz])-1)[:, None, None] / 2.

        # explicitly broadcast
        col = np.atleast_1d(col)
        row = np.atleast_1d(row)
        #rows = rows[:, :, None] + row[None, None, :]
        #cols = cols[:, :, None] + col[None, None, :]

        # photutils seems to use column, row notation
        # only works with photutils <= 1.6.0 - but is wrong there
        #stamps = self.psfgridmodel.evaluate(cols, rows, 1, col, row)
        # it returns something in (nstamps, row, col) shape
        # pretty sure that ought to be (col, row, nstamps) for crowdsource

        # andrew saydjari's version here:
        # it returns something in (nstamps, row, col) shape
        stamps = []
        for i in range(len(col)):
            # the +0.5 is required to actually center the PSF (empirically)
            #stamps.append(self.psfgridmodel.evaluate(cols+col[i]+0.5, rows+row[i]+0.5, 1, col[i], row[i]))
            # the above may have been true when we were using (incorrectly) offset PSFs
            stamps.append(self.psfgridmodel.evaluate(cols+col[i], rows+row[i], 1, col[i], row[i]))

        stamps = np.array(stamps)

        # for oversampled stamps, they may not be normalized
        stamps /= stamps.sum(axis=(1,2))[:,None,None]
        # this is evidently an incorrect transpose
        #stamps = np.transpose(stamps, axes=(0,2,1))

        if deriv:
            dpsfdrow, dpsfdcol = np.gradient(stamps, axis=(1, 2))

        ret = stamps
        if parshape != tparshape:
            ret = ret.reshape(stampsz, stampsz)
            if deriv:
                dpsfdrow = dpsfdrow.reshape(stampsz, stampsz)
                dpsfdcol = dpsfdcol.reshape(stampsz, stampsz)
        if deriv:
            ret = (ret, dpsfdcol, dpsfdrow)

        return ret

    def render_model(self, col, row, stampsz=None):
        """
        this function likely does nothing?
        """
        if stampsz is not None:
            self.stampsz = stampsz

        rows, cols = np.indices(self.stampsz, dtype=float) - (np.array(self.stampsz)-1)[:, None, None] / 2.

        return self.psfgridmodel.evaluate(cols, rows, 1, col, row).T.squeeze()
def get_psf(header, path_prefix='.'):
    if header['INSTRUME'].lower() == 'nircam':
        psfgen = stpsf.NIRCam()
        fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='NIRCam')
    elif header['INSTRUME'].lower() == 'miri':
        psfgen = stpsf.MIRI()
        fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='MIRI')
    instrument = header['INSTRUME']
    filtername = get_filtername(header)
    module = header['MODULE']
    detector = header['DETECTOR']
    print('module',module)
    print('detector', detector)
    ww = wcs.WCS(header)
    try:
        assert ww.wcs.cdelt[1] != 1, "This is not a valid WCS!!! CDELT is wrong!! how did this HAPPEN!?!?"
    except AssertionError as ex:
        print(ex)
        print("ignoring WCS failure so check that stuff is right...")

    psfgen.filter = filtername
    obsdate = header['DATE-OBS']

    with open(os.path.expanduser('~/.mast_api_token'), 'r') as fh:
        api_token = fh.read().strip()

    npsf = 16
    oversample = 2
    fov_pixels = 512
    # ./nircam_nrca1_f140m_fovp512_samp2_npsf16.fits
    #psf_fn = f'{path_prefix}/{detector.lower()}_{filtername.lower()}_samp{oversample}_nspsf{npsf}_npix{fov_pixels}_{detector}.fits'
    wav = int(filtername[1:4])  # e.g., F140M -> 140
    print('wav', wav)
    # now the PSF should be written
    
    psf_fn = f'{path_prefix}/nircam_{detector.lower()}_{filtername.lower()}_fovp{fov_pixels}_samp{oversample}_npsf{npsf}.fits'
    if wav>=300:
        detector = detector[:4] + '5'  # e.g., NRCB4 -> NRCB5
        psf_fn = f'{path_prefix}/nircam_{detector.lower()}_{filtername.lower()}_fovp{fov_pixels}_samp{oversample}_npsf{npsf}.fits'
        print('replaced psf_fn for longwave', psf_fn)
    if module == 'merged':
        project_id = header['PROGRAM'][1:5]
        obs_id = header['OBSERVTN'].strip()
        merged_psf_fn = f'{basepath}/psfs/{filtername.upper()}_{project_id}_{obs_id}_merged_PSFgrid.fits'
        if os.path.exists(psf_fn):
            psf_fn = merged_psf_fn
        else:
            print("stpsf is being used for merged data because merged PSF does not exist", flush=True)

    if os.path.exists(str(psf_fn)):
        # As a file
        log.info(f"Loading grid from psf_fn={psf_fn}")
        big_grid = to_griddedpsfmodel(psf_fn)  # file created 2 cells above
        if isinstance(big_grid, list):
            print(f"PSF IS A LIST OF GRIDS!!!", flush=True)
            big_grid = big_grid[0]
    else:
        log.info(f'PSF file {psf_fn} does not exist; downloading from MAST')
        from astroquery.mast import Mast

        print(f"Attempting to load PSF for {obsdate}")
        try:
            Mast.login(api_token.strip())
            os.environ['MAST_API_TOKEN'] = api_token.strip()

            psfgen.load_wss_opd_by_date(f'{obsdate}T00:00:00')
        except (urllib3.exceptions.ReadTimeoutError, requests.exceptions.ReadTimeout, requests.HTTPError) as ex:
            print(f"Failed to build PSF: {ex}")
        except Exception as ex:
            print("psfgen load_wss_opd_by_date failed")
            print(ex)

        log.info(f"starfinding: Calculating grid for psf_fn={psf_fn}")
        # https://github.com/spacetelescope/stpsf/blob/cc16c909b55b2a26e80b074b9ab79ed9a312f14c/stpsf/stpsf_core.py#L640
        # https://github.com/spacetelescope/stpsf/blob/cc16c909b55b2a26e80b074b9ab79ed9a312f14c/stpsf/gridded_library.py#L424
        big_grid = psfgen.psf_grid(num_psfs=npsf, oversample=oversample,
                                   all_detectors=True, fov_pixels=fov_pixels,
                                   outdir=path_prefix, 
                                   save=True, overwrite=True)

        print(glob.glob(psf_fn.replace(".fits", "*")))
        assert glob.glob(psf_fn.replace(".fits", "*"))
        if isinstance(big_grid, list):
            print(f"PSF FROM PSF_GEN IS A LIST OF GRIDS!!!", flush=True)
            big_grid = big_grid[0]
            # if we really want to get this right, we need to create a new grid of PSF models
            # that is some sort of average of the PSF model grid.
            # There's no way to do it _right_ right without going back to the original data,
            # which is untenable with this approach.  It's a huge project.

    return big_grid
def update_sat_catalogs(filt='F140M'):
    for module in ('nrca', 'nrcb'):
        wav = int(filt[1:4])
        if wav < 300:
            filelist = glob.glob(f"/orange/adamginsburg/jwst/w51/{filt}/pipeline/*{module}*destreak*crf.fits")
        else:
            filelist = glob.glob(f"/orange/adamginsburg/jwst/w51/{filt}/pipeline/*{module}*crf.fits")
        print('filelist', filelist)
        for i, fn in enumerate(filelist):
            print(i, fn)
            if True:
                print(fn)
                sat_catalog_file = fn.replace(".fits", '_satstar_catalog.fits')
                print(sat_catalog_file)
                sat_hdul = fits.open(sat_catalog_file)
                

                # column names: use the FITS API that exists on BinTableHDU
                try:
                    colnames = sat_hdul[1].columns.names
                except Exception:

                    # fallback to numpy dtype names if columns isn't available
                    try:
                        colnames = list(sat_hdul[1].data.dtype.names)
                    except Exception:
                        colnames = None
                print('colname from original sat catalog', colnames )
                tab = sat_hdul[1].data
                xpos = tab['x_fit']
                ypos = tab['y_fit']
             
                img = fits.getdata(fn)
                fitsdat = fits.open(fn)

                #img[np.isnan(fits.getdata(fn)['VAR_POISSON'].data)] = 0

                wcs_img = WCS(fits.getheader(fn, ext=('SCI', 1)))

                """
                fig = plt.figure(figsize=(20,20))
                ax = fig.add_subplot(111, projection=wcs)
                norm = simple_norm(img, 'sqrt', percent=99.)
                ax.imshow(img, norm=norm, origin='lower', cmap='Greys_r')   
                ax.set_title(f'Saturated Stars in {filt} - {module.upper()}')
                ax.scatter(xpos, ypos, s=50, edgecolor='red', facecolor='none', marker='o', label='Saturated Stars')
                ax.set_xlim(0, img.shape[1])
                ax.set_ylim(0, img.shape[0])
                plt.show()
                """

                #nrc = webbpsf.NIRCam()
                #nrc.filter = filt
                #grid = nrc.psf_grid(num_psfs=16, all_detectors=False, verbose=True, save=True)
                header = fits.getheader(fn)
                path_prefix = '/orange/adamginsburg/jwst/w51/psfs'
                big_grid = get_psf(header, path_prefix='.')
                fwhm, fwhm_pix = get_fwhm(header, instrument_replacement='NIRCam')

                def recentering_and_get_flux(x,y,flux, savefigdir='./'):
                    if np.isfinite(flux) == False or flux <=0:
                        size= 101
                    else:
                        size = int(int(np.log10(flux)*5)*2+1)
                    print(x,y, img.shape, size)
                    try:
                        cutout = Cutout2D(img, (x, y), (size, size), wcs=wcs_img)
                    except:
                        print("NoOverlapError for cutout at x,y=", x, y, size, img.shape)
                        return None, None
                    # the recentered positions should be the brightest peak of the center 10x10 pixels
                    try:
                        recentered_x = size//2 + np.unravel_index(np.nanargmax(cutout.data[size//2-5:size//2+5, size//2-5:size//2+5]), (10,10))[1] - 5 + 0.5
                    except ValueError:
                        recentered_x = size//2
                    try:
                        recentered_y = size//2 + np.unravel_index(np.nanargmax(cutout.data[size//2-5:size//2+5, size//2-5:size//2+5]), (10,10))[0] - 5 + 0.5
                    except ValueError:
                        recentered_y = size//2
                    init_params = QTable()
                    init_params['x'] = [recentered_x]
                    init_params['y'] = [recentered_y]
                    lmfitter = LevMarLSQFitter()
                   
                    cutout_data =cutout.data
                    # replace nan with 0
                    cutout_data[np.isnan(cutout_data)] = 0.0
                    # if isinstance(grid, list):
                    #     print(f"Grid is a list: {grid}")
                    #     psf_model = WrappedPSFModel(grid[0])
                    #     dao_psf_model = grid[0]
                    # else:

                    #psf_model = WrappedPSFModel(grid, stampsz=(size,size))
                    psfphot = PSFPhotometry(
                                              localbkg_estimator=None,
                                              fitter=lmfitter,
                                              psf_model=big_grid,
                                              fit_shape=size,
                                              aperture_radius=15*fwhm_pix)
                    try:
                        result_tab = psfphot(cutout_data, init_params=init_params)
                    except Exception as ex:
                        return None, None

                    ny = cutout_data.shape[0]
                    nx = cutout_data.shape[1]
                    model_image = np.zeros_like(cutout_data)

                    for x0, y0, flux in zip(result_tab['x_fit'], result_tab['y_fit'], result_tab['flux_fit']):
                        # Make a local grid around the source
                        if np.isnan(flux):
                            raise ValueError("Flux is NaN; cannot build PSF model image")
                        y, x = np.mgrid[0:ny, 0:nx]
                        #psf_eval = big_grid(x, y, flux=flux, x_0=x0, y_0=y0)  # works for analytic PSF
                        psf_eval = big_grid(x-x0, y-y0) * flux  # works for GriddedPSFModel
                        # cut psf_eval to the image size
                        model_image += psf_eval[0:ny, 0:nx]
                    

                    """
                    print("DEBUG: flux (raw) =", flux)
                    print("DEBUG: size (computed) =", size)
                    print("DEBUG: np.isfinite(flux) =", np.isfinite(flux))
                    print("DEBUG: flux > 0:", flux > 0)

                    print("DEBUG: big_grid type:", type(big_grid))
                    # Try to show some useful attributes if present
                    for attr in ('psf_shape','psf_size','stamps_shape','stampsz','shape','data_shape'):
                        if hasattr(big_grid, attr):
                            print(f"DEBUG: big_grid.{attr} =", getattr(big_grid, attr))
                    # If it's a list, show first element type and attributes
                    if isinstance(big_grid, (list, tuple)):
                        print("DEBUG: big_grid is a list/tuple; len =", len(big_grid))
                        print("DEBUG: first element type:", type(big_grid[0]))
                        for attr in ('psf_shape','psf_size','stamps_shape','shape'):
                            if hasattr(big_grid[0], attr):
                                print(f"DEBUG: big_grid[0].{attr} =", getattr(big_grid[0], attr))
                    if isinstance(big_grid, (list, tuple)):
                        if len(big_grid) == 0:
                            raise RuntimeError("big_grid is empty; cannot build PSF model image")
                        used_psf = big_grid[0]
                    else:
                        used_psf = big_grid

                    # Ensure integer size and build a 2-tuple shape
                    size = int(size)
                    if size <= 0:
                        raise ValueError(f"Computed size is not positive: {size}")
                    model_shape = (size, size)
                    print("DEBUG: using model_shape =", model_shape, "and used_psf type =", type(used_psf))

                    # Try make_psf_model_image with an explicit border_size of 0
                    try:
                        model_data, params = make_psf_model_image(model_shape, used_psf, n_sources=1,
                                                                model_shape=model_shape, border_size=0)
                    except ValueError as ex:
                        print("make_psf_model_image ValueError:", ex)
                        # Fallback: try a larger stamp (guess from used_psf where possible)
                        psf_stamp_guess = None
                        for attr in ('psf_shape', 'stamps_shape', 'shape', 'psf_size', 'stampsz'):
                            psf_stamp_guess = getattr(used_psf, attr, None)
                            if psf_stamp_guess is not None:
                                break
                        if psf_stamp_guess is None and hasattr(used_psf, 'data'):
                            try:
                                psf_stamp_guess = used_psf.data.shape
                            except Exception:
                                psf_stamp_guess = None
                        if isinstance(psf_stamp_guess, tuple):
                            stamp_size = max(psf_stamp_guess)
                        elif isinstance(psf_stamp_guess, (int, float)):
                            stamp_size = int(psf_stamp_guess)
                        else:
                            stamp_size = 25
                        alt_size = max(size, 2 * int(stamp_size) + 1, 51)
                        if alt_size % 2 == 0:
                            alt_size += 1
                        alt_model_shape = (alt_size, alt_size)
                        print(f"Retrying with alt_model_shape={alt_model_shape} (psf_stamp_guess={psf_stamp_guess})")
                        model_data, params = make_psf_model_image(alt_model_shape, used_psf, n_sources=1,
                                                                model_shape=alt_model_shape, border_size=0)
                        """
                        

                   
                    """
                    fig = plt.figure(figsize=(21,7))
                    ax1 = fig.add_subplot(131)
                    norm = simple_norm(cutout.data, 'sqrt', percent=99.)
                    ax1.imshow(cutout.data, origin='lower', cmap='Greys_r', norm = norm)
                    ax1.scatter(cutout.data.shape[1]/2, cutout.data.shape[0]/2, s=100, color='red', marker='x')
                    ax1.scatter(result_tab['x_fit'], result_tab['y_fit'], s=100, color='blue', marker='x')
                    ax1.set_title('Cutout Data')
                    ax2 = fig.add_subplot(132)
                    #norm = simple_norm(model_image, 'sqrt', percent=99.)
                    ax2.imshow(model_image, origin='lower', cmap='Greys_r', norm = norm)
                    ax2.set_title('PSF Model Fit')
                    ax3 = fig.add_subplot(133)
                    norm = simple_norm(cutout.data - model_image, 'sqrt', percent=99.)
                    ax3.imshow(cutout.data - model_image, origin='lower', cmap='Greys_r', norm = norm)
                    ax3.set_title('Residual (Data - Model)')
                    plt.savefig(savefigdir)
                    """


                    return result_tab, cutout
                    #plt.show()



                savefigdir = f'/orange/adamginsburg/jwst/w51/{filt}/sat_fit_figs/'
                if os.path.exists(savefigdir) == False:
                    os.makedirs(savefigdir)
                tab_copied = Table()
                jj=0
                for j in range(len(tab)):
                    if xpos[j] < 0:
                        continue
                    if ypos[j] < 0:
                        continue

                    result_tab, cutout = recentering_and_get_flux(xpos[j], ypos[j], tab['flux_fit'][j], savefigdir=savefigdir+'_%03d.png'%j)
                    # update the original table with new fluxes
                    if result_tab is None:
                        continue

                    sharpness, roundness1, roundness2 = daofind_metrics_at_positions(cutout.data, xpos[j], ypos[j], fwhm=fwhm_pix, threshold=3*np.nanmedian(fitsdat['ERR'].data))

                   
                    result_tab['sharpness'] = [sharpness]
                    result_tab['roundness1'] = [roundness1]
                    result_tab['roundness2'] = [roundness2]
                    row0 = result_tab[0]
                    
                    # if j==0 or basetab is not defined
                    if j ==0 :
                        basetab = row0
                        jj+=1
                    else:
                        if jj==0:
                            basetab = row0
                        else:
                            basetab = vstack([basetab, row0])
                print('base tab length', len(basetab))
                pixcoord = PixCoord(basetab['x_fit'], basetab['y_fit'])
                skycoord = pixcoord.to_sky(wcs_img)

                # Ensure RA/Dec are 1-D numeric arrays (avoid numpy scalars)
                # convert to degrees and force at least 1-dimension so astropy.Column
                # doesn't receive a numpy scalar which triggers the AttributeError.
                try:
                    ra_vals = np.atleast_1d(skycoord.ra.to(u.deg).value)
                    dec_vals = np.atleast_1d(skycoord.dec.to(u.deg).value)
                except Exception:
                    # Fallback: try to coerce attributes to arrays explicitly
                    ra_comp = getattr(skycoord, 'ra', skycoord)
                    dec_comp = getattr(skycoord, 'dec', skycoord)
                    try:
                        ra_vals = np.atleast_1d(ra_comp.to(u.deg).value)
                        dec_vals = np.atleast_1d(dec_comp.to(u.deg).value)
                    except Exception:
                        ra_vals = np.atleast_1d(ra_comp)
                        dec_vals = np.atleast_1d(dec_comp)

                racolumn = Column(name='skycoord_fit.ra', data=ra_vals, unit=u.deg)
                deccolumn = Column(name='skycoord_fit.dec', data=dec_vals, unit=u.deg)

                # If basetab is a single-row object (Row/FITS_record) convert to Table
                # so add_column is available and shapes align.
                if not hasattr(basetab, 'add_column'):
                    try:
                        basetab = Table(basetab)
                    except Exception:
                        # Last-resort: build a minimal Table from dict-like row
                        basetab = Table([basetab])

                basetab.add_column(racolumn)
                basetab.add_column(deccolumn)
                print('colname from basetab catalog', basetab.colnames)

                for col in basetab.columns:
                    colname = col.name if hasattr(col, 'name') else str(col)
                    if colname in basetab.colnames:
                        tab_copied[colname] = basetab[colname]
                    

                
                
                # raise error when the column names of basetab and tab_copied do not match
                if set(basetab.colnames) != set(tab_copied.colnames):
                    print("basetab columns:", basetab.colnames)
                    print("tab_copied columns:", tab_copied.colnames)
                    raise ValueError("Column names of basetab and tab_copied do not match")
              
              
                tab_copied.write(sat_catalog_file.replace('.fits', '_satstars_catalog_recentered.fits'), overwrite=True)
                assert os.path.exists(sat_catalog_file.replace('.fits', '_satstars_catalog_recentered.fits'))

def main():

    #with open(os.path.expanduser('/home/adamginsburg/.mast_api_token'), 'r') as fh:
    #    api_token = fh.read().strip()
    #from astroquery.mast import Mast
    #Mast.login(api_token.strip())
    #os.environ['MAST_API_TOKEN'] = api_token.strip()

    #"fn = '/orange/adamginsburg/jwst/w51/F405N/pipeline/jw02221002001_02201_00001_nrcalong_destreak_o002_crf.fits'
    #remove_saturated_stars(fn, verbose=True)
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--filter", dest="filter",
                      default='F140M',
                      help="filter list", metavar="filter")
    (options, args) = parser.parse_args()
    filt = options.filter
    update_sat_catalogs(filt=filt)
if __name__ == "__main__":
    main()
