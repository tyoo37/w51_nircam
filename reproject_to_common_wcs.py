from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_interp
import numpy as np
import math
import os

def _read_image_and_wcs(fn):
    data, header = fits.getdata(fn, header=True)
    # handle 3D cubes where image is in the last two axes; adapt as needed
    if data.ndim > 2:
        # assume (stokes, y, x) or (1, y, x)
        data2d = data.reshape(-1, data.shape[-2], data.shape[-1])[-1]
    else:
        data2d = data
    w = WCS(header, naxis=2)
    ny, nx = data2d.shape
    return data2d, w, (ny, nx), header

def _corners_world(wcs_obj, nx, ny):
    # corners in pixel coordinates (x, y)
    corners_pix = np.array([[0,0], [nx,0], [0,ny], [nx,ny]], dtype=float)
    x = corners_pix[:,0]
    y = corners_pix[:,1]
    # wcs.wcs_pix2world expects arrays (x, y, origin)
    ra, dec = wcs_obj.wcs_pix2world(x, y, 0)
    return ra, dec

def reproject_images_to_common_wcs(filenames, out_prefix='reproj_', pixel_scale_arcsec=None, reference_index=None, buffer_pix=5):
    """
    Reproject `filenames` to a single output WCS that covers all input footprints.

    Parameters
    - filenames: list of input FITS file paths
    - out_prefix: prefix for output files written alongside inputs
    - pixel_scale_arcsec: if provided, force output pixel scale (arcsec/pix). Otherwise the smallest input pixel scale is used.
    - reference_index: index in filenames to use as orientation/reference. If None, choose image with smallest pixel scale.
    - buffer_pix: additional padding (pixels) around computed bounding box to avoid edge truncation.
    """
    # read all images and WCS
    imgs = []
    for fn in filenames:
        data2d, w, (ny, nx), hdr = _read_image_and_wcs(fn)
        imgs.append({'fn':fn, 'data':data2d, 'wcs':w, 'nx':nx, 'ny':ny, 'hdr':hdr})

    # compute pixel scales (deg/pix) for each image, then convert to arcsec/pix
    pix_scales_deg = [proj_plane_pixel_scales(it['wcs']) for it in imgs]  # (dx, dy) in deg
    # use the mean of the two axes
    pix_scales_arcsec = [1000.0 * np.mean(ps) * 3600.0 / 1000.0 if False else (np.mean(ps)*3600.0) for ps in pix_scales_deg]
    # simpler: proj_plane_pixel_scales returns degrees -> *3600 to arcsec
    pix_scales_arcsec = [np.mean(ps)*3600.0 for ps in pix_scales_deg]

    # choose reference
    if reference_index is None:
        reference_index = int(np.argmin(pix_scales_arcsec))  # highest resolution
    ref = imgs[reference_index]
    ref_wcs = ref['wcs']
    # choose output pixel scale
    if pixel_scale_arcsec is None:
        out_pix_scale_arcsec = min(pix_scales_arcsec)
    else:
        out_pix_scale_arcsec = pixel_scale_arcsec

    # If the chosen out_pix_scale differs from ref_wcs, we will keep ref orientation but update CD/CRPIX
    # compute all corner world coordinates from every image
    all_ra = []
    all_dec = []
    for it in imgs:
        ra, dec = _corners_world(it['wcs'], it['nx'], it['ny'])
        all_ra.extend(ra.tolist())
        all_dec.extend(dec.tolist())
    all_ra = np.array(all_ra)
    all_dec = np.array(all_dec)

    # map these world coords into pixel coords of the reference WCS
    x_ref, y_ref = ref_wcs.wcs_world2pix(all_ra, all_dec, 0)  # arrays
    # compute bounding box in ref_pixel coordinates
    min_x = np.floor(np.nanmin(x_ref)) - buffer_pix
    max_x = np.ceil(np.nanmax(x_ref)) + buffer_pix
    min_y = np.floor(np.nanmin(y_ref)) - buffer_pix
    max_y = np.ceil(np.nanmax(y_ref)) + buffer_pix

    # width/height in pixels (int)
    width = int(max_x - min_x + 1)
    height = int(max_y - min_y + 1)

    # We may need to resample the reference WCS to the chosen pixel scale.
    # Build output WCS from ref_wcs but adjust CD/PC to match out_pix_scale
    out_wcs = ref_wcs.deepcopy()

    # Compute scaling factor between current ref_wcs pixel scale and desired output pixel scale
    ref_pix_scale = np.mean(proj_plane_pixel_scales(ref_wcs)) * 3600.0  # arcsec/pixel
    scale_factor = out_pix_scale_arcsec / ref_pix_scale

    # If scale_factor != 1.0, we adjust the CD matrix proportionally (preserving rotation)
    # CD matrix = ref_wcs.wcs.cd (if available) else construct from CDELT and PC
    try:
        cd = out_wcs.wcs.cd
        if cd is None or not np.any(cd):
            # fall back to CDELT + PC
            cdelt = out_wcs.wcs.cdelt.copy()
            pc = out_wcs.wcs.get_pc()
            cd = np.dot(np.diag(cdelt), pc)
    except Exception:
        cd = out_wcs.wcs.cd
    # Scale cd by factor (so pixel becomes larger if scale_factor>1)
    if cd is not None and np.any(cd):
        out_wcs.wcs.cd = cd * scale_factor
    else:
        # fallback: scale cdelt
        out_wcs.wcs.cdelt = out_wcs.wcs.cdelt * scale_factor

    # Now adjust CRPIX so that the world coordinate that used to map to ref pixel (min_x, min_y)
    # now maps to pixel (0,0) in the output image. The old mapping is:
    # world_at_min = ref_wcs.wcs_pix2world(min_x, min_y, 0)
    # But easier: shift CRPIX by min_x/min_y offset.
    old_crpix = out_wcs.wcs.crpix.copy()
    # crpix is 1-based in FITS WCS; wcs_world2pix/wcs_pix2world use 0-based if origin=0
    # The reference pixel positions (x_ref,y_ref) were computed using origin=0, so treat min_x as 0-based.
    out_wcs.wcs.crpix[0] = old_crpix[0] - min_x
    out_wcs.wcs.crpix[1] = old_crpix[1] - min_y

    shape_out = (height, width)  # (ny, nx) expected by reproject

    print(f"Chosen reference image: {ref['fn']}")
    print(f"Output shape: {shape_out}, output pixel scale (arcsec/pix) = {out_pix_scale_arcsec:.4f}")
    print(f"Writing reprojected files with prefix '{out_prefix}' next to inputs.")

    # Reproject each image
    out_files = []
    for it in imgs:
        data = it['data']
        w = it['wcs']
        print(f"Reprojecting {it['fn']}")
        arr_reproj, footprint = reproject_interp((data, w), out_wcs, shape_out=shape_out)
        # optional: preserve header from ref but update with out_wcs header
        out_header = out_wcs.to_header()
        # if original header has units or BUNIT, copy it
        try:
            orig_hdr = it['hdr']
            if 'BUNIT' in orig_hdr:
                out_header['BUNIT'] = orig_hdr['BUNIT']
        except Exception:
            pass
        out_fn = os.path.join(os.path.dirname(it['fn']), out_prefix + os.path.basename(it['fn']))
        fits.writeto(out_fn, arr_reproj.astype(np.float32), header=out_header, overwrite=True)
        out_files.append(out_fn)

    return out_wcs, shape_out, out_files


    