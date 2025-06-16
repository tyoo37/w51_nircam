import os
import json
from glob import glob
import shutil
from astropy.io import fits
from jwst import datamodels
import asdf
import crds
from astropy.table import Table
import numpy as np
import matplotlib as mpl
from astroquery.mast import Mast, Observations

# This script is used to download and process JWST data for the W51 region.


basepath = '/orange/adamginsburg/jwst/w51/'

os.environ["CRDS_PATH"] = f"{basepath}crds/"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
from jwst.pipeline import calwebb_image3

def merge(regionname='w51',filtername='F140M', field='001', proposal_id='6151',destreak_suffix='_destreak_1m1overf'):
    output_dir = f'/orange/adamginsburg/jwst/{regionname}/{filtername}/pipeline/'


    module='merged'   

    print(basepath)
    fwhm_tbl = Table.read(f'{basepath}/reduction/fwhm_table.ecsv')
    row = fwhm_tbl[fwhm_tbl['Filter'] == filtername]
    fwhm = fwhm_arcsec = float(row['PSF FWHM (arcsec)'][0])
    fwhm_pix = float(row['PSF FWHM (pixel)'][0])

    #destreak_suffix = '' if do_destreak else '_nodestreak'

    # sanity check
    if regionname == 'sgrb2':
        if proposal_id == '5365':
            assert field == '001'
    if regionname == 'w51':
        if proposal_id == '6151':
            assert field == '001'

    os.environ["CRDS_PATH"] = f"{basepath}crds/"
    os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
    mpl.rcParams['savefig.dpi'] = 80
    mpl.rcParams['figure.dpi'] = 80

    # Files created in this notebook will be saved
    # in a subdirectory of the base directory called `Stage3`
    output_dir = f'/orange/adamginsburg/jwst/{regionname}/{filtername}/pipeline/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    # the files are one directory up
    for fn in glob("../*cal.fits"):
        try:
            os.link(fn, './'+os.path.basename(fn))
        except Exception as ex:
            print(f'Failed to link {fn} to {os.path.basename(fn)} because of {ex}')

    Observations.cache_location = output_dir
    obs_table = Observations.query_criteria(
                                            proposal_id=proposal_id,
                                            #proposal_pi="Ginsburg*",
                                           
                                            )
    
   
    if 'filters' in obs_table.colnames:
        filters = obs_table['filters'].filled('')
        obs_id = obs_table['obs_id'].filled('')  # Replace masked values with an empty string
        msk = ((np.char.find(filters, filtername.upper()) >= 0) |
           (np.char.find(obs_id, filtername.lower()) >= 0))
    else:
        print("Warning: 'filters' column not found in obs_table")
  # msk = np.char.find(obs_table['obs_id'], filtername.lower()) >= 0

    data_products_by_obs = Observations.get_product_list(obs_table[msk])
    print("data prodcts by obs length: ", len(data_products_by_obs))

    products_asn = Observations.filter_products(data_products_by_obs, extension="json")
    print("products_asn length:", len(products_asn))
    #valid_obsids = products_asn['obs_id'][np.char.find(np.unique(products_asn['obs_id']), 'jw02221-o001', ) == 0]
    #match = [x for x in valid_obsids if filtername.lower() in x][0]

    asn_mast_data = products_asn#[products_asn['obs_id'] == match]
    print("asn_mast_data:", asn_mast_data)

    manifest = Observations.download_products(asn_mast_data, download_dir=output_dir)
    print("manifest:", manifest)

    # MAST creates deep directory structures we don't want
    for row in manifest:
        try:
            shutil.move(row['Local Path'], os.path.join(output_dir, os.path.basename(row['Local Path'])))
        except Exception as ex:
            print(f"Failed to move file with error {ex}")

    products_fits = Observations.filter_products(data_products_by_obs, extension="fits")
    print("products_fits length:", len(products_fits))
    uncal_mask = np.array([uri.endswith('_uncal.fits') and f'jw0{proposal_id}{field}' in uri for uri in products_fits['dataURI']])
    uncal_mask &= products_fits['productType'] == 'SCIENCE'
    print("uncal length:", (uncal_mask.sum()))

    already_downloaded = np.array([os.path.exists(os.path.basename(uri)) for uri in products_fits['dataURI']])
    uncal_mask &= ~already_downloaded
    print(f"uncal to download: {uncal_mask.sum()}; {already_downloaded.sum()} were already downloaded")

    if uncal_mask.any():
        manifest = Observations.download_products(products_fits[uncal_mask], download_dir=output_dir)
        print("manifest:", manifest)

        # MAST creates deep directory structures we don't want
        for row in manifest:
            try:
                shutil.move(row['Local Path'], os.path.join(output_dir, os.path.basename(row['Local Path'])))
            except Exception as ex:
                print(f"Failed to move file with error {ex}")

    print(f"Searching for {os.path.join(output_dir, f'jw0{proposal_id}-o{field}*_image3_*0[0-9][0-9]_asn.json')}")

    asn_file_search = glob(os.path.join(output_dir, f'jw0{proposal_id}-o{field}*_image3_*0[0-9][0-9]_asn.json'))
    if len(asn_file_search) == 1:
        asn_file = asn_file_search[0]
    elif len(asn_file_search) > 1:
        asn_file = sorted(asn_file_search)[-1]
        print(f"Found multiple asn files: {asn_file_search}.  Using the more recent one, {asn_file}.")
    else:
        raise ValueError(f"Mismatch: Did not find any asn files for module {module} for field {field} in {output_dir}")

   

    mapping = crds.rmap.load_mapping(f'/orange/adamginsburg/jwst/{regionname}/crds/mappings/jwst/jwst_nircam_pars-tweakregstep_0003.rmap')
    print(f"Mapping: {mapping.todict()['selections']}")
    print(f"Filtername: {filtername}")
    filter_match = [x for x in mapping.todict()['selections'] if filtername in x]
    print(f"Filter_match: {filter_match} n={len(filter_match)}")
    tweakreg_asdf_filename = filter_match[0][4]
    tweakreg_asdf = asdf.open(f'https://jwst-crds.stsci.edu/unchecked_get/references/jwst/{tweakreg_asdf_filename}')
    tweakreg_parameters = tweakreg_asdf.tree['parameters']
    tweakreg_parameters.update({'skip': True,
                                'fitgeometry': 'general',
                                # brightest = 5000 was causing problems- maybe the cross-alignment was getting caught on PSF artifacts?
                                'brightest': 5000,
                                'snr_threshold': 20, # was 5, but that produced too many stars
                                # define later 'abs_refcat': abs_refcat,
                                'save_catalogs': True,
                                'catalog_format': 'fits',
                                'kernel_fwhm': fwhm_pix,
                                'nclip': 5,
                                'starfinder': 'dao',
                                # expand_refcat: A boolean indicating whether or not to expand reference catalog with new sources from other input images that have been already aligned to the reference image. (Default=False)
                                'expand_refcat': True,
                                # based on DebugReproduceTweakregStep
                                'sharplo': 0.3,
                                'sharphi': 0.9,
                                'roundlo': -0.25,
                                'roundhi': 0.25,
                                'separation': 0.5, # minimum separation; default is 1
                                'tolerance': 0.1, # tolerance: Matching tolerance for xyxymatch in arcsec. (Default=0.7)
                                'save_results': True,
                                # 'clip_accum': True, # https://github.com/spacetelescope/tweakwcs/pull/169/files
                                })

    
    # Load asn_data for both modules
    with open(asn_file) as f_obj:
        asn_data = json.load(f_obj)
    print(asn_data['products'][0]['members'])
    for member in asn_data['products'][0]['members']:

        fname = member['expname']
    
        member['expname'] = fname.replace("_cal.fits", f"_align_{destreak_suffix}.fits")
        print(fname)
        print(member['expname'])
        shutil.copy(output_dir+fname.replace("_cal.fits", f"_{destreak_suffix}.fits"), output_dir+member['expname'])

    asn_data['products'][0]['name'] = f'jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-merged_{destreak_suffix}'
    asn_file_merged = asn_file.replace("_asn.json", f"_merged_{destreak_suffix}_asn.json")
    with open(asn_file_merged, 'w') as fh:
        json.dump(asn_data, fh)


    calwebb_image3.Image3Pipeline.call(
        asn_file_merged,
        steps={'tweakreg': tweakreg_parameters,},
        #steps={'tweakreg': False,}
        output_dir=output_dir,
        save_results=True)
    print(f"DONE running Image3Pipeline {asn_file_merged}.  This should have produced file {asn_data['products'][0]['name']}_i2d.fits")

    print("After tweakreg step, checking WCS headers:")
    for member in asn_data['products'][0]['members']:
        check_wcs(member['expname'])
    check_wcs(asn_data['products'][0]['name'] + "_i2d.fits")

    print(f"Realigning to VVV (module={module})")# with raoffset={raoffset}, decoffset={decoffset}")
    realigned_vvv_filename = f'{basepath}{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}{destreak_suffix}_realigned-to-vvv.fits'
    print(f"Realigned to VVV filename: {realigned_vvv_filename}")
    shutil.copy(f'{basepath}{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_i2d.fits',
                realigned_vvv_filename)
    # realigned = realign_to_vvv(filtername=filtername.lower(),
    #                            fov_regname=fov_regname[regionname], basepath=basepath, module=module,
    #                            fieldnumber=field, proposal_id=proposal_id,
    #                            imfile=realigned_vvv_filename,
    #                            max_offset=(0.4 if wavelength > 250 else 0.2)*u.arcsec,
    #                            ksmag_limit=15 if filtername.lower() == 'f410m' else 11,
    #                            mag_limit=18 if filtername.lower() == 'f115w' else 15,
    #                            #raoffset=raoffset, decoffset=decoffset
    #                            )

    print(f"Realigning to refcat (module={module})")# with raoffset={raoffset}, decoffset={decoffset}")
    realigned_refcat_filename = f'{basepath}{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}{destreak_suffix}_realigned-to-refcat.fits'
    print(f"Realigned refcat filename: {realigned_refcat_filename}")
    shutil.copy(f'{basepath}{filtername.upper()}/pipeline/jw0{proposal_id}-o{field}_t001_nircam_clear-{filtername.lower()}-{module}_i2d.fits',
                realigned_refcat_filename)


merge(regionname='w51', filtername='F140M', field='001', proposal_id='6151')