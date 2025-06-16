from jwst.datamodels import ImageModel
import os
import image1overf
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.table import Table
os.environ["CRDS_PATH"] = "/orange/adamginsburg/jwst/w51/crds/"
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
from astropy.io import ascii, fits
import numpy as np
import glob
from mpl_toolkits.axes_grid1 import make_axes_locatable
import shutil
basepath = '/orange/adamginsburg/jwst/w51/'

filters = ['F140M', 'F150W','F182M','F187N','F210M','F335M','F360M','F405N','F410M','F480M']
for filt in filters:
    files = glob.glob(f'{basepath}/{filt}/pipeline/*_cal.fits')

    for file in files:
        if os.path.isdir(f'{basepath}/{filt}/no_destreak'):
            os.makedirs(f'{basepath}/{filt}/no_destreak', exist_ok=True)
        shutil.copy(file, f'{basepath}/{filt}/no_destreak/{file.split("/")[-1]}')
        """ 
        string_third =  file.split('/')[-1].split('_')[3]
        print(string_third)
        string_third_char_count = sum([1 for c in string_third if c.isalpha() or c.isdigit()]) 
        print(string_third_char_count)
        file_front = file.split('/')[-1][:26+string_third_char_count]
        """
        #if os.path.exists(f'{basepath}/{filt}/pipeline/{file_front}_destreak_1m1overf.fits'):
        #    print(f'{basepath}/{filt}/pipeline/{file_front}_destreak_1m1overf.fits already exists, skipping')
        #    continue    
        #print(filt, file_front)
        cal = ImageModel(file)

        
        out_im1overf = image1overf.sub1fimaging(fits.open(file), 3.0, 2.0, False, True)
        resid = out_im1overf.data - cal.data[4:2044,4:2044]

        fig, axs = plt.subplots(1,3, figsize=(12, 12))
        minn = 5
        maxx = 90

        axs[0].imshow(np.nan_to_num(cal.data), vmin=np.nanpercentile(cal.data.flatten(), minn), vmax=np.nanpercentile(cal.data.flatten(), maxx))
        axs[1].imshow(np.nan_to_num(out_im1overf.data), vmin=np.nanpercentile(out_im1overf.data.flatten(), minn), vmax=np.nanpercentile(out_im1overf.data.flatten(), maxx))
        c = axs[2].imshow(np.nan_to_num(resid.data), vmin=np.nanpercentile(resid.flatten(), minn), vmax=np.nanpercentile(resid.flatten(), maxx))

        axs[0].set_title('Cal')
        axs[1].set_title('Destreak - image1overf')
        axs[2].set_title('Residual')

        divider = make_axes_locatable(axs[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)

        fig.colorbar(c, cax=cax, orientation='vertical')
        plt.show()
        plt.close()

        with fits.open(file) as hdul:
            hdul_new = fits.HDUList([hdu.copy() for hdu in hdul])
        hdul_new[0].data = out_im1overf.data


        print(f'Saving {file}')
        hdul_new.writeto(file, overwrite=True)

