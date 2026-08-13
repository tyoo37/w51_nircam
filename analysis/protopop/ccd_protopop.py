from protopop.cluster import Cluster
from astropy import units as u
import numpy as np
from optparse import OptionParser

from astropy.io import fits
from astropy.wcs import WCS
from astroquery.svo_fps import SvoFps
from astropy import units as u
import matplotlib.pyplot as plt
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
def get_mag(flux, ww, filtername='f140m' ):
   # print(ww.proj_plane_pixel_area())
    
    #flux= (catalog['flux_fit_' + filtername] * u.MJy/u.sr * ww.proj_plane_pixel_area()).to(u.Jy)
    #eflux_jy = (catalog['flux_err_' + filtername] * u.MJy/u.sr *  ww.proj_plane_pixel_area()).to(u.Jy)

    jfilts = SvoFps.get_filter_list('JWST')
    jfilts.add_index('filterID')
    wav = int(filtername[1:-1])

    zeropoint_ab = 3631 * u.Jy  # Default to AB magnitude zero point
 
    if wav < 500:

        zeropoint_vega = u.Quantity(jfilts.loc[f'JWST/NIRCam.{filtername.upper()}']['ZeroPoint'], u.Jy)
    else:
        zeropoint_vega = u.Quantity(jfilts.loc[f'JWST/MIRI.{filtername.upper()}']['ZeroPoint'], u.Jy)
   
    abmag = -2.5 * np.log10(flux / zeropoint_ab) * u.mag
    #abmag_err = 2.5 / np.log(10) * np.abs(eflux_jy / flux) * u.mag

    vegamag = -2.5 * np.log10(flux / zeropoint_vega) * u.mag
    #vegamag_err = 2.5 / np.log(10) * np.abs(eflux_jy / flux) * u.mag

    return  vegamag
def main():
    parser = OptionParser()
    parser.add_option("--history", dest="history", type="string",
                    default='ca',
                    help="star formation model", metavar="history")
    parser.add_option("--timescale", dest="timescale", type="float",
                    default=1.0,
                    help="timescale", metavar="timescale")
    parser.add_option("--time", dest="time", type="float",
                    default=0.25,
                    help="time", metavar="time")
    parser.add_option("--efficiency", dest="efficiency", type="int",
                    default=100,)
    parser.add_option("--mass", dest="mass", type="float",
                    default=5e3,
                    help="mass", metavar="mass")
    (options, args) = parser.parse_args()

    history = options.history
    timescale = options.timescale * u.Myr
    sample_time = options.time * u.Myr
    efficiency = options.efficiency
    mass = options.mass * u.Msun

    cl = Cluster(distance=5.4 * u.kpc, timescale=timescale * u.Myr, mass=mass, history=history, efficiency=efficiency )
  

    sample_ap = np.array([656])*u.AU
    sample_wavs = np.array([1.62, 2.10, 3.60, 4.80])*u.um
    flux = cl.sample_flux(sample_time, wav=sample_wavs, ap=sample_ap)
    ev_table = cl.sample_ev(sample_time)

    


    f162m_mag = get_mag(flux[:,0].to(u.Jy), WCS(f162m_header), filtername='f162m')
    f210m_mag = get_mag(flux[:,1].to(u.Jy), WCS(f210m_header), filtername='f210m')
    f360m_mag = get_mag(flux[:,2].to(u.Jy), WCS(f360m_header), filtername='f360m')
    f480m_mag = get_mag(flux[:,3].to(u.Jy), WCS(f480m_header), filtername='f480m')

    ev_table['f162m_mag'] = f162m_mag
    ev_table['f210m_mag'] = f210m_mag
    ev_table['f360m_mag'] = f360m_mag
    ev_table['f480m_mag'] = f480m_mag

    savedir = '/orange/adamginsburg/jwst/w51/protopop/'

    ev_table.write(f'{savedir}/ev_table_{history}_mass{mass.value}_ts_{timescale.value}_t{sample_time.value}_eff{efficiency}.fits', overwrite=True)

if __name__ == "__main__":
    main()
    
