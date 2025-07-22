import subprocess
import os
import shutil
# Example command and arguments
filtername='F187N'
os.chdir('/home/t.yoo/w51/w51_nircam/reduction/')
command = [
    "python",
    "NIRCAM_pipeline_short.py",
    "--proposal_id=6151",
    f"--filternames={filtername}"
]

# Run NIRCAM pipeline for nrca, nrcb, merged
result = subprocess.run(command, capture_output=True, text=True)

command = [
    "python",
    "destreak_auto.py",
    f"--filternames={filtername}"
]
# Run destreaking -> move original cal files to no_destreak folder and overwrite cal files with destreaked ones
result = subprocess.run(command, capture_output=True, text=True)

command = [
    "python",
    "NIRCAM_pipeline_short.py",
    "--proposal_id=6151",
    f"--filternames={filtername}",
    "--module=merged"
]
# copy the original merged files to no_destreak folder

shutil.copy(f'/orange/adamginsburg/jwst/w51/{filtername}/pipeline/jw06151-o001_t001_nircam_clear-{filtername.lower()}-merged_i2d.fits', f'/orange/adamginsburg/jwst/w51/{filtername}/no_destreak/jw06151-o001_t001_nircam_clear-{filtername.lower()}-merged_i2d.fits')

# Run NIRCAM pipeline for merged (will generate merged for destreaked ones)
result = subprocess.run(command, capture_output=True, text=True)


