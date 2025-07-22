### Reduction
Reduction is based on NIRCAM pipeline in brick repository (https://github.com/keflavich/brick-jwst-2221/tree/main)
The main file is `reduction/run_pipeline_with_destreak.py`. This will run `pipeline_rerun_copy_of_NIRCAM-LONG_copied.py` which has been tweaked from the one in brick repository. Saturated pixels are partly recovered but not fully with this pipeline.
1/f noise pattern is cleared out (but not entirely!) with `destreak/destreak_auto.py`. By default, `image1overf` method (https://github.com/chriswillott/jwst) is used.
