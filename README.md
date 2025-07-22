### Reduction
Reduction is based on NIRCAM pipeline in brick repository
The main file is `reduction/run_pipeline_with_destreak.py`. This will run `pipeline_rerun_copy_of_NIRCAM-LONG_copied.py` which has been tweaked from the one in brick repository. Saturated pixels are partly recovered but not fully with this pipeline.
1/f noise pattern is cleared out with `destreak/destreak_auto.py`. By default, `image1overf` method is used.
