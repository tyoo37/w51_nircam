#!/bin/bash
mem=32gb
taskname=sed_fitting
export STPSF_PATH=/blue/adamginsburg/t.yoo/from_red/stpsf-data
sbatch --job-name=${taskname}_upper --output=${taskname}_upper_%j-%A_%a.log  --array=0-19 --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "python /blue/adamginsburg/t.yoo/w51_nircam/analysis/classfication/sed_fitter_for_upper_cleancat.py"    
sbatch --job-name=${taskname}_lower --output=${taskname}_lower_%j-%A_%a.log  --array=0-19 --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "python /blue/adamginsburg/t.yoo/w51_nircam/analysis/classfication/sed_fitter_for_lower_cleancat.py" 

