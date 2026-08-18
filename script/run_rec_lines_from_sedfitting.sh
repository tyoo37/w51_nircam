#!/bin/bash
mem=32gb
taskname=rec_lines_from_sed
sbatch --job-name=${taskname} --output=${taskname}_%j-%A.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "python /blue/adamginsburg/t.yoo/w51_nircam/analysis/HII_regions/get_bias_for_rec_line.py"
