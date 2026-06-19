#!/bin/bash

mem=64gb
taskname=reproject
sbatch --job-name=${taskname} --output=${taskname}_%j-%A.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "python /home/t.yoo/w51/w51_nircam/tool/reproject/reproject_to_alma_extended.py"