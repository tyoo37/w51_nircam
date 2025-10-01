#!/bin/bash
mem=64gb
taskname=reprojection
#for filter in F140M F150W F162M F182M F187N F210M F335M F360M F405N F410M F480M; do
sbatch --job-name=${taskname} --output=${taskname}_%j-%A.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "python /home/t.yoo/w51/w51_nircam/reproject_images.py"  
