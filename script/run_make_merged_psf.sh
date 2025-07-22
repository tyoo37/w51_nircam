#!/bin/bash

mem=32gb
taskname=make_merged_psf

for filter in F162M F182M F187N F335M F360M F410M F480M; do
    echo "Processing filter: ${filter}"
    sbatch --job-name=${taskname}-${filter} --output=${taskname}-${filter}_%j-%A.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "python /home/t.yoo/w51/w51_nircam/catalog/make_merged_psf.py --filternames=${filter} --proposal_id=6151"
done