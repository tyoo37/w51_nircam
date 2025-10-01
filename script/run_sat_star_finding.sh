#!/bin/bash
mem=64gb
taskname=sat_star_finding


#daoloop=("--daophot --skip-crowdsource" " ")
#mem=32gb

for filter in F140M F150W F162M F182M F187N F210M F335M F360M F405N F410M F480M; do
    sbatch --job-name=webb_sat_star_find-${filter} --output=webb_sat_star_find-${filter}_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "python /home/t.yoo/w51/w51_nircam/catalog/saturated_star_finding.py --filter=${filter}"
done


