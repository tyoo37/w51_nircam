#!/bin/bash
mem=16gb
taskname=nircam_pipe

for filter in F140M F150W F162M F182M F187N F210M F335M F360M F405N F410M F480M; do   
    sbatch --job-name=${taskname}-${filter}-${module}-eachexp --output=${taskname}-${filter}-${module}-eachexp_%j-%A.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "python /home/t.yoo/w51/w51_nircam/pipeline_rerun_copy_of_NIRCAM-LONG.py --filternames=${filter} --proposal_id=6151"      
done