#!/bin/bash
mem=64gb
taskname=crowdsource
#for filter in F140M F150W F162M F182M F187N F210M F335M F360M F405N F410M F480M; do   
for filter in F140M F150W F210M F405N ; do
#    for module in nrca nrcb merged; do
    for module in merged; do
        if [ "${module}" == "merged" ]; then
            sbatch --job-name=${taskname}-${filter}-${module}-eachexp --output=${taskname}-${filter}-${module}-eachexp_%j-%A.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "python /home/t.yoo/w51/w51_nircam/catalog/crowdsource_catalogs_long.py --filternames=${filter} --proposal_id=6151 --module=${module} --daophot " 
        else
            sbatch --job-name=${taskname}-${filter}-${module}-eachexp --output=${taskname}-${filter}-${module}-eachexp_%j-%A.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "python /home/t.yoo/w51/w51_nircam/catalog/crowdsource_catalogs_long.py --filternames=${filter} --proposal_id=6151 --module=${module} --each-exposure --daophot "
        fi
           
    done  
done