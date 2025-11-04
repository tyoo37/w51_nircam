
#for module in nrca nrcb merged; do
#    sbatch --array=0-10 --job-name=webb-cat-${module}-dao --output=webb-cat-${module}-dao_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=64gb --time=96:00:00 --wrap "python /home/t.yoo/w51/w51_nircam/catalog/merge_catalog.py --modules=$module --indiv-merge-methods=dao --skip-crowdsource"
#    sbatch --array=0-10 --job-name=webb-cat-${module}-crowdsource --output=webb-cat-${module}-crowdsource_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=64gb --time=96:00:00 --wrap "python /home/t.yoo/w51/w51_nircam/catalog/merge_catalog.py --modules=$module --indiv-merge-methods=crowdsource --skip-dao"
#    sbatch --array=0-10 --job-name=webb-cat-${module}-iterative --output=webb-cat-${module}-iterative_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=64gb --time=96:00:00 --wrap "python /home/t.yoo/w51/w51_nircam/catalog/merge_catalog.py --modules=$module --indiv-merge-methods=iterative --skip-crowdsource"
#done

#for module in nrca nrcb merged; do
for module in nrca nrcb; do
    #sbatch --array=0-20 --job-name=webb-cat-${module}-singlefields-dao --output=webb-cat-${module}-singlefields-dao_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=64gb --time=96:00:00 --wrap "python /home/t.yoo/w51/w51_nircam/catalog/merge_catalog.py --merge-singlefields --modules=${module} --indiv-merge-methods=dao --skip-crowdsource"
    sbatch --array=0-31 --job-name=merge-${module}-singlefields-dao --output=webb-cat-${module}-singlefields-dao_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=64gb --time=96:00:00 --wrap "python /home/t.yoo/w51/w51_nircam/catalog/merge_catalog_after_appending_sat_stars.py --merge-singlefields --modules=${module} --indiv-merge-methods=dao_after_merger --skip-crowdsource"
   # sbatch --array=0-20 --job-name=webb-cat-${module}-singlefields-crowdsource --output=webb-cat-${module}-singlefields-crowdsource_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=64gb --time=96:00:00 --wrap "python /home/t.yoo/w51/w51_nircam/catalog/merge_catalog.py --merge-singlefields --modules=${module} --indiv-merge-methods=crowdsource --skip-dao"
   # sbatch --array=0-20 --job-name=webb-cat-${module}-singlefields-iterative --output=webb-cat-${module}-singlefields-iterative_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=64gb --time=96:00:00 --wrap "python /home/t.yoo/w51/w51_nircam/catalog/merge_catalog.py --merge-singlefields --modules=${module} --indiv-merge-methods=iterative --skip-crowdsource"
done
#sbatch --array=0-31 --job-name=webb-cat-merge-singlefields-crowdsource --output=webb-cat-merge-singlefields-crowdsource_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=64gb --time=96:00:00 --wrap "python /home/t.yoo/w51/w51_nircam/catalog/merge_catalog.py --merge-singlefields --modules=merged --indiv-merge-methods=crowdsource --skip-dao"
#sbatch --array=0-31 --job-name=webb-cat-merge-singlefields-iterative --output=webb-cat-merge-singlefields-iterative%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=64gb --time=96:00:00 --wrap "python /home/t.yoo/w51/w51_nircam/catalog/merge_catalog.py --merge-singlefields --modules=merged --indiv-merge-methods=iterative --skip-crowdsource"


# #!/bin/bash
# #SBATCH --job-name=webb-cat-merge-singlefields
# #SBATCH --output=webb-cat-merge-singlefields_%j_%A_%a.out
# #SBATCH --error=webb-cat-merge-singlefields_%j_%A_%a.err
# #SBATCH --array=0-9
# #SBATCH --account=astronomy-dept
# #SBATCH --qos=astronomy-dept-b
# #SBATCH --ntasks=1
# #SBATCH --nodes=1
# #SBATCH --time=96:00:00
# #SBATCH --mem=16gb
# 
# # filter order
# filternames=(f410m f212n f466n f405n f187n f182m f444w f356w f200w f115w)
# memory=(16gb 128gb 16gb 16gb 32gb 128gb 16gb 16gb 256gb 256gb)
# MEM=${memory[$SLURM_ARRAY_TASK_ID]}
# echo $SLURM_ARRAY_TASK_ID $MEM
# 
# #SBATCH --mem=${MEM}
# 
# srun --mem=$MEM /blue/adamginsburg/adamginsburg/miniconda3/envs/python310/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/merge_catalogs.py --merge-singlefields --modules=merged
# 
# 
# # sbatch --array=0-9 --job-name=webb-cat-merge-singlefields-dao --output=webb-cat-merge-singlefields-dao_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=32gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python310/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=dao --skip-crowdsource"
# # sbatch --array=0-9 --job-name=webb-cat-merge-singlefields-crowdsource --output=webb-cat-merge-singlefields-crowdsource_%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=32gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python310/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=crowdsource --skip-dao"
# # sbatch --array=0-9 --job-name=webb-cat-merge-singlefields-iterative --output=webb-cat-merge-singlefields-iterative%j-%A_%a.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=1 --nodes=1 --mem=32gb --time=96:00:00 --wrap "/blue/adamginsburg/adamginsburg/miniconda3/envs/python310/bin/python /blue/adamginsburg/adamginsburg/jwst/brick/analysis/merge_catalogs.py --merge-singlefields --modules=merged --indiv-merge-methods=iterative --skip-crowdsource"