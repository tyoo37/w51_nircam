#!/bin/bash
export HDF5_USE_FILE_LOCKING=FALSE
mem=32gb
taskname=protopop_ccd
for history in ca is tc expl; do
    for timescale in 1.0 2.0 4.0; do
        for efficiency in 100; do
            for snap_time in 0.2 0.5 1.0; do
                sbatch --job-name=${taskname}_his${history}_ts${timescale}_eff${efficiency}_st${snap_time} --output=${taskname}_his${history}_ts${timescale}_eff${efficiency}_st${snap_time}_%j-%A.log  --account=astronomy-dept --qos=astronomy-dept-b --ntasks=2 --nodes=1 --mem=${mem} --time=96:00:00 --wrap "python /blue/adamginsburg/t.yoo/w51_nircam/analysis/protopop/ccd_protopop.py --history ${history} --timescale ${timescale} --efficiency ${efficiency} --time ${snap_time}"
            done
        done
    done
done

