#!/bin/bash

#clariden
for cfgfile in $(ls src/hirad/input_data/configs/era-all-2020*.yaml);
do
    cmd="sbatch -A a161 -t 12:00:00 -n 1 -c 1 --begin=now+18hour --environment=modulus_env src/hirad/interpolate.sh ${cfgfile}"
    echo $cmd
    $cmd
done

# balfrin
#for cfgfile in $(ls src/hirad/input_data/configs/era-all-2016*.yaml);
#do
#    cmd="sbatch -p postproc -t 12:00:00 -n 1 -c 1 src/hirad/interpolate.sh ${cfgfile}"
#    echo $cmd
#    $cmd
#done
