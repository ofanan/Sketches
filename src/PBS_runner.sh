#!/bin/bash

### PBS -q [WD, RGB, RTX]
#PBS -q WD
#PBS -k eod
#PBS -e log.err
#PBS -o log.out
#PBS -l select=1:ncpus=36:mpiprocs=36

###Always load modules first
module load python/3/3.9.6 gcc hdf5/serial

cd $PBS_O_WORKDIR

# # # python3 ./SingleCntrSimulator.py
python3 ./CountMinSketch.py

