#!/bin/bash

export OMP_NUM_THREADS=8
export OMP_STACKSIZE=2000000
export MODULEPATH=$MODULEPATH:/share/home/kuangy/modulefiles/
module load Anaconda/anaconda3/2019.10
module load compiler/intel/ips2018/u1
module load mkl/intel/ips2018/u1
module load EANN/2.0

source activate EANN

addr=unix_eann
port=1257
socktype=unix

python3 client_EANN.py $addr $port $socktype > logEANN

conda deactivate
