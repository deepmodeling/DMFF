#!/usr/bin/env python3
import os
import sys
import driver
import numpy as np
import EANN
#import time


class OpenmmDriver(driver.BaseDriver):

    def __init__(self, addr, port, socktype):
        addr = addr + '_%s'%os.environ['SLURM_JOB_ID'] 
        driver.BaseDriver.__init__(self, port, addr, socktype)
        
        return

    def grad(self, crd, cell): # receive SI input, return SI values
        positions = np.array(crd*1e10) # convert to Angstrom
        length = np.size(positions,0)
        coor = np.zeros((3,length),dtype=np.float64,order="F")
        coor = positions.transpose()
        y = np.zeros(1,dtype=np.float64,order="F")
        eaforce = np.zeros(3*length,dtype=np.float64,order="F")
        table = 0
        start_force = 1 
        #time1 = time.time()
        EANN.eann_out(table,start_force,coor,y,eaforce)
        #time2 = time.time()
        #print('calculate:',time2-time1)
        # finish calculating but wrong format & unit
        grad = np.zeros((length,3))
        grad = eaforce.reshape(length,3)

        # convert to SI
        energy = y * 1000 / 6.0221409e+23 # kj/mol to Joules
        grad = -(grad * 1000 / 6.0221409e+23 * 1e10) # convert kj/mol/A to joule/m
        #print(grad) 
        return energy, grad

if __name__ == '__main__':
    EANN.init_pes()
    addr = sys.argv[1]
    port = int(sys.argv[2])
    socktype = sys.argv[3]
    driver_openmm = OpenmmDriver(addr, port, socktype)
    while True:
        driver_openmm.parse()
    EANN.deallocate_all()

