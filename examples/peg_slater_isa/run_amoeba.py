#!/usr/bin/env python
import sys
import numpy as np
import openmm
from openmm import *
from openmm.app import *
from openmm.unit import *
import pickle

if __name__ == '__main__':
    pdb_AB = PDBFile('peg2_dimer.pdb')
    pdb_A = PDBFile('peg2.pdb')
    pdb_B = PDBFile('peg2.pdb')
    forcefield = ForceField('benchmark.xml')

    system_AB = forcefield.createSystem(pdb_AB.topology, nonbondedMethod=PME, nonbondedCutoff=15*angstrom)
    system_A = forcefield.createSystem(pdb_A.topology, nonbondedMethod=PME, nonbondedCutoff=15*angstrom)
    system_B = forcefield.createSystem(pdb_B.topology, nonbondedMethod=PME, nonbondedCutoff=15*angstrom)
    forces_AB = system_AB.getForces()
    forces_A = system_A.getForces()
    forces_B = system_B.getForces()
    for i in range(len(forces_AB)):
        forces_AB[i].setForceGroup(i)
        forces_A[i].setForceGroup(i)
        forces_B[i].setForceGroup(i)

    platform_AB = Platform.getPlatformByName('CUDA')
    platform_A = Platform.getPlatformByName('CUDA')
    platform_B = Platform.getPlatformByName('CUDA')
    properties = {}

    integrator_AB = LangevinIntegrator(300*kelvin, 1.0/picosecond, 1*femtosecond)
    integrator_A = LangevinIntegrator(300*kelvin, 1.0/picosecond, 1*femtosecond)
    integrator_B = LangevinIntegrator(300*kelvin, 1.0/picosecond, 1*femtosecond)

    simulation_AB = Simulation(pdb_AB.topology, system_AB, integrator_AB, platform=platform_AB)
    simulation_A = Simulation(pdb_A.topology, system_A, integrator_A, platform=platform_A)
    simulation_B = Simulation(pdb_B.topology, system_B, integrator_B, platform=platform_B)


    pos_AB0 = np.array(pdb_AB.positions._value) * 10
    n_atoms = len(pos_AB0)
    n_atoms_A = n_atoms // 2
    n_atoms_B = n_atoms // 2
    pos_A0 = pos_AB0[:n_atoms_A]
    pos_B0 = pos_AB0[n_atoms_A: n_atoms]

    with open('data.pickle', 'rb') as ifile:
        data = pickle.load(ifile)

    for sid in ['000']:
        scan_res = data[sid]

        for ipt in range(len(scan_res['posA'])):
            pos_A = np.array(scan_res['posA'][ipt])
            pos_B = np.array(scan_res['posB'][ipt])
            pos_AB = np.vstack([pos_A, pos_B])
            E_es_ref = scan_res['es'][ipt]
            E_pol_ref = scan_res['pol'][ipt]

            simulation_AB.context.setPositions(pos_AB * angstrom)
            simulation_A.context.setPositions(pos_A * angstrom)
            simulation_B.context.setPositions(pos_B * angstrom)

            state_AB = simulation_AB.context.getState(getEnergy=True, groups=2**0)
            state_A = simulation_A.context.getState(getEnergy=True, groups=2**0)
            state_B = simulation_B.context.getState(getEnergy=True, groups=2**0)
            # state_AB = simulation_AB.context.getState(getEnergy=True)
            # state_A = simulation_A.context.getState(getEnergy=True)
            # state_B = simulation_B.context.getState(getEnergy=True)

            E_AB = state_AB.getPotentialEnergy()._value
            E_A = state_A.getPotentialEnergy()._value
            E_B = state_B.getPotentialEnergy()._value

            print(E_AB - E_A - E_B)
