#!/usr/bin/env python
import sys
from openmm import *
from openmm.app import *
from openmm.unit import *
import openmm.app as app
import mpidplugin


if __name__ == '__main__':

    ff = ForceField('forcefield.xml')
    nb_gen, pme_gen = ff.getGenerators()
    app.Topology.loadBondDefinitions("residues.xml")
    pdb = app.PDBFile("waterbox_31ang.pdb")
    rc = 15
    system = ff.createSystem(pdb.topology, nonbondedCutoff=rc*angstrom, nonbondedMethod=PME, rigidWater=False)

    nb_force, pme_force, _ = system.getForces()
    pme_gen.force.setPolarizationType(0)
    nb_force.setForceGroup(1)
    pme_force.setForceGroup(2)
    
    platform = Platform.getPlatformByName('Reference')
    properties = {}

    integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 0.5*femtosecond)

    simulation = Simulation(pdb.topology, system, integrator, platform, properties)
    simulation.context.setPositions(pdb.positions)

    state = simulation.context.getState(getEnergy=True, groups=2**2)
    print('Electrostatic+Polarization Energy:')
    print(state.getPotentialEnergy())

    state = simulation.context.getState(getEnergy=True, groups=2**1)
    print('Dispersion+Damping Energy:')
    print(state.getPotentialEnergy())
