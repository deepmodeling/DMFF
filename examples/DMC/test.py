#!/usr/bin/env python
import jax
import jax.numpy as jnp
from openmm import *
from openmm.app import *
from openmm.unit import *
import openmm.openmm as mm
from dmff import *

top_pdb = app.PDBFile("box_DMC.pdb")
rc = 0.9
xml = 'prm1.xml'

# OpenMM Calc
ff = ForceField(xml)

system = ff.createSystem(top_pdb.topology, 
                         nonbondedMethod=app.PME, 
                         nonbondedCutoff=rc*unit.nanometer, 
                         constraints=None)
barostat = mm.MonteCarloBarostat(1.0*unit.bar, 293.0*unit.kelvin, 20)
#barostat.setRandomNumberSeed(12345)
system.addForce(barostat)
for force in system.getForces():
    if isinstance(force, mm.NonbondedForce):
        force.setUseDispersionCorrection(False)
        force.setUseSwitchingFunction(False)
integ = mm.LangevinIntegrator(293*unit.kelvin, 5/unit.picosecond, 1*unit.femtosecond)

simulation = app.Simulation(top_pdb.topology, system, integ)
simulation.context.setPositions(top_pdb.getPositions()) 
state = simulation.context.getState(getEnergy=True)
print(state.getPotentialEnergy())

# DMFF Calc
H = Hamiltonian(xml)


pot = H.createPotential(top_pdb.topology, 
                        nonbondedMethod=app.PME, 
                        nonbondedCutoff=rc*unit.nanometer, 
                        useDispersionCorrection=False)
efunc = pot.getPotentialFunc()

pos = jnp.array(top_pdb.getPositions()._value)
box = jnp.array(top_pdb.topology.getPeriodicBoxVectors()._value)
nbl = dmff.NeighborListFreud(box, rc, pot.meta['cov_map'])
nbl.allocate(pos)
pairs = nbl.pairs

print(efunc(pos, box, pairs, H.paramtree))
