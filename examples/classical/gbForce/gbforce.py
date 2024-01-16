#!/usr/bin/env python
from openmm import *
from openmm.app import *
from openmm.unit import *
import numpy as np
import sys
sys.path.append("..")
sys.path.append(".")
sys.path.append("...")
from dmff.api.hamiltonian import Hamiltonian
from dmff.common import nblist
from jax import jit
import jax.numpy as jnp
import mdtraj as md

def forcegroupify(system):
    forcegroups = {}
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        force.setForceGroup(i)
        forcegroups[force] = i
    return forcegroups

def getEnergyDecomposition(context, forcegroups):
    energies = {}
    for f, i in forcegroups.items():
        energies[f] = context.getState(getEnergy=True, groups=2 ** i).getPotentialEnergy()
    return energies

print("MM Reference Energy:")
pdb = PDBFile("10p.pdb")
ff = ForceField("1_5corrV2.xml")
system = ff.createSystem(pdb.topology, nonbondedMethod=NoCutoff, constraints=None, removeCMMotion=False)
h = Hamiltonian("1_5corrV2.xml")
params = h.getParameters()
compoundBondForceParam = params["Custom1_5BondForce"]
length = compoundBondForceParam["length"]
k = compoundBondForceParam["k"]
customCompoundForce = openmm.CustomCompoundBondForce(2, "0.5*k*(distance(p1,p2)-length)^2")
customCompoundForce.addPerBondParameter("length")
customCompoundForce.addPerBondParameter("k")
for i, leng in enumerate(length):
    customCompoundForce.addBond([i, i+4], [leng, k[i]])
system.addForce(customCompoundForce)
print("Dih info:")
for force in system.getForces():
    if isinstance(force, PeriodicTorsionForce):
        print("No. of dihs:", force.getNumTorsions())

forcegroups = forcegroupify(system)
integrator = VerletIntegrator(0.1)
context = Context(system, integrator, Platform.getPlatformByName("Reference"))
context.setPositions(pdb.positions)
state = context.getState(getEnergy=True)
energy = state.getPotentialEnergy()
energies = getEnergyDecomposition(context, forcegroups)
print("Total energy:", energy)
for key in energies.keys():
    print(key.getName(), energies[key])

print("Jax Energy")
h = Hamiltonian("1_5corrV2.xml")
pot = h.createPotential(pdb.topology, nonbondedMethod=NoCutoff)
params = h.getParameters()
positions = pdb.getPositions(asNumpy=True).value_in_unit(nanometer)
positions = jnp.array(positions)
box = np.array([
    [30.0, 0.0, 0.0],
    [0.0, 30.0, 0.0],
    [0.0, 0.0, 30.0]
])

# neighbor list
rc = 6.0
nbl = nblist.NeighborList(box, rc, pot.meta['cov_map'])
nbl.allocate(positions)
pairs = nbl.pairs

bondE = pot.dmff_potentials['HarmonicBondForce']
print("Bond:", bondE(positions, box, pairs, params))

angleE = pot.dmff_potentials['HarmonicAngleForce']
print("Angle:", angleE(positions, box, pairs, params))

gbE = pot.dmff_potentials['CustomGBForce']
print("CustomGBForce:", gbE(positions, box, pairs, params))

E1_5 = pot.dmff_potentials['Custom1_5BondForce']
print("Custom1_5BondForce:", E1_5(positions, box, pairs, params))

dihE = pot.dmff_potentials['CustomTorsionForce']
print("Torsion:", dihE(positions, box, pairs, params))

nbE = pot.dmff_potentials['NonbondedForce']
print("Nonbonded:", nbE(positions, box, pairs, params))

etotal = pot.getPotentialFunc()
print("Total:", etotal(positions, box, pairs, params))

