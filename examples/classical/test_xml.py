#!/usr/bin/env python
import openmm as mm 
import openmm.app as app
import openmm.unit as unit 
import numpy as np
import sys
from dmff import Hamiltonian
from dmff.common import nblist
from jax import jit
import jax.numpy as jnp

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
        energies[f] = context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()
    return energies

if __name__ == "__main__":

    print("MM Reference Energy:")
    app.Topology.loadBondDefinitions("lig-top.xml")
    pdb = app.PDBFile("lig.pdb")
    ff = app.ForceField("gaff-2.11.xml", "lig-prm.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, removeCMMotion=False)

    print("Dih info:")
    for force in system.getForces():
        if isinstance(force, mm.PeriodicTorsionForce):
            print("No. of dihs:", force.getNumTorsions())

    forcegroups = forcegroupify(system)
    integrator = mm.VerletIntegrator(0.1)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions(pdb.positions)
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy()
    energies = getEnergyDecomposition(context, forcegroups)
    print(energy)
    for key in energies.keys():
        print(key.getName(), energies[key])

    print()
    print("Jax Energy")
    
    
    h = Hamiltonian("gaff-2.11.xml", "lig-prm.xml")
    pot = h.createPotential(pdb.topology, nonbondedMethod=app.NoCutoff)
    params = h.getParameters()

    positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    positions = jnp.array(positions)
    box = np.array([
        [10.0,  0.0,  0.0],
        [ 0.0, 10.0,  0.0],
        [ 0.0,  0.0, 10.0]
    ])
    
    # neighbor list
    rc = 4
    nbl = nblist.NeighborList(box, rc, pot.meta['cov_map'])
    nbl.allocate(positions)
    pairs = nbl.pairs

    bondE = pot.dmff_potentials['HarmonicBondForce']
    print("Bond:", bondE(positions, box, pairs, params))

    angleE = pot.dmff_potentials['HarmonicAngleForce']
    print("Angle:", angleE(positions, box, pairs, params))

    dihE = pot.dmff_potentials['PeriodicTorsionForce']
    print("Torsion:", dihE(positions, box, pairs, params))

    nbE = pot.dmff_potentials['NonbondedForce']
    print("Nonbonded:", nbE(positions, box, pairs, params))

    etotal = pot.getPotentialFunc()
    print("Total:", etotal(positions, box, pairs, params))

