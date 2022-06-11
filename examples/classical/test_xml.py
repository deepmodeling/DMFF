import openmm as mm 
import openmm.app as app
import openmm.unit as unit 
import numpy as np
from jax_md import space, partition
import sys
from dmff.api import Hamiltonian
from jax import jit

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
    context = mm.Context(system, integrator)
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
    system = h.createPotential(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, removeCMMotion=False)

    positions = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    box = np.array([
        [10.0,  0.0,  0.0],
        [ 0.0, 10.0,  0.0],
        [ 0.0,  0.0, 10.0]
    ])
    
    # neighbor list
    rc = 4
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
    nbr = neighbor_list_fn.allocate(positions)
    pairs = nbr.idx.T        

    bondE = h._potentials[0]
    print("Bond:", bondE(positions, box, pairs, h.getGenerators()[0].params))

    angleE = h._potentials[1]
    print("Angle:", angleE(positions, box, pairs, h.getGenerators()[1].params))

    dihE = h._potentials[2]
    print("Torsion:", dihE(positions, box, pairs, h.getGenerators()[2].params))
