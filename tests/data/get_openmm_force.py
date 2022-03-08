import openmm as mm 
import openmm.app as app
import openmm.unit as unit 
import numpy as np

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
    pdb = app.PDBFile("lj1.pdb")
    ff = app.ForceField("lj1.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None, removeCMMotion=False)
    

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

    
