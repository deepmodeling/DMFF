import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

pdb = app.PDBFile("tests/data/methane_water.pdb")
ff = app.ForceField("tests/data/methane_water_coul.xml")
system = ff.createSystem(
            pdb.topology, 
            nonbondedMethod=app.PME, 
            constraints=app.HBonds, 
            removeCMMotion=False, 
            nonbondedCutoff=0.5 * unit.nanometers,
            useDispersionCorrection=False
        )
ctx = mm.Context(system, mm.VerletIntegrator(0.1))
ctx.setPositions(pdb.positions)
state = ctx.getState(getEnergy=True)
print(state.getPotentialEnergy())