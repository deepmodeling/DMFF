import openmm as mm
import openmm.app as app
import openmm.unit as unit


pdb = app.PDBFile("structure.pdb")
ff = app.ForceField("param.xml")
system = ff.createSystem(pdb.topology, nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=1.0*unit.nanometer)
integrator = mm.VerletIntegrator(0.001*unit.picoseconds)
context = mm.Context(system, integrator)
context.setPositions(pdb.positions)
state = context.getState(getEnergy=True)
print(state.getPotentialEnergy())