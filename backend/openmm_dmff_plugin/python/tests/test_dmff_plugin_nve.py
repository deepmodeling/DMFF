import os
import numpy as np
import time
import argparse

try:
    import openmm as mm
    from openmm import unit as u
    from openmm.app import PDBFile, StateDataReporter, DCDReporter, Simulation
except:
    import simtk.openmm as mm
    from simtk import unit as u
    from simtk.openmm.app import PDBFile, StateDataReporter, DCDReporter, Simulation

from OpenMMDMFFPlugin import DMFFModel


def test_dmff_nve(nsteps = 1000, time_step = 0.2, pdb_file = None, model_dir = None, platform_name = "Reference", output_temp_dir = "/tmp/openmm_dmff_plugin_test_nve_output", energy_std_tol = 0.005, has_aux = False ):
    if not os.path.exists(output_temp_dir):
        os.mkdir(output_temp_dir)
    
    dmff_model_file = model_dir
    
    output_dcd = os.path.join(output_temp_dir, "test.nve.dcd")
    output_log = os.path.join(output_temp_dir, "test.nve.log")
    
    # Set up the simulation parameters.
    nsteps = nsteps
    time_step = time_step # unit is femtosecond.
    report_frequency = 10
    box = [24.413, 0, 0, 0, 24.413, 0, 0, 0, 24.413]
    box = [mm.Vec3(box[0], box[1], box[2]), mm.Vec3(box[3], box[4], box[5]), mm.Vec3(box[6], box[7], box[8])] * u.angstroms
    
    liquid_water = PDBFile(pdb_file)
    topology = liquid_water.topology
    positions = liquid_water.getPositions()
    num_atoms = topology.getNumAtoms()
    
    # Set up the dmff_system with the dmff_model.    
    dmff_model = DMFFModel(dmff_model_file)
    dmff_model.setUnitTransformCoefficients(1, 1, 1)
    if has_aux:
        dmff_model.setHasAux()
    dmff_system = dmff_model.createSystem(topology)
    
    integrator = mm.VerletIntegrator(time_step*u.femtoseconds)
    platform = mm.Platform.getPlatformByName(platform_name)
    
    # Build up the simulation object.
    sim = Simulation(topology, dmff_system, integrator, platform)
    sim.context.setPeriodicBoxVectors(box[0], box[1], box[2])
    sim.context.setPositions(positions)

    # Add state reporters
    sim.reporters.append(DCDReporter(output_dcd, report_frequency, enforcePeriodicBox=False))
    sim.reporters.append(
        StateDataReporter(output_log, report_frequency, step=True, time=True, totalEnergy=True, kineticEnergy=True, potentialEnergy=True, temperature=True, progress=True,
                          remainingTime=True, speed=True,  density=True,totalSteps=nsteps, separator='\t')
    )
    
    # Run dynamics
    print("Running dynamics")
    start_time = time.time()
    sim.step(nsteps)
    end_time = time.time()
    cost_time = end_time - start_time
    print("Running on %s platform, time cost: %.4f s"%(platform_name, cost_time))
    
    # Fetch the total energy from the log file.
    total_energy = []
    tot_energy_index = -5
    with open(output_log, "r") as f:
        log_content = f.readlines()
    for ii , line in enumerate(log_content):
        if ii == 0:
            continue
        temp = line.split()
        total_energy.append(float(temp[tot_energy_index]))
    total_energy = np.array(total_energy)
    
    # Check the total energy fluctuations over # of atoms is smaller than energy_std_tol, unit in kJ/mol.
    print("Total energy std: %.4f kJ/mol"%(np.std(total_energy)))
    print("Mean total energy: %.4f kJ/mol"%(np.mean(total_energy)))
    assert(np.std(total_energy) / num_atoms < energy_std_tol)    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nsteps', type = int, dest='nsteps', help='Number of steps', default=100)
    parser.add_argument('--dt', type = float, dest='timestep', help='Time step for simulation, unit is femtosecond', default=0.2)
    parser.add_argument('--pdb', type = str, dest='pdb', help='PDB file for simulation.', default=None)
    parser.add_argument('--model', type = str, dest='model', help='DMFF model dir for simulation. Saved by backend/save_dmff2tf.py.', default=None)
    parser.add_argument('--platform', type = str, dest='platform', help='Platform for simulation.', default="Reference")
    parser.add_argument('--has_aux', type = bool, dest='has_aux', help='Whether the model has aux output.', default=False)
    
    args = parser.parse_args()

    nsteps = args.nsteps
    time_step = args.timestep
    platform_name = args.platform
    
    pdb = args.pdb
    model_dir = args.model
    
    if pdb is None:
        pdb = os.path.join(os.path.dirname(__file__), "../data", "lj_fluid.pdb")
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), "../data", "lj_fluid_gpu")
    
    test_dmff_nve(nsteps=nsteps, time_step=time_step, pdb_file=pdb, model_dir=model_dir, platform_name=platform_name)
    
