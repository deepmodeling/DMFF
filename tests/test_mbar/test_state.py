from dmff.mbar import MBAREstimator, Sample, SampleState, TargetState, OpenMMSampleState, buildTrajEnergyFunction
import dmff
import pytest
import jax
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
import openmm as mm
import numpy as np
import numpy.testing as npt
import mdtraj as md
from pymbar import MBAR
from dmff import Hamiltonian, NeighborListFreud


class TestState:
    @pytest.mark.parametrize(
        "pdb, prm1, traj1",
        [("tests/data/waterbox.pdb", "tests/data/water1.xml",
          "tests/data/w1_npt.dcd")])
    def test_openmm_sampler_state(self, pdb, prm1, traj1):
        pdbobj = app.PDBFile(pdb)

        ff_settings = [
                {'nonbondedMethod': app.PME,
                 'nonbondedCutoff': 0.95 * unit.nanometer,
                 'useDispersionCorrection': False,
                 'useSwitchingFunction': False},
                {'nonbondedMethod': app.CutoffPeriodic,
                 'nonbondedCutoff': 0.95 * unit.nanometer,
                 'useDispersionCorrection': False,
                 'useSwitchingFunction': False},
                {'nonbondedMethod': app.CutoffPeriodic,
                 'nonbondedCutoff': 0.95 * unit.nanometer,
                 'useDispersionCorrection': True,
                 'useSwitchingFunction': False},
                ]
        
        for ff_setting in ff_settings:
            nbmethod = ff_setting['nonbondedMethod']
            rc = ff_setting['nonbondedCutoff']
            useDispersionCorrection = ff_setting['useDispersionCorrection']
            useSwitchingFunction = ff_setting['useSwitchingFunction']
    
            # construct target state
            H = Hamiltonian(prm1)
            pot = H.createPotential(pdbobj.topology,
                                    nonbondedMethod=nbmethod,
                                    nonbondedCutoff=rc,
                                    useDispersionCorrection=useDispersionCorrection)
            efunc = pot.getPotentialFunc()
            target_energy_func = buildTrajEnergyFunction(efunc,
                                                pot.meta['cov_map'],
                                                rc._value,
                                                pressure=1.0)
            target_state = TargetState(300.0, target_energy_func)

    
            # construct openmm state
            omm_state = OpenMMSampleState('ref',
                                           prm1,
                                           pdb,
                                           pressure=1.0,
                                           useDispersionCorrection=useDispersionCorrection,
                                           useSwitchingFunction=useSwitchingFunction,
                                           nonbondedMethod=nbmethod,
                                           nonbondedCutoff=rc)
    
            # check consistency
            traj = md.load(traj1, top=pdb)[20::4]
    
            ene1 = target_state.calc_energy(traj, H.paramtree)
            ene2 = omm_state.calc_energy(traj)
    
            if nbmethod == app.PME:
                npt.assert_allclose(ene1, ene2, rtol=10**-3.5)
            else:
                npt.assert_allclose(ene1, ene2, rtol=10**-3.5)
    
