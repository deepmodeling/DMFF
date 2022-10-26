from dmff.mbar import MBAREstimator, Sample, BaseState
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
from tqdm import tqdm


class OMMNPTState(BaseState):
    def __init__(self, temperature, name, parameter, topology):
        super(OMMNPTState, self).__init__(temperature, name)
        # create a context
        pdb = app.PDBFile(topology)
        ff = app.ForceField(parameter)
        system = ff.createSystem(pdb.topology,
                                 nonbondedMethod=app.PME,
                                 nonbondedCutoff=0.9 * unit.nanometer,
                                 constraints=None,
                                 rigidWater=False)

        for force in system.getForces():
            if isinstance(force, mm.NonbondedForce):
                force.setUseDispersionCorrection(False)
                force.setUseSwitchingFunction(False)

        integ = mm.LangevinIntegrator(0 * unit.kelvin, 5 / unit.picosecond,
                                      1.0 * unit.femtosecond)
        self.ctx = mm.Context(system, integ)

    def calc_energy_frame(self, frame):
        self.ctx.setPositions(frame.openmm_positions(0))
        self.ctx.setPeriodicBoxVectors(*frame.openmm_boxes(0))
        state = self.ctx.getState(getEnergy=True)
        vol = frame.unitcell_volumes[0]
        ener = state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole) + 0.06023 * vol
        return ener


class TestMBAR:
    @pytest.mark.parametrize("pdb, prm1, traj, prm2", [(
        "tests/data/waterbox.pdb",
        "tests/data/water1.xml",
        "tests/data/w1_npt.dcd",
        "tests/data/water3.xml"
    )])
    def test_mbar_free_energy_mono(self, pdb, prm1, traj, prm2):
        pdbobj = app.PDBFile(pdb)

        # prepare MBAR estimator
        traj = md.load(traj, top=pdb)[20:]
        ref_state = OMMNPTState(300, "ref", prm1, pdb)
        target_state = OMMNPTState(300, "target", prm2, pdb)
        sample = Sample(traj, "ref")

        mbar = MBAREstimator()
        mbar.add_state(ref_state)
        mbar.add_sample(sample)
        mbar.optimize_mbar()

        # rerun DMFF energies
        h = Hamiltonian(prm2)
        pot = h.createPotential(pdbobj.topology,
                                nonbondedMethod=app.PME,
                                nonbondedCutoff=0.9 * unit.nanometer)
        efunc = pot.getPotentialFunc()
        nbgen = None
        for gen in h.getGenerators():
            if isinstance(gen, dmff.generators.NonbondedJaxGenerator):
                nbgen = gen

        elist = []
        for frame in tqdm(mbar._full_samples):
            aa, bb, cc = frame.openmm_boxes(0).value_in_unit(unit.nanometer)
            box = jnp.array([[aa[0], aa[1], aa[2]], [bb[0], bb[1], bb[2]],
                             [cc[0], cc[1], cc[2]]])
            vol = aa[0] * bb[1] * cc[2]
            positions = jnp.array(frame.xyz[0, :, :])
            nbobj = NeighborListFreud(box, 0.9, nbgen.covalent_map)
            nbobj.capacity_multiplier = 1
            pairs = nbobj.allocate(positions)
            energy = efunc(positions, box, pairs, h.paramtree) + 0.06023 * vol
            elist.append(energy)

        # estimate free energy
        beta = 1. / 300. / 8.314 * 1000
        ulist = jnp.concatenate([beta * e.reshape((1, )) for e in elist])
        freeE = mbar.estimate_free_energy(ulist)
        
        # calc reference using PyMBAR
        umat_ref = np.zeros((2, ulist.shape[0]))
        umat_ref[0,:] = mbar._umat[0,:]
        umat_ref[1,:] = ulist[:]
        nk = np.array([traj.n_frames, 0])
        mbar_ref = MBAR(umat_ref, nk, initialize="BAR")
        npt.assert_almost_equal(freeE, mbar_ref.f_k[-1], decimal=3)


    @pytest.mark.parametrize("pdb, prm1, traj1, prm2, traj2, prm3", [(
        "tests/data/waterbox.pdb",
        "tests/data/water1.xml",
        "tests/data/w1_npt.dcd",
        "tests/data/water2.xml",
        "tests/data/w2_npt.dcd",
        "tests/data/water3.xml"
    )])
    def test_mbar_free_energy_multi(self, pdb, prm1, traj1, prm2, traj2, prm3):
        pdbobj = app.PDBFile(pdb)

        # prepare MBAR estimator
        traj1 = md.load(traj1, top=pdb)[20:]
        traj2 = md.load(traj2, top=pdb)[20:]
        ref_state1 = OMMNPTState(300, "ref1", prm1, pdb)
        ref_state2 = OMMNPTState(300, "ref2", prm2, pdb)
        sample1 = Sample(traj1, "ref1")
        sample2 = Sample(traj2, "ref2")

        mbar = MBAREstimator()
        mbar.add_state(ref_state1)
        mbar.add_sample(sample1)
        mbar.add_state(ref_state2)
        mbar.add_sample(sample2)
        mbar.optimize_mbar()

        # rerun DMFF energies
        h = Hamiltonian(prm3)
        pot = h.createPotential(pdbobj.topology,
                                nonbondedMethod=app.PME,
                                nonbondedCutoff=0.9 * unit.nanometer)
        efunc = pot.getPotentialFunc()
        nbgen = None
        for gen in h.getGenerators():
            if isinstance(gen, dmff.generators.NonbondedJaxGenerator):
                nbgen = gen

        elist = []
        for frame in tqdm(mbar._full_samples):
            aa, bb, cc = frame.openmm_boxes(0).value_in_unit(unit.nanometer)
            box = jnp.array([[aa[0], aa[1], aa[2]], [bb[0], bb[1], bb[2]],
                             [cc[0], cc[1], cc[2]]])
            vol = aa[0] * bb[1] * cc[2]
            positions = jnp.array(frame.xyz[0, :, :])
            nbobj = NeighborListFreud(box, 0.9, nbgen.covalent_map)
            nbobj.capacity_multiplier = 1
            pairs = nbobj.allocate(positions)
            energy = efunc(positions, box, pairs, h.paramtree) + 0.06023 * vol
            elist.append(energy)

        # estimate free energy
        beta = 1. / 300. / 8.314 * 1000
        ulist = jnp.concatenate([beta * e.reshape((1, )) for e in elist])
        freeE = mbar.estimate_free_energy(ulist)

        # calc reference using PyMBAR
        umat_ref = np.zeros((3, ulist.shape[0]))
        umat_ref[0,:] = mbar._umat[0,:]
        umat_ref[1,:] = mbar._umat[1,:]
        umat_ref[2,:] = ulist[:]
        nk = np.array([traj1.n_frames, traj2.n_frames, 0])
        mbar_ref = MBAR(umat_ref, nk, initialize="BAR")
        npt.assert_almost_equal(freeE, mbar_ref.f_k[-1], decimal=3)


    @pytest.mark.parametrize("pdb, prm1, traj1, prm2, traj2, prm3", [(
        "tests/data/waterbox.pdb",
        "tests/data/water1.xml",
        "tests/data/w1_npt.dcd",
        "tests/data/water2.xml",
        "tests/data/w2_npt.dcd",
        "tests/data/water3.xml"
    )])
    def test_mbar_weight(self, pdb, prm1, traj1, prm2, traj2, prm3):
        pdbobj = app.PDBFile(pdb)

        # prepare MBAR estimator
        traj1 = md.load(traj1, top=pdb)[20:]
        traj2 = md.load(traj2, top=pdb)[20:]
        ref_state1 = OMMNPTState(300, "ref1", prm1, pdb)
        ref_state2 = OMMNPTState(300, "ref2", prm2, pdb)
        sample1 = Sample(traj1, "ref1")
        sample2 = Sample(traj2, "ref2")

        mbar = MBAREstimator()
        mbar.add_state(ref_state1)
        mbar.add_sample(sample1)
        mbar.add_state(ref_state2)
        mbar.add_sample(sample2)
        mbar.optimize_mbar()

        # rerun DMFF energies
        h = Hamiltonian(prm3)
        pot = h.createPotential(pdbobj.topology,
                                nonbondedMethod=app.PME,
                                nonbondedCutoff=0.9 * unit.nanometer)
        efunc = pot.getPotentialFunc()
        nbgen = None
        for gen in h.getGenerators():
            if isinstance(gen, dmff.generators.NonbondedJaxGenerator):
                nbgen = gen

        elist = []
        for frame in tqdm(mbar._full_samples):
            aa, bb, cc = frame.openmm_boxes(0).value_in_unit(unit.nanometer)
            box = jnp.array([[aa[0], aa[1], aa[2]], [bb[0], bb[1], bb[2]],
                             [cc[0], cc[1], cc[2]]])
            vol = aa[0] * bb[1] * cc[2]
            positions = jnp.array(frame.xyz[0, :, :])
            nbobj = NeighborListFreud(box, 0.9, nbgen.covalent_map)
            nbobj.capacity_multiplier = 1
            pairs = nbobj.allocate(positions)
            energy = efunc(positions, box, pairs, h.paramtree) + 0.06023 * vol
            elist.append(energy)

        # estimate weight
        beta = 1. / 300. / 8.314 * 1000
        ulist = jnp.concatenate([beta * e.reshape((1, )) for e in elist])
        weight = mbar.estimate_weight(ulist)

        # calc reference using PyMBAR
        umat_ref = np.zeros((3, ulist.shape[0]))
        umat_ref[0,:] = mbar._umat[0,:]
        umat_ref[1,:] = mbar._umat[1,:]
        umat_ref[2,:] = ulist[:]
        nk = np.array([traj1.n_frames, traj2.n_frames, 0])
        mbar_ref = MBAR(umat_ref, nk, initialize="BAR")
        weight_ref = mbar_ref.W_nk.T[-1,:]
        rmse = np.sqrt(np.power(weight - weight_ref, 2).mean())
        npt.assert_almost_equal(rmse, 0.0, decimal=3)