import pytest
import jax
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
import numpy as np
import numpy.testing as npt
from dmff import Hamiltonian, NeighborList
from dmff.api import DMFFTopology


class TestCoulomb:

    @pytest.mark.parametrize(
        "pdb, prm, value",
        [("tests/data/lj2.pdb", "tests/data/coul2.xml", 83.72177124023438)])
    def test_coul_force(self, pdb, prm, value):
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        potential = h.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        pos = jnp.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        rc = 4
        nblist = NeighborList(box, rc, potential.meta["cov_map"])
        nblist.allocate(pos)
        pairs = nblist.pairs
        coulE = potential.getPotentialFunc(names=["NonbondedForce"])
        energy = coulE(pos, box, pairs, h.paramset)
        npt.assert_almost_equal(energy, value, decimal=3)
        
        energy = jax.jit(coulE)(pos, box, pairs, h.paramset.parameters)
        npt.assert_almost_equal(energy, value, decimal=3)

    @pytest.mark.parametrize(
        "pdb, prm, value",
        [("tests/data/lj3.pdb", "tests/data/coul3.xml", -446.82037353515625)])
    def test_coul_large_force(self, pdb, prm, value):
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        potential = h.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        pos = jnp.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        rc = 4
        
        nblist = NeighborList(box, rc, potential.meta["cov_map"])
        nblist.allocate(pos)
        pairs = nblist.pairs
        coulE = potential.getPotentialFunc()
        energy = coulE(pos, box, pairs, h.paramset)
        npt.assert_almost_equal(energy, value, decimal=3)
        
        energy = jax.jit(coulE)(pos, box, pairs, h.paramset)
        npt.assert_almost_equal(energy, value, decimal=3)

    @pytest.mark.parametrize(
        "pdb, prm, value",
        [("tests/data/lj3.pdb", "tests/data/coul3-res.xml", -446.82037353515625)])
    def test_coul_res_large_force(self, pdb, prm, value):
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        potential = h.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        pos = jnp.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        rc = 4
        nblist = NeighborList(box, rc, potential.meta["cov_map"])
        nblist.allocate(pos)
        pairs = nblist.pairs
        coulE = potential.getPotentialFunc()
        energy = coulE(pos, box, pairs, h.paramset.parameters)
        npt.assert_almost_equal(energy, value, decimal=3)
        
        energy = jax.jit(coulE)(pos, box, pairs, h.paramset.parameters)
        npt.assert_almost_equal(energy, value, decimal=3)
    
    @pytest.mark.parametrize(
        "pdb, prm, value",
        [
            (
                "tests/data/methane_water.pdb", 
                "tests/data/methane_water_coul.xml",
                -3.540451
            )
        ]
    )
    def test_coul_pme(self, pdb, prm, value):
        rcut = 0.5 # nanometers
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        potential = h.createPotential(
            pdb.topology, 
            nonbondedMethod=app.PME, 
            nonbondedCutoff=rcut * unit.nanometers,
            constraints=app.HBonds, 
            removeCMMotion=False, 
            rigidWater=False,
            useDispersionCorrection=False,
            PmeCoeffMethod="gromacs",
            PmeSpacing=0.10
        )
        positions = jnp.array(
            pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        )
        box = jnp.array([
            [ 1.20,  0.00,  0.00],
            [ 0.00,  1.20,  0.00],
            [ 0.00,  0.00,  1.20]
        ])

        nbList = NeighborList(box, rcut, potential.meta["cov_map"])
        nbList.allocate(positions)
        pairs = nbList.pairs
        func = potential.getPotentialFunc(names=["NonbondedForce"])
        #func = potential.dmff_potentials["NonbondedForce"]
        ene = func(
            positions, 
            box, 
            pairs,
            h.paramset.parameters
        )
        assert np.allclose(ene, value, atol=1e-2)