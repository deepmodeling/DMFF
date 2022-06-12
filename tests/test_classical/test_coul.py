import pytest
import jax
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
import numpy as np
import numpy.testing as npt
from dmff import Hamiltonian, NeighborList


class TestCoulomb:

    @pytest.mark.parametrize(
        "pdb, prm, value",
        [("tests/data/lj2.pdb", "tests/data/coul2.xml", 83.72177124023438)])
    def test_coul_force(self, pdb, prm, value):
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        system = h.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        pos = jnp.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        pairs = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
                         dtype=int)
        coulE = h._potentials[0]
        energy = coulE(pos, box, pairs, h.getGenerators()[0].params)
        npt.assert_almost_equal(energy, value, decimal=3)
        
        energy = jax.jit(coulE)(pos, box, pairs, h.getGenerators()[0].params)
        npt.assert_almost_equal(energy, value, decimal=3)

    @pytest.mark.parametrize(
        "pdb, prm, value",
        [("tests/data/lj3.pdb", "tests/data/coul3.xml", -446.82037353515625)])
    def test_coul_large_force(self, pdb, prm, value):
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        system = h.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        pos = jnp.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        pairs = []
        for ii in range(10):
            for jj in range(ii + 1, 10):
                pairs.append((ii, jj))
        pairs = np.array(pairs, dtype=int)
        coulE = h._potentials[0]
        energy = coulE(pos, box, pairs, h.getGenerators()[0].params)
        npt.assert_almost_equal(energy, value, decimal=3)
        
        energy = jax.jit(coulE)(pos, box, pairs, h.getGenerators()[0].params)
        npt.assert_almost_equal(energy, value, decimal=3)

    @pytest.mark.parametrize(
        "pdb, prm, value",
        [("tests/data/lj3.pdb", "tests/data/coul3-res.xml", -446.82037353515625)])
    def test_coul_res_large_force(self, pdb, prm, value):
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        system = h.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        pos = jnp.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        pairs = []
        for ii in range(10):
            for jj in range(ii + 1, 10):
                pairs.append((ii, jj))
        pairs = np.array(pairs, dtype=int)
        coulE = h._potentials[0]
        energy = coulE(pos, box, pairs, h.getGenerators()[0].params)
        npt.assert_almost_equal(energy, value, decimal=3)
        
        energy = jax.jit(coulE)(pos, box, pairs, h.getGenerators()[0].params)
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
        potentials = h.createPotential(
            pdb.topology, 
            nonbondedMethod=app.PME, 
            constraints=app.HBonds, 
            removeCMMotion=False, 
            nonbondedCutoff=rcut * unit.nanometers,
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
        ], dtype=jnp.float64)
        nbList = NeighborList(box, rc=rcut)
        nbList.allocate(positions)
        pairs = nbList.pairs
        func = potentials[-1]
        ene = func(
            positions, 
            box, 
            pairs,
            h.getGenerators()[-1].params
        )
        assert np.allclose(ene, value, atol=1e-2)
