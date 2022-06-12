import pytest
import jax
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
import numpy as np
import numpy.testing as npt
from dmff.api import Hamiltonian


class TestBonded:

    @pytest.mark.parametrize(
        "pdb, prm, value",
        [
            ("tests/data/bond1.pdb", "tests/data/bond1.xml", 1389.162109375),
            #("tests/data/bond2.pdb", "tests/data/bond2.xml", 100.00),
        ])
    def test_harmonic_bond_force(self, pdb, prm, value):
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        system = h.createPotential(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            removeCMMotion=False
        )
        pos = jnp.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        pairs = np.array([[]], dtype=int)
        bondE = h._potentials[0]
        energy = bondE(pos, box, pairs, h.getGenerators()[0].params)
        npt.assert_almost_equal(energy, value, decimal=3)
        
        energy = jax.jit(bondE)(pos, box, pairs, h.getGenerators()[0].params)
        npt.assert_almost_equal(energy, value, decimal=3)

    @pytest.mark.parametrize(
        "pdb, prm, value",
        [
            ("tests/data/proper1.pdb", "tests/data/proper1.xml", 8.368000030517578),
            ("tests/data/proper2.pdb", "tests/data/proper2.xml", 20.931230545),
            ("tests/data/impr1.pdb", "tests/data/impr1.xml", 2.9460556507110596),
            ("tests/data/proper1.pdb", "tests/data/wild1.xml", 8.368000030517578),
            ("tests/data/impr1.pdb", "tests/data/wild2.xml", 2.9460556507110596),
            #("tests/data/tor2.pdb", "tests/data/tor2.xml", 100.00)
        ])
    def test_periodic_torsion_force(self, pdb, prm, value):
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        system = h.createPotential(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            removeCMMotion=False
        )
        pos = jnp.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        pairs = np.array([[]], dtype=int)
        bondE = h._potentials[0]
        energy = bondE(pos, box, pairs, h.getGenerators()[0].params)
        npt.assert_almost_equal(energy, value, decimal=3)

        energy = jax.jit(bondE)(pos, box, pairs, h.getGenerators()[0].params)
        npt.assert_almost_equal(energy, value, decimal=3)