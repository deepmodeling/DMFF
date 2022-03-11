import jax.numpy as jnp
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
import numpy.testing as npt
from dmff.api import Hamiltonian
import pytest


class TestClassical:
    @pytest.mark.parametrize(
        "pdb, prm, value",
        [
            ("data/bond1.pdb", "data/bond1.xml", 1389.162109375),
            #("data/bond2.pdb", "data/bond2.xml", 100.00),
        ])
    def test_harmonic_bond_force(self, pdb, prm, value):
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        system = h.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        pairs = np.array([[]], dtype=int)
        bondE = h._potentials[0]
        energy = bondE(pos, box, pairs, h.getGenerators()[0].params)
        npt.assert_almost_equal(energy, value, decimal=3)

    @pytest.mark.parametrize(
        "pdb, prm, value",
        [
            ("data/angle1.pdb", "data/angle1.xml", 315.88775634765625),
            #("data/angle2.pdb", "data/angle2.xml", 100.00),
        ])
    def test_harmonic_angle_force(self, pdb, prm, value):
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        system = h.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        pairs = np.array([[]], dtype=int)
        bondE = h._potentials[0]
        energy = bondE(pos, box, pairs, h.getGenerators()[0].params)
        npt.assert_almost_equal(energy, value, decimal=3)

    @pytest.mark.parametrize(
        "pdb, prm, value",
        [
            ("data/proper1.pdb", "data/proper1.xml", 8.368000030517578),
            ("data/proper2.pdb", "data/proper2.xml", 20.931230545),
            ("data/impr1.pdb", "data/impr1.xml", 2.9460556507110596),
            ("data/proper1.pdb", "data/wild1.xml", 8.368000030517578),
            ("data/impr1.pdb", "data/wild2.xml", 2.9460556507110596),
            #("data/tor2.pdb", "data/tor2.xml", 100.00)
        ])
    def test_periodic_torsion_force(self, pdb, prm, value):
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        system = h.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        pairs = np.array([[]], dtype=int)
        bondE = h._potentials[0]
        energy = bondE(pos, box, pairs, h.getGenerators()[0].params)
        npt.assert_almost_equal(energy, value, decimal=3)


    @pytest.mark.parametrize(
        "pdb, prm, value",
        [
            ('data/lj1.pdb', 'data/lj1.xml', 123)
        ]
    )
    def test_lj_force(self, pdb, prm, value):
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        system = h.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        npt.assert_allclose(pos, jnp.array([[0, 0.1, 0], [0.1, 0, 0.1]]))
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        pairs = np.array([[0, 1]], dtype=int)
        ljE = h._potentials[0]
        # energy = ljE(pos, box, pairs, h.getGenerators()[0].params)
        
