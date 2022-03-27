import jax.numpy as jnp
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
import numpy.testing as npt
from dmff.api import Hamiltonian
import dmff.api as api
import pytest

from dmff.classical.inter import LennardJonesForce


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
        [("data/lj2.pdb", "data/lj2.xml", -1.85001802444458)])
    def test_lj_force(self, pdb, prm, value):
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        system = h.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        pairs = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
                         dtype=int)
        ljE = h._potentials[0]
        energy = ljE(pos, box, pairs, h.getGenerators()[0].params)
        npt.assert_almost_equal(energy, value, decimal=3)

    @pytest.mark.parametrize(
        "pdb, prm, value",
        [("data/lj3.pdb", "data/lj3.xml", -2.001220464706421)])
    def test_lj_large_force(self, pdb, prm, value):
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        system = h.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        pairs = []
        for ii in range(10):
            for jj in range(ii + 1, 10):
                pairs.append((ii, jj))
        pairs = np.array(pairs, dtype=int)
        ljE = h._potentials[0]
        energy = ljE(pos, box, pairs, h.getGenerators()[0].params)
        npt.assert_almost_equal(energy, value, decimal=3)

    @pytest.mark.parametrize(
        "pdb, prm, value",
        [("data/lj2.pdb", "data/coul2.xml", 83.72177124023438)])
    def test_coul_force(self, pdb, prm, value):
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        system = h.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        pairs = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
                         dtype=int)
        coulE = h._potentials[0]
        energy = coulE(pos, box, pairs, h.getGenerators()[0].params)
        npt.assert_almost_equal(energy, value, decimal=3)

    @pytest.mark.parametrize(
        "pdb, prm, value",
        [("data/lj3.pdb", "data/coul3.xml", -446.82037353515625)])
    def test_coul_large_force(self, pdb, prm, value):
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        system = h.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        pairs = []
        for ii in range(10):
            for jj in range(ii + 1, 10):
                pairs.append((ii, jj))
        pairs = np.array(pairs, dtype=int)
        coulE = h._potentials[0]
        energy = coulE(pos, box, pairs, h.getGenerators()[0].params)
        npt.assert_almost_equal(energy, value, decimal=3)

    @pytest.mark.parametrize(
        "pdb, prm, value",
        [("data/lj3.pdb", "data/coul3-res.xml", -446.82037353515625)])
    def test_coul_res_large_force(self, pdb, prm, value):
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        system = h.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        pairs = []
        for ii in range(10):
            for jj in range(ii + 1, 10):
                pairs.append((ii, jj))
        pairs = np.array(pairs, dtype=int)
        coulE = h._potentials[0]
        energy = coulE(pos, box, pairs, h.getGenerators()[0].params)
        npt.assert_almost_equal(energy, value, decimal=3)

    @pytest.mark.parametrize("pdb, prm, values", [
        ("data/lig.pdb", ["data/gaff-2.11.xml", "data/lig-prm-lj.xml"], [
            174.16702270507812, 99.81585693359375, 99.0631103515625,
            22.778038024902344
        ]),
        #("data/lig.pdb", ["data/gaff-2.11.xml", "data/lig-prm.xml"], []),
    ])
    def test_gaff2_force(self, pdb, prm, values):
        app.Topology.loadBondDefinitions("data/lig-top.xml")
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(*prm)
        system = h.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        pairs = []
        for ii in range(pos.shape[0]):
            for jj in range(ii + 1, pos.shape[0]):
                pairs.append((ii, jj))
        pairs = np.array(pairs, dtype=int)

        generators = h.getGenerators()
        be_ref, ae_ref, tore_ref, lj_ref = values
        for ne, energy in enumerate(h._potentials):
            energy = energy(pos, box, pairs, h.getGenerators()[ne].params)
            npt.assert_almost_equal(energy, values[ne], decimal=3)