import pytest
import jax
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
import numpy as np
import numpy.testing as npt
from dmff.api import Hamiltonian


class TestGaff2:

    @pytest.mark.parametrize(
        "pdb, prm, value",
        [("tests/data/lig.pdb", "tests/data/lig-prm-single-lj.xml", 22.77804946899414)])
    def test_gaff2_lj_force(self, pdb, prm, value):
        app.Topology.loadBondDefinitions("tests/data/lig-top.xml")
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(prm)
        system = h.createPotential(pdb.topology,
                                   nonbondedMethod=app.NoCutoff,
                                   constraints=None,
                                   removeCMMotion=False)
        pos = jnp.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        box = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        pairs = []
        for ii in range(pos.shape[0]):
            for jj in range(ii + 1, pos.shape[0]):
                pairs.append((ii, jj))
        pairs = jnp.array(pairs, dtype=int)
        ljE = h._potentials[0]
        energy = ljE(pos, box, pairs, h.getGenerators()[0].params)
        npt.assert_almost_equal(energy, value, decimal=3)
        
        energy = jax.jit(ljE)(pos, box, pairs, h.getGenerators()[0].params)
        npt.assert_almost_equal(energy, value, decimal=3)

    @pytest.mark.parametrize(
        "pdb, prm, values", 
        [
            (
                "tests/data/lig.pdb", 
                ["tests/data/gaff-2.11.xml", "tests/data/lig-prm-lj.xml"], 
                [
                    174.16702270507812, 99.81585693359375, 
                    99.0631103515625, 22.778038024902344
                ]
            ),
        ]
    )
    def test_gaff2_force(self, pdb, prm, values):
        app.Topology.loadBondDefinitions("tests/data/lig-top.xml")
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(*prm)
        system = h.createPotential(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            removeCMMotion=False
        )
        pos = jnp.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        box = np.array([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]])
        pairs = []
        for ii in range(pos.shape[0]):
            for jj in range(ii + 1, pos.shape[0]):
                pairs.append((ii, jj))
        pairs = np.array(pairs, dtype=int)
        for ne, energy in enumerate(h._potentials):
            E = energy(pos, box, pairs, h.getGenerators()[ne].params)
            npt.assert_almost_equal(E, values[ne], decimal=3)
            
            E = jax.jit(energy)(pos, box, pairs, h.getGenerators()[ne].params)
            npt.assert_almost_equal(E, values[ne], decimal=3)

    @pytest.mark.parametrize(
        "pdb, prm, values", 
        [
            (
                "tests/data/lig.pdb", 
                ["tests/data/gaff-2.11.xml", "tests/data/lig-prm-lj.xml"], 
                [
                    174.16702270507812, 99.81585693359375, 
                    99.0631103515625, 22.778038024902344
                ]
            ),
        ]
    )
    def test_gaff2_total(self, pdb, prm, values):
        app.Topology.loadBondDefinitions("tests/data/lig-top.xml")
        pdb = app.PDBFile(pdb)
        h = Hamiltonian(*prm)
        system = h.createPotential(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            removeCMMotion=False
        )
        pos = jnp.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        box = np.array([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]])
        pairs = []
        for ii in range(pos.shape[0]):
            for jj in range(ii + 1, pos.shape[0]):
                pairs.append((ii, jj))
        pairs = np.array(pairs, dtype=int)
        efunc = h.getPotentialFunc()
        params = h.getParameters()
        Eref = sum(values)
        Ecalc = efunc(pos, box, pairs, params)
        npt.assert_almost_equal(Ecalc, Eref, decimal=3)