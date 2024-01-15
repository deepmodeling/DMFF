import pytest
import jax
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
import numpy as np
import numpy.testing as npt
from dmff.api import Hamiltonian
from dmff.common import nblist


@pytest.mark.parametrize(
    "pdb, prm, value",
    [
        ("./tests/data/10p.pdb", "./tests/data/1_5corrV2.xml", -11184.921239189738),
        ("./tests/data/pBox.pdb", "./tests/data/polyp_amberImp.xml", -13914.34177591779),
    ])
def test_custom_gb_force(pdb, prm, value):
    pdb = app.PDBFile(pdb)
    h = Hamiltonian(prm)
    potential = h.createPotential(
        pdb.topology,
        nonbondedMethod=app.NoCutoff
    )
    pos = jnp.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    box = np.array([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]])
    rc = 6.0
    nbl = nblist.NeighborList(box, rc, potential.meta['cov_map'])
    nbl.allocate(pos)
    pairs = nbl.pairs
    gbE = potential.getPotentialFunc(names=["CustomGBForce"])
    energy = gbE(pos, box, pairs, h.paramset)
    print(energy)
    npt.assert_almost_equal(energy, value, decimal=3)


@pytest.mark.parametrize(
    "pdb, prm, value",
    [
        ("./tests/data/10p.pdb", "./tests/data/1_5corrV2.xml", 59.53033875302844),
    ])
def test_custom_torsion_force(pdb, prm, value):
    pdb = app.PDBFile(pdb)
    h = Hamiltonian(prm)
    potential = h.createPotential(
        pdb.topology,
        nonbondedMethod=app.NoCutoff
    )
    pos = jnp.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    box = np.array([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]])
    rc = 6.0
    nbl = nblist.NeighborList(box, rc, potential.meta['cov_map'])
    nbl.allocate(pos)
    pairs = nbl.pairs
    gbE = potential.getPotentialFunc(names=["CustomTorsionForce"])
    energy = gbE(pos, box, pairs, h.paramset)
    npt.assert_almost_equal(energy, value, decimal=3)


@pytest.mark.parametrize(
    "pdb, prm, value",
    [
        ("./tests/data/10p.pdb", "./tests/data/1_5corrV2.xml", 117.95416362791674),
    ])
def test_custom_1_5bond_force(pdb, prm, value):
    pdb = app.PDBFile(pdb)
    h = Hamiltonian(prm)
    potential = h.createPotential(
        pdb.topology,
        nonbondedMethod=app.NoCutoff
    )
    pos = jnp.asarray(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    box = np.array([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]])
    rc = 6.0
    nbl = nblist.NeighborList(box, rc, potential.meta['cov_map'])
    nbl.allocate(pos)
    pairs = nbl.pairs
    gbE = potential.getPotentialFunc(names=["Custom1_5BondForce"])
    energy = gbE(pos, box, pairs, h.paramset)
    npt.assert_almost_equal(energy, value, decimal=3)