import pytest

import numpy as np
import jax
import jax.numpy as jnp
from rdkit import Chem
import openmm.app as app

from dmff import Hamiltonian, NeighborList

    
@pytest.mark.parametrize(
    "name",
    ["methane", "ethane", "formaldehyde"]
)
def test_smirks(name: str):
    pdb = app.PDBFile(f"tests/data/{name}.pdb")
    h_typing = Hamiltonian(f"tests/data/{name}.xml")
    pot_typing = h_typing.createPotential(pdb.topology)
    cov_map = h_typing.getCovalentMap()
    
    pos = jnp.array(pdb.getPositions(asNumpy=True))
    box = jnp.eye(3, dtype=jnp.float32)
    nblist = NeighborList(box, 1.0, cov_map)
    nblist.allocate(pos)
    pairs = nblist.pairs

    ref_data = {}
    for key, pot in pot_typing.dmff_potentials.items():
        ref_data[key] = pot(pos, box, pairs, h_typing.paramtree)
    
    rdmol = Chem.MolFromMolFile(f"tests/data/{name}.mol", removeHs=False)
    h_smirks = Hamiltonian(f"tests/data/{name}_smirks.xml", noOmmSys=True)
    top = h_smirks.buildTopologyFromMol(rdmol)
    pot_smirks = h_smirks.createPotential(top, rdmol=rdmol)
    
    # print(ref_data)
    
    for key in ref_data.keys():
        value_smirks = pot_smirks.dmff_potentials[key](pos, box, pairs, h_smirks.paramtree)
        assert jnp.allclose(value_smirks, ref_data[key], atol=1e-6), f"{key} does not match"


@pytest.mark.parametrize(
    "name",
    ["chloropyridine"]
)
def test_vsite(name: str):
    rdmol = Chem.MolFromMolFile(f"tests/data/{name}.mol", removeHs=False)
    h_smirks = Hamiltonian(f"tests/data/{name}_vsite.xml", noOmmSys=True)
    top = h_smirks.buildTopologyFromMol(rdmol)
    pot_vsite = h_smirks.createPotential(top, rdmol=rdmol)
    newmol = h_smirks.addVirtualSiteToMol(rdmol, h_smirks.paramtree)
    # Chem.MolToMolFile(newmol, f"tests/data/{name}_vsite.mol")
    rdmol_vsite = Chem.MolFromMolFile(f"tests/data/{name}_vsite.mol", removeHs=False)

    pos_vsite = jnp.array(newmol.GetConformer().GetPositions(), dtype=jnp.float32) / 10
    box = jnp.eye(3, dtype=jnp.float32)
    nblist = NeighborList(box, 1.0, pot_vsite.meta["cov_map"])
    nblist.allocate(pos_vsite)
    pairs_vsite = nblist.pairs

    nbfunc_vsite = jax.value_and_grad(pot_vsite.dmff_potentials['NonbondedForce'], argnums=-1, allow_int=True)
    nbene_vsite, nbene_grad_vsite = nbfunc_vsite(pos_vsite, box, pairs_vsite, h_smirks.paramtree)
    nbene_dbcc = jnp.dot(
        h_smirks.getTopologyMatrix().T,
        nbene_grad_vsite['NonbondedForce']['charge'].reshape(-1, 1)
    )
    # test grad bcc
    assert jnp.allclose(nbene_dbcc, nbene_grad_vsite['NonbondedForce']['bcc'])
    
    # test vsite coordinates
    assert np.allclose(
        newmol.GetConformer().GetPositions(),
        rdmol_vsite.GetConformer().GetPositions(),
        atol=1e-4
    )

    h_typing = Hamiltonian(f"tests/data/{name}.xml", removeHs=False)
    pot_typing = h_typing.createPotential(top)
    pos = jnp.array(rdmol.GetConformer().GetPositions(), dtype=jnp.float32) / 10
    box = jnp.eye(3, dtype=jnp.float32)
    nblist = NeighborList(box, 1.0, pot_typing.meta["cov_map"])
    nblist.allocate(pos)
    pairs = nblist.pairs
    nbfunc = jax.value_and_grad(pot_typing.dmff_potentials['NonbondedForce'], argnums=-1, allow_int=True)
    nbene, nbene_grad = nbfunc(pos, box, pairs, h_typing.paramtree)
    # test energies
    assert np.allclose(nbene, nbene_vsite, atol=1e-6)


@pytest.mark.parametrize(
    "name",
    ["ethane"]
)
def test_bcc(name: str):
    rdmol = Chem.MolFromMolFile(f"tests/data/{name}.mol", removeHs=False)
    h_smirks = Hamiltonian(f"tests/data/{name}_smirks.xml", noOmmSys=True)
    top = h_smirks.buildTopologyFromMol(rdmol)
    h_smirks.createPotential(top, rdmol=rdmol)
    bccchg = jnp.dot(h_smirks.getTopologyMatrix(), h_smirks.paramtree['NonbondedForce']['bcc']).flatten()
    prechg = h_smirks.paramtree['NonbondedForce']['charge']

    h_typing = Hamiltonian(f"tests/data/{name}.xml")
    h_typing.createPotential(top)
    refchg = h_typing.paramtree['NonbondedForce']['charge']

    assert jnp.allclose(bccchg+prechg, refchg, atol=1e-6)

