import openmm.app as app
import openmm.unit as unit
from dmff.api import Hamiltonian
from dmff.api import DMFFTopology
from dmff.api.xmlio import XMLIO
from dmff import NeighborList
import jax.numpy as jnp
import jax
import numpy as np


def test_qeq_energy():
    rc = 0.8
    xml = XMLIO()
    xml.loadXML("tests/data/qeq.xml")
    res = xml.parseResidues()
    charges = [a["charge"] for a in res[0]["particles"]]
    types = [a["type"] for a in res[0]["particles"]]

    pdb = app.PDBFile("tests/data/qeq.pdb")
    dmfftop = DMFFTopology(from_top=pdb.topology)
    pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    pos = jnp.array(pos)
    box = dmfftop.getPeriodicBoxVectors()
    hamilt = Hamiltonian("tests/data/qeq.xml")

    atoms = [a for a in dmfftop.atoms()]
    for na, a in enumerate(atoms):
        a.meta["charge"] = charges[na]
        a.meta["type"] = types[na]

    nblist = NeighborList(box, rc, dmfftop.buildCovMat())
    pairs = nblist.allocate(pos)
    const_list = []
    const_list.append([])
    for ii in range(144):
        const_list[-1].append(ii)

    pot = hamilt.createPotential(dmfftop, nonbondedCutoff=rc*unit.nanometer, nonbondedMethod=app.PME, 
                                ethresh=1e-3, neutral=False, slab=False, constQ=True,pbc_flag=True,part_const=True,
                                has_aux=True
                                )
    efunc = pot.getPotentialFunc()
    n_template = len(const_list)
    q = jnp.array([i.meta['charge'] for i in atoms]) 
    lagmt = jnp.ones(n_template)
    q_0 = jnp.concatenate((q,lagmt))
    aux = {
        "q": jnp.array(q_0),
        }
    energy,aux = efunc(pos, box, pairs, hamilt.paramset.parameters,aux)
    np.testing.assert_almost_equal(energy, -37.84692763, decimal=2)


def test_qeq_energy_2res():
    rc = 0.8
    xml = XMLIO()
    xml.loadXML("tests/data/qeq2.xml")
    res = xml.parseResidues()
    charges = [a["charge"] for a in res[0]["particles"]] + [a["charge"] for a in res[1]["particles"]]
    charges = np.zeros((len(charges),))
    types = [a["type"] for a in res[0]["particles"]] + [a["type"] for a in res[1]["particles"]]

    pdb = app.PDBFile("tests/data/qeq2.pdb")
    top = pdb.topology
    dmfftop = DMFFTopology(from_top=top)
    atoms = [a for a in dmfftop.atoms()]
    pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    pos = jnp.array(pos)
    box = dmfftop.getPeriodicBoxVectors()
    hamilt = Hamiltonian("tests/data/qeq2.xml")

    atoms = [a for a in dmfftop.atoms()]
    for na, a in enumerate(atoms):
        a.meta["charge"] = charges[na]
        a.meta["type"] = types[na]

    nblist = NeighborList(box, rc, dmfftop.buildCovMat())
    pairs = nblist.allocate(pos)

    const_list = []
    const_list.append([])
    for ii in range(144):
        const_list[-1].append(ii)
    const_list.append([])
    for ii in range(144, 147):
        const_list[-1].append(ii)
    const_val = [0.0, 0.0]

    pot = hamilt.createPotential(dmfftop, nonbondedCutoff=rc*unit.nanometer, nonbondedMethod=app.PME, 
                                ethresh=1e-3, neutral=False, slab=False, constQ=True,
                                const_list=const_list, const_vals=const_val,
                                has_aux=True
                                )
    efunc = pot.getPotentialFunc()
   
    n_template = len(const_list)
    q = jnp.array([i.meta['charge'] for i in atoms]) 
    lagmt = jnp.ones(n_template)
    q_0 = jnp.concatenate((q,lagmt))
    aux = {
        "q": jnp.array(q_0),
       # "lagmt": jnp.array([1.0, 1.0])
    }
    energy, aux0 = efunc(pos, box, pairs, hamilt.paramset.parameters, aux=aux)
    print(aux['q'])
    np.testing.assert_almost_equal(energy, 4817.295171, decimal=0)

    grad = jax.grad(efunc, argnums=0, has_aux=True)
    gradient, aux = grad(pos, box, pairs, hamilt.paramset.parameters, aux=aux)
    print(gradient)


def _test_qeq_energy_2res_jit():
    rc = 0.8
    xml = XMLIO()
    xml.loadXML("tests/data/qeq2.xml")
    res = xml.parseResidues()
    charges = [a["charge"] for a in res[0]["particles"]] + [a["charge"] for a in res[1]["particles"]]
    charges = np.zeros((len(charges),))
    types = [a["type"] for a in res[0]["particles"]] + [a["type"] for a in res[1]["particles"]]

    pdb = app.PDBFile("tests/data/qeq2.pdb")
    top = pdb.topology
    dmfftop = DMFFTopology(from_top=top)
    atoms = [a for a in dmfftop.atoms()]
    pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    pos = jnp.array(pos)
    box = dmfftop.getPeriodicBoxVectors()
    hamilt = Hamiltonian("tests/data/qeq2.xml")

    atoms = [a for a in dmfftop.atoms()]
    for na, a in enumerate(atoms):
        a.meta["charge"] = charges[na]
        a.meta["type"] = types[na]

    nblist = NeighborList(box, rc, dmfftop.buildCovMat())
    pairs = nblist.allocate(pos)

    const_list = []
    const_list.append([])
    for ii in range(144):
        const_list[-1].append(ii)
    const_list.append([])
    for ii in range(144, 147):
        const_list[-1].append(ii)
    const_val = [0.0, 0.0]

    pot = hamilt.createPotential(dmfftop, nonbondedCutoff=rc*unit.nanometer, nonbondedMethod=app.PME, 
                                ethresh=1e-3, neutral=False, slab=False, constQ=True,
                                const_list=const_list, const_vals=const_val,
                                has_aux=True
                                )
    efunc = jax.jit(pot.getPotentialFunc())
    grad = jax.jit(jax.grad(efunc, argnums=0, has_aux=True))
    q = jnp.array(charges)
    lagmt = jnp.array([1.0, 1.0])
    q_0 = jnp.concatenate((q,lagmt))
    aux = {
        "q": jnp.array(q_0), 
    }
    print("Start computing energy and force.")
    energy, aux = efunc(pos, box, pairs, hamilt.paramset.parameters, aux=aux)
    print(aux)
    np.testing.assert_almost_equal(energy, 4817.295171, decimal=0)

    grad = jax.grad(efunc, argnums=0, has_aux=True)
    gradient, aux = grad(pos, box, pairs, hamilt.paramset.parameters, aux=aux)
    print(gradient)
