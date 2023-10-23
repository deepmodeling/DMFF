import openmm.app as app
import openmm.unit as unit
from dmff.api import Hamiltonian
from dmff.api import DMFFTopology
from dmff.api.xmlio import XMLIO
from dmff import NeighborList
import jax.numpy as jnp
import numpy as np


def test_qeq_energy():
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

    nblist = NeighborList(box, 0.6, dmfftop.buildCovMat())
    pairs = nblist.allocate(pos)

    pot = hamilt.createPotential(dmfftop, nonbondedCutoff=0.6*unit.nanometer, nonbondedMethod=app.PME, 
                                ethresh=1e-3, neutral=True, slab=True, constQ=True
                                )
    efunc = pot.getPotentialFunc()
    energy = efunc(pos, box, pairs, hamilt.paramset.parameters)
    np.testing.assert_almost_equal(energy, -37.84692763, decimal=3)


def test_qeq_energy2():
    rc = 0.6
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
    print(pairs)

    const_list = []
    const_list.append([])
    for ii in range(144):
        const_list[-1].append(ii)
    const_list.append([])
    for ii in range(144, 147):
        const_list[-1].append(ii)
    const_val = [0.0, 0.0]

    pot = hamilt.createPotential(dmfftop, nonbondedCutoff=rc*unit.nanometer, nonbondedMethod=app.PME, 
                                ethresh=1e-3, neutral=True, slab=True, constQ=True,
                                const_list=const_list, const_vals=const_val,
                                has_aux=True
                                )
    efunc = pot.getPotentialFunc()
    aux = {
        "q": jnp.array(charges),
        "lagmt": jnp.array([1.0, 1.0])
    }
    energy, aux = efunc(pos, box, pairs, hamilt.paramset.parameters, aux=aux)
    print(aux)
    np.testing.assert_almost_equal(energy, 4817.295171, decimal=3)