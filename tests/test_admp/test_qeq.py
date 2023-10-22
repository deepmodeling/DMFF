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
                                ethresh=5e-4, neutral=True, slab=False, constQ=True
                                )
    efunc = pot.getPotentialFunc()
    energy = efunc(pos, box, pairs, hamilt.paramset.parameters)
    np.testing.assert_almost_equal(energy, -37.84692763, decimal=3)