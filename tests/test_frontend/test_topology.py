import openmm.app as app
import openmm.unit as unit
from dmff.api.topology import DMFFTopology
import json
import pytest


@pytest.mark.parametrize('pdb,top,reference', [
    ("tests/data/lig.pdb", "tests/data/lig-top.xml", "tests/data/lig-structure.json")
])
def test_find_substructures(pdb, top, reference):
    app.Topology.loadBondDefinitions(top)
    omm_top = app.PDBFile(pdb).topology
    topData = DMFFTopology(omm_top)
    with open(reference, "r") as f:
        refdata = json.load(f)
    assert len(topData.bonds) == refdata["nbonds"]
    assert len(topData.angles) == refdata["nangles"]
    assert len(topData.propers) == refdata["nprops"]
    assert len(topData.impropers) == refdata["nimprs"]
