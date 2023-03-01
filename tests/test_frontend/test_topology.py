import openmm.app as app
import openmm.unit as unit
from dmff_new.topology import TopologyData
import json
import pytest


@pytest.mark.parametrize('pdb,top,reference', [
    ("test/data/lig.pdb", "test/data/lig-top.xml", "test/data/lig-structure.json")
])
def test_find_substructures(pdb, top, reference):
    app.Topology.loadBondDefinitions(top)
    omm_top = app.PDBFile(pdb).topology
    topData = TopologyData(omm_top)
    with open(reference, "r") as f:
        refdata = json.load(f)
    assert len(topData.bonds) == refdata["nbonds"]
    assert len(topData.angles) == refdata["nangles"]
    assert len(topData.propers) == refdata["nprops"]
    assert len(topData.impropers) == refdata["nimprs"]
