from dmff.api.vstools import VSiteTool
from dmff.api.topology import TopologyData
from dmff.api.hamiltonian import Hamiltonian
import openmm.app as app
import openmm.unit as unit


def test_template_vs_and_template_atype():
    print("TEMPLATE VS & TEMPLATE ATYPE")
    pdb = app.PDBFile("tests/data/template_vsite.pdb")
    top = pdb.topology
    topdata = TopologyData(top)

    vstool = VSiteTool(template_files=["tests/data/template_and_vsite.xml"])
    topdata = vstool.addVSite(topdata)

    for nres in range(topdata.topology.getNumResidues()):
        topdata.setOperatorToResidue(nres, "template")

    hamilt = Hamiltonian("tests/data/template_and_vsite.xml")
    hamilt.prepTopData(topdata)
    for ameta in topdata.atom_meta:
        print(ameta)


def test_smarts_vs_and_template_atype():
    print("SMARTS VS & TEMPLATE ATYPE")
    pdb = app.PDBFile("tests/data/template_vsite.pdb")
    top = pdb.topology
    topdata = TopologyData(top)

    vstool = VSiteTool(smarts_files=["tests/data/smarts_and_vsite.xml"])
    topdata = vstool.addVSite(topdata)

    for nres in range(topdata.topology.getNumResidues()):
        topdata.setOperatorToResidue(nres, "template")

    hamilt = Hamiltonian("tests/data/template_and_vsite.xml")
    hamilt.prepTopData(topdata)
    for ameta in topdata.atom_meta:
        print(ameta)

def test_smarts_vs_and_smarts_atype():
    print("SMARTS VS & SMARTS ATYPE")
    pdb = app.PDBFile("tests/data/template_vsite.pdb")
    top = pdb.topology
    topdata = TopologyData(top)

    vstool = VSiteTool(smarts_files=["tests/data/smarts_and_vsite.xml"])
    topdata = vstool.addVSite(topdata)

    for nres in range(topdata.topology.getNumResidues()):
        topdata.setOperatorToResidue(nres, "smarts")

    hamilt = Hamiltonian("tests/data/smarts_test1.xml")
    hamilt.prepTopData(topdata)
    for ameta in topdata.atom_meta:
        print(ameta)

def test_template_vs_and_smarts_atype():
    print("TEMPLATE VS & SMARTS ATYPE")
    pdb = app.PDBFile("tests/data/template_vsite.pdb")
    top = pdb.topology
    topdata = TopologyData(top)

    vstool = VSiteTool(template_files=["tests/data/template_and_vsite.xml"])
    topdata = vstool.addVSite(topdata)

    for nres in range(topdata.topology.getNumResidues()):
        topdata.setOperatorToResidue(nres, "smarts")

    hamilt = Hamiltonian("tests/data/smarts_test1.xml")
    hamilt.prepTopData(topdata)
    for ameta in topdata.atom_meta:
        print(ameta)

if __name__ == "__main__":
    test_template_vs_and_template_atype()
    test_smarts_vs_and_template_atype()
    test_smarts_vs_and_smarts_atype()
    test_template_vs_and_smarts_atype()