from dmff.api.vstools import VSiteTool
from dmff.api.topology import TopologyData
from dmff.api.xmlio import XMLIO
from dmff.api.hamiltonian import Hamiltonian
import openmm.app as app
import openmm.unit as unit


def test_template():
    pdb = app.PDBFile("tests/data/template_vsite.pdb")
    top = pdb.topology
    topdata = TopologyData(top)

    vstool = VSiteTool(template_files=["tests/data/template_and_vsite.xml"])
    topdata = vstool.addVSite(topdata)

    for nres in range(topdata.topology.getNumResidues()):
        topdata.setOperatorToResidue(nres, "template")

    hamilt = Hamiltonian("tests/data/template_and_vsite.xml")
    hamilt.operate(topdata)

def test_smarts():
