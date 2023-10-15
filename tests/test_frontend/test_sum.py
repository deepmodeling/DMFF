from dmff.api import DMFFTopology
from dmff.api.hamiltonian import Hamiltonian
from dmff.operators.smartsvsite import SMARTSVSiteOperator
from dmff.operators.templatevsite import TemplateVSiteOperator
from dmff.operators.templatetype import TemplateATypeOperator
from dmff.operators.smartstype import SMARTSATypeOperator
from dmff.utils import DMFFException
import openmm.app as app
import openmm.unit as unit


def test_template_vs_and_template_atype():
    print("TEMPLATE VS & TEMPLATE ATYPE")
    pdb = app.PDBFile("tests/data/template_vsite.pdb")
    top = pdb.topology
    topdata = DMFFTopology(top)
    hamilt = Hamiltonian("tests/data/template_and_vsite.xml")

    # add vsite
    op = TemplateVSiteOperator(hamilt.ffinfo)
    op2 = TemplateATypeOperator(hamilt.ffinfo)
    topdata = op(topdata)
    topdata = op2(topdata)

    for atom in topdata.atoms():
        print(atom.meta)
        if "type" not in atom.meta:
            raise DMFFException(f"Atomtype of {atom} not found.")


def test_smarts_vs_and_template_atype():
    print("SMARTS VS & TEMPLATE ATYPE")
    pdb = app.PDBFile("tests/data/template_vsite.pdb")
    top = pdb.topology
    topdata = DMFFTopology(top)
    hamilt = Hamiltonian("tests/data/template_and_vsite.xml", "tests/data/smarts_and_vsite.xml")
    op = SMARTSVSiteOperator(hamilt.ffinfo)
    op2 = TemplateATypeOperator(hamilt.ffinfo)
    topdata = op(topdata)
    topdata = op2(topdata)
    
    for atom in topdata.atoms():
        print(atom.meta)
        if "type" not in atom.meta:
            raise DMFFException(f"Atomtype of {atom} not found.")

def test_smarts_vs_and_smarts_atype():
    print("SMARTS VS & SMARTS ATYPE")
    pdb = app.PDBFile("tests/data/template_vsite.pdb")
    top = pdb.topology
    topdata = DMFFTopology(top)
    hamilt = Hamiltonian("tests/data/smarts_test1.xml", "tests/data/smarts_and_vsite.xml")
    op = SMARTSVSiteOperator(hamilt.ffinfo)
    op2 = SMARTSATypeOperator(hamilt.ffinfo)
    topdata = op(topdata)
    topdata = op2(topdata)
    for atom in topdata.atoms():
        print(atom.meta)
        if "type" not in atom.meta:
            raise DMFFException(f"Atomtype of {atom} not found.")

if __name__ == "__main__":
    test_template_vs_and_template_atype()
    test_smarts_vs_and_template_atype()
    test_smarts_vs_and_smarts_atype()