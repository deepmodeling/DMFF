import openmm.app as app
from dmff.api.topology import DMFFTopology
from dmff.operators import TemplateVSiteOperator, SMARTSVSiteOperator, TemplateATypeOperator, SMARTSATypeOperator, AM1ChargeOperator
from dmff.api.xmlio import XMLIO


def build_test_mol():
    ret = DMFFTopology()
    top = DMFFTopology()
    chain = top.addChain()
    res = top.addResidue("MOL", chain)
    n1 = top.addAtom("N1", app.element.nitrogen, res)
    c1 = top.addAtom("C1", app.element.carbon, res)
    c2 = top.addAtom("C2", app.element.carbon, res)
    c3 = top.addAtom("C3", app.element.carbon, res)
    c4 = top.addAtom("C4", app.element.carbon, res)
    h1 = top.addAtom("H1", app.element.hydrogen, res)
    h2 = top.addAtom("H2", app.element.hydrogen, res)
    h3 = top.addAtom("H3", app.element.hydrogen, res)
    h4 = top.addAtom("H4", app.element.hydrogen, res)
    br = top.addAtom("BR", app.element.bromine, res)
    top.addBond(n1, c1, order=1)
    top.addBond(c1, c2, order=2)
    top.addBond(c2, c3, order=1)
    top.addBond(c3, c4, order=2)
    top.addBond(c4, n1, order=1)
    top.addBond(n1, br, order=1)
    top.addBond(c1, h1, order=1)
    top.addBond(c2, h2, order=1)
    top.addBond(c3, h3, order=1)
    top.addBond(c4, h4, order=1)
    ret.add(top)
    ret.add(top)
    ret.updateMolecules()
    return ret

def build_test_C2H6():
    ret = DMFFTopology()
    top = DMFFTopology()
    chain = top.addChain()
    res = top.addResidue("MOL", chain)
    c1 = top.addAtom("C1", app.element.carbon, res)
    h1 = top.addAtom("H1", app.element.hydrogen, res)
    h2 = top.addAtom("H2", app.element.hydrogen, res)
    h3 = top.addAtom("H3", app.element.hydrogen, res)
    c2 = top.addAtom("C2", app.element.carbon, res)
    h21 = top.addAtom("H21", app.element.hydrogen, res)
    h22 = top.addAtom("H22", app.element.hydrogen, res)
    c3 = top.addAtom("C3", app.element.carbon, res)
    h31 = top.addAtom("H31", app.element.hydrogen, res)
    h32 = top.addAtom("H32", app.element.hydrogen, res)
    c4 = top.addAtom("C4", app.element.carbon, res)
    h4 = top.addAtom("H4", app.element.hydrogen, res)
    h5 = top.addAtom("H5", app.element.hydrogen, res)
    h6 = top.addAtom("H6", app.element.hydrogen, res)
    top.addBond(c1, h1, order=1)
    top.addBond(c1, h2, order=1)
    top.addBond(c1, h3, order=1)
    top.addBond(c4, h4, order=1)
    top.addBond(c4, h5, order=1)
    top.addBond(c4, h6, order=1)

    top.addBond(c2, h21, order=1)
    top.addBond(c2, h22, order=1)

    top.addBond(c3, h31, order=1)
    top.addBond(c3, h32, order=1)

    top.addBond(c1, c2, order=1)
    top.addBond(c2, c3, order=1)
    top.addBond(c3, c4, order=1)
    ret.add(top)
    ret.updateMolecules()
    return ret


def test_cov_mat():
    top = build_test_C2H6()
    cov_mat = top.buildCovMat()
    print(cov_mat)

def test_run_coul_generator():
    mol = build_test_mol()
    smartsVSOP = SMARTSVSiteOperator("tests/data/smarts_and_vsite.xml")
    xmlio = XMLIO()
    xmlio.loadXML("tests/data/smarts_test1.xml")
    ffinfo = xmlio.parseXML()
    smartsATypeOP = SMARTSATypeOperator(ffinfo)
    am1ChargeOP = AM1ChargeOperator()
    mol_vsite = am1ChargeOP(smartsATypeOP(smartsVSOP(mol)))
    for atom in mol_vsite.atoms():
        print(atom.meta)

def test_eqv_list():
    mol = build_test_C2H6()
    eq_info = mol.getEquivalentAtoms()
    print(eq_info)

if __name__ == "__main__":
    test_cov_mat()
    test_eqv_list()