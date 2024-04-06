import openmm.app as app
from dmff.api.topology import DMFFTopology
from dmff.operators import TemplateVSiteOperator, SMARTSVSiteOperator, TemplateATypeOperator, SMARTSATypeOperator, AM1ChargeOperator
from dmff.api.xmlio import XMLIO


def build_test_mol():
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
    top.updateMolecules()
    return top

def test_better_top():
    top = DMFFTopology()
    mol = build_test_mol()
    for nmol in range(10):
        top.add(mol)
    print(top)

def test_add_vsite_from_template():
    mol = build_test_mol()
    print(mol)
    xmlio = XMLIO()
    xmlio.loadXML("tests/data/template_and_vsite.xml")
    ffinfo = xmlio.parseXML()
    templateVSOP = TemplateVSiteOperator(ffinfo)
    mol_vsite = templateVSOP(mol)
    for atom in mol_vsite.atoms():
        print(atom)

def test_add_vsite_from_smarts():
    mol = build_test_mol()
    print(mol)
    xmlio = XMLIO()
    xmlio.loadXML("tests/data/smarts_and_vsite.xml")
    ffinfo = xmlio.parseXML()
    smartsVSOP = SMARTSVSiteOperator(ffinfo)
    mol_vsite = smartsVSOP(mol)
    for atom in mol_vsite.atoms():
        print(atom)

def test_add_atype_from_smarts():
    top = DMFFTopology()
    mol = build_test_mol()
    xmlio = XMLIO()
    xmlio.loadXML("tests/data/smarts_test1.xml")
    ffinfo = xmlio.parseXML()
    smartsATypeOP = SMARTSATypeOperator(ffinfo)
    mol = smartsATypeOP(mol)
    top.add(mol)
    for atom in top.atoms():
        print(atom.meta)

def test_add_atype_from_template():
    mol = build_test_mol()
    xmlio = XMLIO()
    xmlio.loadXML("tests/data/template_and_vsite.xml")
    ffinfo = xmlio.parseXML()
    templateVSOP = TemplateVSiteOperator(ffinfo)
    templateATypeOP = TemplateATypeOperator(ffinfo)
    mol_vsite = templateATypeOP(templateVSOP(mol))
    for atom in mol_vsite.atoms():
        print(atom.meta)

def test_add_atype_from_smarts_with_vs():
    top = DMFFTopology()
    mol = build_test_mol()
    xmlio = XMLIO()
    xmlio.loadXML("tests/data/smarts_test1.xml")
    xmlio.loadXML("tests/data/smarts_and_vsite.xml")
    ffinfo = xmlio.parseXML()
    smartsVSOP = SMARTSVSiteOperator(ffinfo)
    smartsATypeOP = SMARTSATypeOperator(ffinfo)
    mol = smartsATypeOP(smartsVSOP(mol))
    top.add(mol)
    for atom in top.atoms():
        print(atom.meta)

def test_add_am1_charge():
    mol = build_test_mol()
    xmlio = XMLIO()
    xmlio.loadXML("tests/data/smarts_test1.xml")
    xmlio.loadXML("tests/data/smarts_and_vsite.xml")
    ffinfo = xmlio.parseXML()
    smartsVSOP = SMARTSVSiteOperator(ffinfo)
    smartsATypeOP = SMARTSATypeOperator(ffinfo)
    am1ChargeOP = AM1ChargeOperator(ffinfo)
    mol_vsite = am1ChargeOP(smartsATypeOP(smartsVSOP(mol)))
    for atom in mol_vsite.atoms():
        print(atom.meta)

<<<<<<< HEAD
def test_add_resp_charge():
    pass

=======
>>>>>>> upstream/devel

if __name__ == "__main__":
    print("--")
    test_better_top()
    print("--")
    test_add_atype_from_smarts()
    print("--")
    test_add_vsite_from_template()
    print("--")
    test_add_vsite_from_smarts()
    print("--")
    test_add_atype_from_template()
    print("--")
    test_add_atype_from_smarts_with_vs()
    print("--")
    test_add_am1_charge()