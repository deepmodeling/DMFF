import openmm.app as app
from dmff.api.bettertopology import BetterTopology
from dmff.operators import TemplateVSiteOperator, SMARTSVSiteOperator, TemplateATypeOperator, SMARTSATypeOperator, AM1ChargeOperator


def build_test_mol():
    top = BetterTopology()
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
    hn = top.addAtom("HN", app.element.hydrogen, res)
    top.addBond(n1, c1, order=1)
    top.addBond(c1, c2, order=2)
    top.addBond(c2, c3, order=1)
    top.addBond(c3, c4, order=2)
    top.addBond(c4, n1, order=1)
    top.addBond(n1, hn, order=1)
    top.addBond(c1, h1, order=1)
    top.addBond(c2, h2, order=1)
    top.addBond(c3, h3, order=1)
    top.addBond(c4, h4, order=1)
    top.updateMolecules()
    return top

def test_better_top():
    top = BetterTopology()
    mol = build_test_mol()
    top.add(mol)
    top.add(mol)
    print(top)

def test_add_vsite_from_template():
    mol = build_test_mol()
    print(mol)
    templateVSOP = TemplateVSiteOperator("tests/data/template_and_vsite.xml")
    mol_vsite = templateVSOP(mol)
    print(mol_vsite)

def test_add_vsite_from_smarts():
    mol = build_test_mol()
    print(mol)
    smartsVSOP = SMARTSVSiteOperator("tests/data/smarts_and_vsite.xml")
    mol_vsite = smartsVSOP(mol)
    print(mol_vsite)

def test_add_atype_from_template():
    mol = build_test_mol()
    templateVSOP = TemplateVSiteOperator("tests/data/template_and_vsite.xml")
    templateATypeOP = TemplateATypeOperator("tests/data/template_and_vsite.xml")
    mol_vsite = templateATypeOP(templateVSOP(mol))
    for atom in mol_vsite.atoms():
        print(atom.meta)

def test_add_atype_from_smarts():
    mol = build_test_mol()
    smartsVSOP = SMARTSVSiteOperator("tests/data/smarts_and_vsite.xml")
    smartsATypeOP = SMARTSATypeOperator("tests/data/smarts_and_vsite.xml")
    mol_vsite = smartsATypeOP(smartsVSOP(mol))
    for atom in mol_vsite.atoms():
        print(atom.meta)

def test_add_am1_charge():
    mol = build_test_mol()
    smartsVSOP = SMARTSVSiteOperator("tests/data/smarts_and_vsite.xml")
    smartsATypeOP = SMARTSATypeOperator("tests/data/smarts_and_vsite.xml")
    am1ChargeOP = AM1ChargeOperator()
    mol_vsite = am1ChargeOP(smartsATypeOP(smartsVSOP(mol)))
    for atom in mol_vsite.atoms():
        print(atom.meta)

def test_add_resp_charge():
    pass


if __name__ == "__main__":
    test_better_top()