import openmm.app as app
from dmff.api.vstools import addVSiteToTopology, TwoParticleAverageSite, ThreeParticleAverageSite, TemplateVSitePatcher, SMARTSVSitePatcher
from dmff.api.topology import TopologyData


def build_test_mol():
    top = app.Topology()
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
    return top

def build_vsite_test_mol():
    top = app.Topology()
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
    return top


def test_add_virtual_site_to_topology():
    top = build_test_mol()
    atoms = list(top.atoms())

    vslist = [
        ThreeParticleAverageSite([atoms[0], atoms[1], atoms[4]], [
                                 0.33333, 0.33333, 0.33333]),
        TwoParticleAverageSite([atoms[0], atoms[9]], [0.5, 0.5])
    ]

    newtop, new_vslist = addVSiteToTopology(top, vslist)
    for atom in list(newtop.atoms()):
        print(atom)
    for bond in list(newtop.bonds()):
        print(bond)
    for vs in new_vslist:
        print(vs.atoms)
        print(vs.weights)
        print(vs.vatom)


def test_add_virtual_site_to_topodata():
    top = build_test_mol()
    atoms = list(top.atoms())
    vslist = [
        ThreeParticleAverageSite([atoms[0], atoms[1], atoms[4]], [
                                 0.33333, 0.33333, 0.33333]),
        TwoParticleAverageSite([atoms[0], atoms[9]], [0.5, 0.5])
    ]

    newtop, new_vslist = addVSiteToTopology(top, vslist)
    topdata = TopologyData(newtop)
    topdata.addVirtualSiteList(new_vslist)


def test_patch_vsite_from_template():
    top = build_vsite_test_mol()
    vslist = []
    topdata = TopologyData(top)
    patcher = TemplateVSitePatcher()
    patcher.loadFile("tests/data/template_and_vsite.xml")
    patcher.parse()
    patcher.patch(topdata, None, vslist)
    newtop, new_vslist = addVSiteToTopology(top, vslist)
    topdata = TopologyData(newtop)
    topdata.addVirtualSiteList(new_vslist)
    print(topdata.topology)
    


def test_patch_vsite_from_smarts():
    top = build_vsite_test_mol()
    vslist = []
    topdata = TopologyData(top)
    patcher = SMARTSVSitePatcher()
    patcher.loadFile("tests/data/smarts_and_vsite.xml")
    patcher.parse()
    patcher.patch(topdata, None, vslist)
    newtop, new_vslist = addVSiteToTopology(top, vslist)
    topdata = TopologyData(newtop)
    topdata.addVirtualSiteList(new_vslist)
    print(new_vslist)
    print(topdata.topology)


def test_typification_with_template_and_vsite():
    pass


def test_typification_with_smarts_and_vsite():
    pass


if __name__ == "__main__":
    test_add_virtual_site_to_topology()
    test_add_virtual_site_to_topodata()
    test_patch_vsite_from_template()
    test_patch_vsite_from_smarts()
    test_typification_with_template_and_vsite()
    test_typification_with_smarts_and_vsite()
