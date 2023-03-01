import openmm.app as app
import openmm.unit as unit
from dmff.topology import TopologyData
from dmff.xmlio import XMLIO
from dmff.operators.smartstype import SMARTSOperator
import json
import pytest


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


def test_atype_from_smarts_single():
    top = build_test_mol()
    topdata = TopologyData(top)
    topdata.setOperatorToResidue(0, "smarts")

    io = XMLIO()
    io.loadXML("tests/data/smarts_test1.xml")
    ret = io.parseXML()

    operator = SMARTSOperator(ret)
    operator.operate(topdata)

    reftype = ["n1", "c1", "c2", "c2", "c1", "hc", "hc", "hc", "hc", "hn"]
    for na, a in enumerate(topdata.atom_meta):
        print(a)
    for na, a in enumerate(topdata.atom_meta):
        assert a["type"] == reftype[na]


if __name__ == "__main__":
    test_atype_from_smarts_single()