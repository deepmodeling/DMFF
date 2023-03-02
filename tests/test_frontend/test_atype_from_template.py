import openmm.app as app
import openmm.unit as unit
from dmff.api.topology import TopologyData
from dmff.api.xmlio import XMLIO
from dmff.operators.templatetype import TemplateOperator
from dmff.api.vstools import addVSiteToTopology, TwoParticleAverageSite, ThreeParticleAverageSite, TemplateVSitePatcher, SMARTSVSitePatcher
import json
import pytest


def test_atype_from_template_single():
    pdb = app.PDBFile("tests/data/template_test1.pdb")
    top = pdb.topology
    topdata = TopologyData(top)
    for nres in range(top.getNumResidues()):
        topdata.setOperatorToResidue(nres, "template")

    io = XMLIO()
    io.loadXML("tests/data/template_test1.xml")
    ret = io.parseXML()

    operator = TemplateOperator(ret)
    operator.operate(topdata)
    for na, a in enumerate(topdata.atom_meta):
        print(a)
    
def test_atype_from_template_multi():
    pdb = app.PDBFile("tests/data/ala3.pdb")
    top = pdb.topology
    topdata = TopologyData(top)
    for nres in range(top.getNumResidues()):
        topdata.setOperatorToResidue(nres, "template")

    io = XMLIO()
    io.loadXML("tests/data/amber14_prot.xml")
    ret = io.parseXML()
    
    typifier = TemplateOperator(ret)
    typifier.operate(topdata)
    for na, a in enumerate(topdata.atom_meta):
        print(a)

def test_atype_from_template_with_vsite():
    pdb = app.PDBFile("tests/data/template_vsite.pdb")
    top = pdb.topology
    topdata = TopologyData(top)
    vslist = []
    patcher = TemplateVSitePatcher()
    patcher.loadFile("tests/data/template_and_vsite.xml")
    patcher.parse()
    patcher.patch(topdata, None, vslist)
    newtop, new_vslist = addVSiteToTopology(top, vslist)
    topdata = TopologyData(newtop)
    topdata.addVirtualSiteList(new_vslist)

    for nres in range(newtop.getNumResidues()):
        topdata.setOperatorToResidue(nres, "template")

    io = XMLIO()
    io.loadXML("tests/data/template_and_vsite.xml")
    ret = io.parseXML()

    operator = TemplateOperator(ret)
    operator.operate(topdata)
    for na, a in enumerate(topdata.atom_meta):
        print(a)


if __name__ == "__main__":
    # test_atype_from_template_single()
    # test_atype_from_template_multi()
    test_atype_from_template_with_vsite()