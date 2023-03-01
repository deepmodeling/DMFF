import openmm.app as app
import openmm.unit as unit
from dmff_new.topology import TopologyData
from dmff_new.xmlio import XMLIO
from dmff_new.operators.templatetype import TemplateOperator
import json
import pytest


def test_atype_from_template_single():
    pdb = app.PDBFile("test/data/template_test1.pdb")
    top = pdb.topology
    topdata = TopologyData(top)
    for nres in range(top.getNumResidues()):
        topdata.setOperatorToResidue(nres, "template")

    io = XMLIO()
    io.loadXML("test/data/template_test1.xml")
    ret = io.parseXML()

    operator = TemplateOperator(ret)
    operator.operate(topdata)
    for na, a in enumerate(topdata.atom_meta):
        print(a)
    
def test_atype_from_template_multi():
    pdb = app.PDBFile("test/data/ala3.pdb")
    top = pdb.topology
    topdata = TopologyData(top)
    for nres in range(top.getNumResidues()):
        topdata.setOperatorToResidue(nres, "template")

    io = XMLIO()
    io.loadXML("test/data/amber14_prot.xml")
    ret = io.parseXML()
    
    typifier = TemplateOperator(ret)
    typifier.operate(topdata)
    for na, a in enumerate(topdata.atom_meta):
        print(a)


if __name__ == "__main__":
    test_atype_from_template_single()
    test_atype_from_template_multi()