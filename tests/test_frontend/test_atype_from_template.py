import openmm.app as app
import openmm.unit as unit
from dmff.api.topology import DMFFTopology
from dmff.api.xmlio import XMLIO
from dmff.operators.templatetype import TemplateATypeOperator
import json
import pytest


def test_atype_from_template_single():
    pdb = app.PDBFile("tests/data/template_test1.pdb")
    top = pdb.topology
    topdata = DMFFTopology(top)

    io = XMLIO()
    io.loadXML("tests/data/template_test1.xml")
    ret = io.parseXML()

    operator = TemplateATypeOperator(ret)
    topdata = operator.operate(topdata)
    for na, a in enumerate(topdata.atoms()):
        print(a.meta)
    
def test_atype_from_template_multi():
    pdb = app.PDBFile("tests/data/ala3.pdb")
    top = pdb.topology
    topdata = DMFFTopology(top)

    io = XMLIO()
    io.loadXML("tests/data/amber14_prot.xml")
    ret = io.parseXML()
    
    typifier = TemplateATypeOperator(ret)
    topdata = typifier.operate(topdata)
    for na, a in enumerate(topdata.atoms()):
        print(a.meta)

def test_atype_from_template_with_vsite():
    pdb = app.PDBFile("tests/data/template_vsite.pdb")
    top = pdb.topology
    topdata = DMFFTopology(top)

    io = XMLIO()
    io.loadXML("tests/data/template_and_vsite.xml")
    ret = io.parseXML()

    operator = TemplateATypeOperator(ret)
    operator.operate(topdata)
    for na, a in enumerate(topdata.atoms()):
        print(a.meta)


if __name__ == "__main__":
    # test_atype_from_template_single()
    # test_atype_from_template_multi()
    test_atype_from_template_with_vsite()