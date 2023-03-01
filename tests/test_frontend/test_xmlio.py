from dmff.xmlio import XMLIO
import pytest
import os


@pytest.mark.parametrize('input', ['tests/data/test1.xml'])
def test_read_single_xml(input):
    io = XMLIO()
    io.loadXML(input)
    ret = io.parseXML()
    assert len(ret["AtomTypes"]) == 4

    assert len(ret["Residues"]) == 1
    assert len(ret['Residues'][0]['particles']) == 4
    assert len(ret['Residues'][0]['bonds']) == 3
    assert len(ret['Residues'][0]['externals']) == 2

    assert len(ret["Forces"]) == 2
    assert len(ret["Forces"]['HarmonicBondForce']['node']) == 3
    assert "k" in ret["Forces"]['HarmonicBondForce']['node'][0]["attrib"]
    assert "length" in ret["Forces"]['HarmonicBondForce']['node'][0]["attrib"]
    assert "type1" in ret["Forces"]['HarmonicBondForce']['node'][0]["attrib"]
    assert "type2" in ret["Forces"]['HarmonicBondForce']['node'][0]["attrib"]
    assert len(ret["Forces"]['NonbondedForce']['node']) == 4
    assert "charge" in ret["Forces"]['NonbondedForce']['node'][0]["attrib"]
    assert "epsilon" in ret["Forces"]['NonbondedForce']['node'][0]["attrib"]
    assert "sigma" in ret["Forces"]['NonbondedForce']['node'][0]["attrib"]
    assert "type" in ret["Forces"]['NonbondedForce']['node'][0]["attrib"]
    print(ret)

@pytest.mark.parametrize('input1,input2', [('tests/data/test1.xml', 'tests/data/test2.xml')])
def test_read_multi_xml(input1, input2):
    io = XMLIO()
    io.loadXML(input1)
    io.loadXML(input2)
    ret = io.parseXML()
    assert len(ret["AtomTypes"]) == 8

    assert len(ret["Residues"]) == 2
    assert len(ret['Residues'][0]['particles']) == 4
    assert len(ret['Residues'][0]['bonds']) == 3
    assert len(ret['Residues'][0]['externals']) == 2
    assert len(ret['Residues'][1]['particles']) == 5
    assert len(ret['Residues'][1]['bonds']) == 4
    assert len(ret['Residues'][1]['externals']) == 2

    assert len(ret["Forces"]) == 2
    assert len(ret["Forces"]['HarmonicBondForce']['node']) == 6
    assert "k" in ret["Forces"]['HarmonicBondForce']['node'][0]["attrib"]
    assert "length" in ret["Forces"]['HarmonicBondForce']['node'][0]["attrib"]
    assert "type1" in ret["Forces"]['HarmonicBondForce']['node'][0]["attrib"]
    assert "type2" in ret["Forces"]['HarmonicBondForce']['node'][0]["attrib"]
    assert len(ret["Forces"]['NonbondedForce']['node']) == 8
    assert "charge" in ret["Forces"]['NonbondedForce']['node'][0]["attrib"]
    assert "epsilon" in ret["Forces"]['NonbondedForce']['node'][0]["attrib"]
    assert "sigma" in ret["Forces"]['NonbondedForce']['node'][0]["attrib"]
    assert "type" in ret["Forces"]['NonbondedForce']['node'][0]["attrib"]


@pytest.mark.parametrize('input', ['tests/data/test1.xml'])
def test_write_xml(input):
    io = XMLIO()
    io.loadXML(input)
    ret = io.parseXML()
    io.writeXML("tests/data/test_out.xml", ret)

    io = XMLIO()
    io.loadXML("tests/data/test_out.xml")
    ret = io.parseXML()
    assert len(ret["AtomTypes"]) == 4

    assert len(ret["Residues"]) == 1
    assert len(ret['Residues'][0]['particles']) == 4
    assert len(ret['Residues'][0]['bonds']) == 3
    assert len(ret['Residues'][0]['externals']) == 2

    assert len(ret["Forces"]) == 2
    assert len(ret["Forces"]['HarmonicBondForce']['node']) == 3
    assert "k" in ret["Forces"]['HarmonicBondForce']['node'][0]["attrib"]
    assert "length" in ret["Forces"]['HarmonicBondForce']['node'][0]["attrib"]
    assert "type1" in ret["Forces"]['HarmonicBondForce']['node'][0]["attrib"]
    assert "type2" in ret["Forces"]['HarmonicBondForce']['node'][0]["attrib"]
    assert len(ret["Forces"]['NonbondedForce']['node']) == 4
    assert "charge" in ret["Forces"]['NonbondedForce']['node'][0]["attrib"]
    assert "epsilon" in ret["Forces"]['NonbondedForce']['node'][0]["attrib"]
    assert "sigma" in ret["Forces"]['NonbondedForce']['node'][0]["attrib"]
    assert "type" in ret["Forces"]['NonbondedForce']['node'][0]["attrib"]
    os.remove("tests/data/test_out.xml")


if __name__ == "__main__":
    test_read_single_xml('tests/data/test1.xml')