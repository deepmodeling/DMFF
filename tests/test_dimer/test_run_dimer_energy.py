from dmff.api.xmlio import XMLIO


def test_parse_xml():
    xmlio = XMLIO()
    xmlio.loadXML("tests/data/dimer/forcefield.xml")
    xmlio.loadXML("tests/data/dimer/gaff2.xml")
    xmlio.loadXML("tests/data/dimer/amber14_prot.xml")
    ffinfo = xmlio.parseXML()


if __name__ == "__main__":
    test_parse_xml()