from dmff.api.xmlio import XMLIO
from dmff.api.topology import DMFFTopology
from dmff.operators import SMARTSATypeOperator, SMARTSVSiteOperator, AM1ChargeOperator, TemplateATypeOperator
import openmm.app as app


def load_xml():
    xmlio = XMLIO()
    xmlio.loadXML("tests/data/dimer/forcefield.xml")
    xmlio.loadXML("tests/data/dimer/gaff2.xml")
    xmlio.loadXML("tests/data/dimer/amber14_prot.xml")
    ffinfo = xmlio.parseXML()
    return ffinfo

def test_load_sdf():
    ffinfo = load_xml()
    top = DMFFTopology(from_mol="tests/data/dimer/mol.sdf")
    smarts_type = SMARTSATypeOperator(ffinfo)
    smarts_vsite = SMARTSVSiteOperator(ffinfo)
    am1_charge = AM1ChargeOperator(ffinfo)
    top = smarts_vsite(smarts_type(am1_charge(top)))
    for atom in top.atoms():
        print(atom.meta)
    print(top.parseSMARTS("[#0:1]~[#17:2]"))
    print(top.parseSMARTS("[#0:1]~[#7:2]"))

def test_load_protein():
    ffinfo = load_xml()
    pdb = app.PDBFile("tests/data/dimer/ala.pdb")
    top = DMFFTopology(from_top=pdb.topology)
    template_type = TemplateATypeOperator(ffinfo)
    top = template_type(top)
    for atom in top.atoms():
        print(atom.meta)

def test_build_dimer():
    ffinfo = load_xml()

    mol_top = DMFFTopology(from_mol="tests/data/dimer/mol.sdf")
    smarts_type = SMARTSATypeOperator(ffinfo)
    smarts_vsite = SMARTSVSiteOperator(ffinfo)
    am1_charge = AM1ChargeOperator(ffinfo)
    mol_top = smarts_vsite(smarts_type(am1_charge(mol_top)))

    pdb = app.PDBFile("tests/data/dimer/ala.pdb")
    prot_top = DMFFTopology(from_top=pdb.topology)
    template_type = TemplateATypeOperator(ffinfo)
    prot_top = template_type(prot_top)

    mol_top.add(prot_top)
    for atom in mol_top.atoms():
        print(atom.meta)
    
    update_func = mol_top.buildVSiteUpdateFunction()


if __name__ == "__main__":
    test_load_sdf()
    test_load_protein()
    test_build_dimer()