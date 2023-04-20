from dmff.api.xmlio import XMLIO
from dmff.api.topology import DMFFTopology
from dmff.operators import SMARTSATypeOperator, SMARTSVSiteOperator, AM1ChargeOperator, TemplateATypeOperator
from dmff.api.paramset import ParamSet
from dmff.generators.classical import CoulombGenerator, LennardJonesGenerator
import jax.numpy as jnp
import jax
import openmm.app as app
import openmm.unit as unit
from rdkit import Chem
import numpy as np


def load_xml():
    xmlio = XMLIO()
    xmlio.loadXML("tests/data/dimer/forcefield.xml")
    xmlio.loadXML("tests/data/dimer/gaff2.xml")
    xmlio.loadXML("tests/data/dimer/amber14_prot.xml")
    ffinfo = xmlio.parseXML()
    return ffinfo


def test_load_sdf():
    ffinfo = load_xml()
    top = DMFFTopology(from_sdf="tests/data/dimer/ligand.mol")
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

    mol_top = DMFFTopology(from_sdf="tests/data/dimer/ligand.mol")
    smarts_type = SMARTSATypeOperator(ffinfo)
    smarts_vsite = SMARTSVSiteOperator(ffinfo)
    am1_charge = AM1ChargeOperator(ffinfo)
    mol_top = smarts_vsite(smarts_type(am1_charge(mol_top)))

    mol_pos = jnp.array(Chem.MolFromMolFile("tests/data/dimer/ligand.mol", removeHs=False).GetConformer().GetPositions()) * 0.1
    mol_pos = mol_top.addVSiteToPos(mol_pos)

    pdb = app.PDBFile("tests/data/dimer/ala.pdb")
    prot_top = DMFFTopology(from_top=pdb.topology)
    template_type = TemplateATypeOperator(ffinfo)
    prot_top = template_type(prot_top)
    prot_pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    mol_top.add(prot_top)
    for atom in mol_top.atoms():
        print(atom.meta)

    update_func = mol_top.buildVSiteUpdateFunction()

    paramset = ParamSet()
    coul_gen = CoulombGenerator(ffinfo, paramset)
    lj_gen = LennardJonesGenerator(ffinfo, paramset)

    coul_force = coul_gen.createPotential(
        mol_top, nonbondedMethod=app.NoCutoff, nonbondedCutoff=1.0, args={})
    lj_force = lj_gen.createPotential(
        mol_top, nonbondedMethod=app.NoCutoff, nonbondedCutoff=1.0, args={})

    cov_mat = mol_top.buildCovMat()
    pairs = []
    for ii in range(mol_top.getNumAtoms()):
        for jj in range(ii+1, mol_top.getNumAtoms()):
            pairs.append([ii, jj, cov_mat[ii, jj]])
    pairs = jnp.array(pairs, dtype=int)

    pos = np.concatenate([mol_pos, prot_pos], axis=0)
    pos = jnp.array(pos)

    box = jnp.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ])

    def energy_func(pos, box, pairs, params):
        newpos = update_func(pos)
        e_coul = coul_force(pos, box, pairs, params)
        e_lj = lj_force(pos, box, pairs, params)
        return e_coul + e_lj

    print(energy_func(pos, box, pairs, paramset))
    print(jax.value_and_grad(energy_func, argnums=3)(pos, box, pairs, paramset))
    print(mol_top._meta["bcc_top_mat"])


if __name__ == "__main__":
    test_load_sdf()
    test_load_protein()
    test_build_dimer()
