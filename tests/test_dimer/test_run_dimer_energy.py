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

def test_dimer_coul():
    ffinfo = load_xml()

    print("------- Jax Ener Func --------")
    # mol1
    top1 = DMFFTopology(from_sdf="tests/data/dimer/mol1.mol")
    smarts_type = SMARTSATypeOperator(ffinfo)
    smarts_vsite = SMARTSVSiteOperator(ffinfo)
    am1_charge = AM1ChargeOperator(ffinfo)
    top1 = smarts_vsite(smarts_type(am1_charge(top1)))
    pos1 = jnp.array(Chem.MolFromMolFile("tests/data/dimer/mol1.mol", removeHs=False).GetConformer().GetPositions()) * 0.1
    pos1 = top1.addVSiteToPos(pos1)

    paramset = ParamSet()
    coul_gen = CoulombGenerator(ffinfo, paramset)

    # mol2
    top2 = DMFFTopology(from_sdf="tests/data/dimer/mol2.mol")
    top2 = smarts_vsite(smarts_type(am1_charge(top2)))
    pos2 = jnp.array(Chem.MolFromMolFile("tests/data/dimer/mol2.mol", removeHs=False).GetConformer().GetPositions()) * 0.1
    pos2 = top2.addVSiteToPos(pos2)

    box = jnp.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ])

    # mol1_energy
    cov_mat1 = top1.buildCovMat()
    pairs1 = []
    for ii in range(top1.getNumAtoms()):
        for jj in range(ii+1, top1.getNumAtoms()):
            pairs1.append([ii, jj, cov_mat1[ii, jj]])
    pairs1 = jnp.array(pairs1, dtype=int)

    coul_force1 = coul_gen.createPotential(
        top1, nonbondedMethod=app.NoCutoff, nonbondedCutoff=1.0, args={})
    ener_mol1 = coul_force1(pos1, box, pairs1, paramset)
    print("JAX mol1: ", ener_mol1)

    # mol2_energy
    cov_mat2 = top2.buildCovMat()
    pairs2 = []
    for ii in range(top2.getNumAtoms()):
        for jj in range(ii+1, top2.getNumAtoms()):
            pairs2.append([ii, jj, cov_mat2[ii, jj]])
    pairs2 = jnp.array(pairs2, dtype=int)

    coul_force2 = coul_gen.createPotential(
        top2, nonbondedMethod=app.NoCutoff, nonbondedCutoff=1.0, args={})
    ener_mol2 = coul_force2(pos2, box, pairs2, paramset)
    print("JAX mol2", ener_mol2)

    # dimer_energy
    pos_sum = jnp.concatenate([pos1, pos2], axis=0)

    top = DMFFTopology()
    top.add(top1)
    top.add(top2)

    cov_mat_sum = top.buildCovMat()
    pairs_sum = []
    for ii in range(top.getNumAtoms()):
        for jj in range(ii+1, top.getNumAtoms()):
            pairs_sum.append([ii, jj, cov_mat_sum[ii, jj]])
    pairs_sum = jnp.array(pairs_sum, dtype=int)

    coul_force_sum = coul_gen.createPotential(
        top, nonbondedMethod=app.NoCutoff, nonbondedCutoff=1.0, args={})
    ener_sum = coul_force_sum(pos_sum, box, pairs_sum, paramset)
    print("JAX sum: ", ener_sum)

    # interaction
    print("JAX Interaction: ", ener_sum - ener_mol1 - ener_mol2)

    print("------- Calc by hand --------")

    mscales_coul = jnp.array([0.0, 0.0, 0.8333333, 1.0, 1.0,
                                  1.0])
    # mol1_calc
    atoms_top1 = [a for a in top1.atoms()]
    charges = jnp.array([a.meta["charge"] for a in atoms_top1])
    bcc_mat = top1._meta["bcc_top_mat"]
    bcc_charges = charges + jnp.dot(bcc_mat, paramset["CoulombForce"]["bcc"].reshape((-1, 1))).ravel()
    e1_calc = 0.0
    for ii, jj, cdist in pairs1:
        dist = jnp.linalg.norm(pos1[ii,:] - pos1[jj,:])
        chrg1 = bcc_charges[ii]
        chrg2 = bcc_charges[jj]
        e1_calc += 138.935455846 * chrg1 * chrg2 / dist * mscales_coul[cdist-1]
    print("Calc mol1: ", e1_calc )

    # mol2_calc
    atoms_top2 = [a for a in top2.atoms()]
    charges = jnp.array([a.meta["charge"] for a in atoms_top2])
    bcc_mat = top2._meta["bcc_top_mat"]
    bcc_charges = charges + jnp.dot(bcc_mat, paramset["CoulombForce"]["bcc"].reshape((-1, 1))).ravel()
    e2_calc = 0.0
    for ii, jj, cdist in pairs2:
        dist = jnp.linalg.norm(pos2[ii,:] - pos2[jj,:])
        chrg1 = bcc_charges[ii]
        chrg2 = bcc_charges[jj]
        e2_calc += 138.935455846 * chrg1 * chrg2 / dist * mscales_coul[cdist-1]
    print("Calc mol2: ", e2_calc )

    # sum_calc
    atoms_top = [a for a in top.atoms()]
    charges = jnp.array([a.meta["charge"] for a in atoms_top])
    bcc_mat = top._meta["bcc_top_mat"]
    bcc_charges = charges + jnp.dot(bcc_mat, paramset["CoulombForce"]["bcc"].reshape((-1, 1))).ravel()
    esum_calc, einter_calc = 0.0, 0.0
    for ii, jj, cdist in pairs_sum:
        dist = jnp.linalg.norm(pos_sum[ii,:] - pos_sum[jj,:])
        chrg1 = bcc_charges[ii]
        chrg2 = bcc_charges[jj]
        e = 138.935455846 * chrg1 * chrg2 / dist * mscales_coul[cdist-1]
        if e > 4000:
            print(ii, jj, cdist, e)
        esum_calc += e
        if ii < top1.getNumAtoms() and jj >= top1.getNumAtoms():
            einter_calc += e
    print("Calc sum: ", esum_calc)
    print("Calc inter: ", einter_calc)


def test_dimer_lj():
    ffinfo = load_xml()
    paramset = ParamSet()
    lj_gen = LennardJonesGenerator(ffinfo, paramset)

    print("------- Jax Ener Func --------")
    # mol1
    top1 = DMFFTopology(from_sdf="tests/data/dimer/mol1.mol")
    smarts_type = SMARTSATypeOperator(ffinfo)
    smarts_vsite = SMARTSVSiteOperator(ffinfo)
    am1_charge = AM1ChargeOperator(ffinfo)
    top1 = smarts_vsite(smarts_type(am1_charge(top1)))
    pos1 = jnp.array(Chem.MolFromMolFile("tests/data/dimer/mol1.mol", removeHs=False).GetConformer().GetPositions()) * 0.1
    pos1 = top1.addVSiteToPos(pos1)

    # mol2
    top2 = DMFFTopology(from_sdf="tests/data/dimer/mol2.mol")
    top2 = smarts_vsite(smarts_type(am1_charge(top2)))
    pos2 = jnp.array(Chem.MolFromMolFile("tests/data/dimer/mol2.mol", removeHs=False).GetConformer().GetPositions()) * 0.1
    pos2 = top2.addVSiteToPos(pos2)

    box = jnp.array([
        [10.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0]
    ])

    # mol1_energy
    cov_mat1 = top1.buildCovMat()
    pairs1 = []
    for ii in range(top1.getNumAtoms()):
        for jj in range(ii+1, top1.getNumAtoms()):
            pairs1.append([ii, jj, cov_mat1[ii, jj]])
    pairs1 = jnp.array(pairs1, dtype=int)

    lj_force1 = lj_gen.createPotential(
        top1, nonbondedMethod=app.NoCutoff, nonbondedCutoff=1.0, args={})
    ener_mol1 = lj_force1(pos1, box, pairs1, paramset)
    print("JAX mol1: ", ener_mol1)

    # mol2_energy
    cov_mat2 = top2.buildCovMat()
    pairs2 = []
    for ii in range(top2.getNumAtoms()):
        for jj in range(ii+1, top2.getNumAtoms()):
            pairs2.append([ii, jj, cov_mat2[ii, jj]])
    pairs2 = jnp.array(pairs2, dtype=int)

    lj_force2 = lj_gen.createPotential(
        top2, nonbondedMethod=app.NoCutoff, nonbondedCutoff=1.0, args={})
    ener_mol2 = lj_force2(pos2, box, pairs2, paramset)
    print("JAX mol2", ener_mol2)

    # dimer_energy
    pos_sum = jnp.concatenate([pos1, pos2], axis=0)

    top = DMFFTopology()
    top.add(top1)
    top.add(top2)

    cov_mat_sum = top.buildCovMat()
    pairs_sum = []
    for ii in range(top.getNumAtoms()):
        for jj in range(ii+1, top.getNumAtoms()):
            pairs_sum.append([ii, jj, cov_mat_sum[ii, jj]])
    pairs_sum = jnp.array(pairs_sum, dtype=int)

    lj_force_sum = lj_gen.createPotential(
        top, nonbondedMethod=app.NoCutoff, nonbondedCutoff=1.0, args={})
    ener_sum = lj_force_sum(pos_sum, box, pairs_sum, paramset)
    print("JAX sum: ", ener_sum)

    # interaction
    print("JAX Interaction: ", ener_sum - ener_mol1 - ener_mol2)

    print("------- Calc by hand --------")

    mscales_lj = jnp.array([0.0, 0.0, 0.5, 1.0, 1.0,
                                  1.0])
    # mol1_calc
    map_idx = top1._meta["lj_map_idx"]
    e1_calc = 0.0
    for ii, jj, cdist in pairs1:
        dist = jnp.linalg.norm(pos1[ii,:] - pos1[jj,:])
        sig1 = paramset["LennardJonesForce"]["sigma"][map_idx[ii]]
        sig2 = paramset["LennardJonesForce"]["sigma"][map_idx[jj]]
        eps1 = paramset["LennardJonesForce"]["epsilon"][map_idx[ii]]
        eps2 = paramset["LennardJonesForce"]["epsilon"][map_idx[jj]]
        eps = jnp.sqrt(eps1 * eps2)
        sig = (sig1 + sig2) * 0.5

        dr_inv = 1.0 / dist
        sig_dr = sig * dr_inv
        sig_dr6 = jnp.power(sig_dr, 6)
        sig_dr12 = jnp.power(sig_dr6, 2)
        e = 4.0 * eps * (sig_dr12 - sig_dr6)
        e1_calc += e * mscales_lj[cdist-1]
    print("Calc mol1: ", e1_calc )

    # mol2_calc
    map_idx = top2._meta["lj_map_idx"]
    e2_calc = 0.0
    for ii, jj, cdist in pairs2:
        dist = jnp.linalg.norm(pos2[ii,:] - pos2[jj,:])
        sig1 = paramset["LennardJonesForce"]["sigma"][map_idx[ii]]
        sig2 = paramset["LennardJonesForce"]["sigma"][map_idx[jj]]
        eps1 = paramset["LennardJonesForce"]["epsilon"][map_idx[ii]]
        eps2 = paramset["LennardJonesForce"]["epsilon"][map_idx[jj]]
        eps = jnp.sqrt(eps1 * eps2)
        sig = (sig1 + sig2) * 0.5

        dr_inv = 1.0 / dist
        sig_dr = sig * dr_inv
        sig_dr6 = jnp.power(sig_dr, 6)
        sig_dr12 = jnp.power(sig_dr6, 2)
        e = 4.0 * eps * (sig_dr12 - sig_dr6)
        e2_calc += e * mscales_lj[cdist-1]
    print("Calc mol2: ", e2_calc )

    # sum_calc
    map_idx = top._meta["lj_map_idx"]
    esum_calc = 0.0
    einter_calc = 0.0
    for ii, jj, cdist in pairs_sum:
        dist = jnp.linalg.norm(pos_sum[ii,:] - pos_sum[jj,:])
        sig1 = paramset["LennardJonesForce"]["sigma"][map_idx[ii]]
        sig2 = paramset["LennardJonesForce"]["sigma"][map_idx[jj]]
        eps1 = paramset["LennardJonesForce"]["epsilon"][map_idx[ii]]
        eps2 = paramset["LennardJonesForce"]["epsilon"][map_idx[jj]]
        eps = jnp.sqrt(eps1 * eps2)
        sig = (sig1 + sig2) * 0.5

        dr_inv = 1.0 / dist
        sig_dr = sig * dr_inv
        sig_dr6 = jnp.power(sig_dr, 6)
        sig_dr12 = jnp.power(sig_dr6, 2)
        e = 4.0 * eps * (sig_dr12 - sig_dr6)
        esum_calc += e * mscales_lj[cdist-1]
        if ii < top1.getNumAtoms() and jj >= top1.getNumAtoms():
            einter_calc += e
    print("Calc sum: ", esum_calc)
    print("Calc inter: ", einter_calc)



if __name__ == "__main__":
    # test_load_sdf()
    # test_load_protein()
    # test_build_dimer()
    print(">>> Test Coul - NoCutoff")
    test_dimer_coul()
    print(">>> Test LJ - NoCutoff")
    test_dimer_lj()
