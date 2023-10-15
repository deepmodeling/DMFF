import openmm.app as app
import openmm.unit as unit
from dmff.api.topology import DMFFTopology
from dmff.api.paramset import ParamSet
from dmff.operators import TemplateVSiteOperator, SMARTSVSiteOperator, TemplateATypeOperator, SMARTSATypeOperator, AM1ChargeOperator, GAFFTypeOperator
from dmff.api.xmlio import XMLIO
from dmff.generators.classical import CoulombGenerator, LennardJonesGenerator
import numpy as np
import jax.numpy as jnp
from dmff.settings import update_jax_precision
update_jax_precision("double")

def build_test_mol():
    ret = DMFFTopology()
    top = DMFFTopology()
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
    ret.add(top)
    ret.add(top)
    ret.updateMolecules()
    return ret

def build_test_C4H10():
    ret = DMFFTopology()
    top = DMFFTopology()
    chain = top.addChain()
    res = top.addResidue("MOL", chain)
    c1 = top.addAtom("C1", app.element.carbon, res)
    h1 = top.addAtom("H1", app.element.hydrogen, res)
    h2 = top.addAtom("H2", app.element.hydrogen, res)
    h3 = top.addAtom("H3", app.element.hydrogen, res)
    c2 = top.addAtom("C2", app.element.carbon, res)
    h21 = top.addAtom("H21", app.element.hydrogen, res)
    h22 = top.addAtom("H22", app.element.hydrogen, res)
    c3 = top.addAtom("C3", app.element.carbon, res)
    h31 = top.addAtom("H31", app.element.hydrogen, res)
    h32 = top.addAtom("H32", app.element.hydrogen, res)
    c4 = top.addAtom("C4", app.element.carbon, res)
    h4 = top.addAtom("H4", app.element.hydrogen, res)
    h5 = top.addAtom("H5", app.element.hydrogen, res)
    h6 = top.addAtom("H6", app.element.hydrogen, res)
    top.addBond(c1, h1, order=1)
    top.addBond(c1, h2, order=1)
    top.addBond(c1, h3, order=1)
    top.addBond(c4, h4, order=1)
    top.addBond(c4, h5, order=1)
    top.addBond(c4, h6, order=1)

    top.addBond(c2, h21, order=1)
    top.addBond(c2, h22, order=1)

    top.addBond(c3, h31, order=1)
    top.addBond(c3, h32, order=1)

    top.addBond(c1, c2, order=1)
    top.addBond(c2, c3, order=1)
    top.addBond(c3, c4, order=1)
    ret.add(top)
    ret.updateMolecules()
    return ret


def test_cov_mat():
    top = build_test_C4H10()
    cov_mat = top.buildCovMat()
    print(cov_mat)

def test_run_coul_and_lj_nocutoff():
    pdb = app.PDBFile("tests/data/c4h10.pdb")
    pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    pos = jnp.array(pos)
    mol = build_test_C4H10()
    mol.setPeriodicBoxVectors(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) * 3.0)
    atoms = [a for a in mol.atoms()]

    xmlio = XMLIO()
    xmlio.loadXML("tests/data/smarts_test1.xml")
    xmlio.loadXML("tests/data/smarts_and_vsite.xml")
    xmlio.loadXML("tests/data/inter_test.xml")
    ffinfo = xmlio.parseXML()
    smartsVSOP = SMARTSVSiteOperator(ffinfo)
    gaffTypeOP = GAFFTypeOperator(ffinfo)
    am1ChargeOP = AM1ChargeOperator(ffinfo)
    mol_vsite = am1ChargeOP(gaffTypeOP(smartsVSOP(mol)))
    for atom in mol_vsite.atoms():
        print(atom.meta)
        pass

    cov_mat = mol.buildCovMat()
    print(cov_mat)

    paramset = ParamSet()
    box = mol.getPeriodicBoxVectors(use_jax=True)
    pairs = []
    for ii in range(mol.getNumAtoms()):
        for jj in range(ii+1, mol.getNumAtoms()):
            pairs.append([ii, jj, 0])
    pairs = jnp.array(pairs)
    pairs = pairs.at[:,2].set(cov_mat[pairs[:,0], pairs[:,1]])
    
    generator = CoulombGenerator(ffinfo, paramset)
    pot_func = generator.createPotential(mol, nonbondedMethod=app.NoCutoff, nonbondedCutoff=999.9, args={})
    print("E_coul", pot_func(pos, box, pairs, paramset))

    generator_lj = LennardJonesGenerator(ffinfo, paramset)
    pot_func_lj = generator_lj.createPotential(mol, nonbondedMethod=app.NoCutoff, nonbondedCutoff=999.9, args={})
    print("E_LJ", pot_func_lj(pos, box, pairs, paramset))

    # <Atom class="c3" sigma="0.3397709531243626" epsilon="0.4510352"/>
    # <Atom class="hc" sigma="0.2600176998764394" epsilon="0.0870272"/>
    mscales_lj = jnp.array([0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0])
    elj = 0.0
    def dist(ii, jj):
        return np.linalg.norm(pos[ii] - pos[jj])
    for ii, jj, norder in pairs:
        d = dist(ii, jj)
        if atoms[ii].meta["element"] == "C":
            eps1, sig1 = 0.4510352, 0.3397710
        elif atoms[ii].meta["element"] == "H":
            eps1, sig1 = 0.0870272, 0.2601770
        else:
            raise BaseException(f"ELEM: {atoms[ii].meta['element']}")
        
        if atoms[jj].meta["element"] == "C":
            eps2, sig2 = 0.4510352, 0.3397710
        elif atoms[jj].meta["element"] == "H":
            eps2, sig2 = 0.0870272, 0.2601770
        else:
            raise BaseException(f"ELEM: {atoms[ii].meta['element']}")
        
        sig = (sig1 + sig2) / 2
        eps = np.sqrt(eps1 * eps2)
        one_sig = sig / d
        de = mscales_lj[norder-1] * 4. * eps * (one_sig ** 12 - one_sig ** 6)
        # print(f"{ii} {jj} {sig:.4f}, {eps:.4f}, {d:.4f}, {mscales_lj[norder-1]:.2f}, {de:.4f}")
        elj += de
    print("E_LJ calc: ", elj)

def test_run_coul_and_lj_pme():
    pdb = app.PDBFile("tests/data/c4h10.pdb")
    pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    pos = jnp.array(pos)
    mol = build_test_C4H10()
    mol.setPeriodicBoxVectors(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) * 2.5)
    cov_mat = mol.buildCovMat()

    xmlio = XMLIO()
    xmlio.loadXML("tests/data/smarts_test1.xml")
    xmlio.loadXML("tests/data/smarts_and_vsite.xml")
    xmlio.loadXML("tests/data/inter_test.xml")
    ffinfo = xmlio.parseXML()
    smartsVSOP = SMARTSVSiteOperator(ffinfo)
    gaffTypeOP = GAFFTypeOperator(ffinfo)
    am1ChargeOP = AM1ChargeOperator(ffinfo)
    mol_vsite = am1ChargeOP(gaffTypeOP(smartsVSOP(mol)))
    for atom in mol_vsite.atoms():
        print(atom.meta)

    paramset = ParamSet()
    generator = CoulombGenerator(ffinfo, paramset)
    pot_func = generator.createPotential(mol, nonbondedMethod=app.PME, nonbondedCutoff=1.0, args={})
    box = mol.getPeriodicBoxVectors(use_jax=True)
    pairs = []
    for ii in range(mol.getNumAtoms()):
        for jj in range(ii+1, mol.getNumAtoms()):
            if np.linalg.norm(pos[ii] - pos[jj]) < 1.0:
                pairs.append([ii, jj, 0])
    pairs = jnp.array(pairs)
    pairs = pairs.at[:,2].set(cov_mat[pairs[:,0], pairs[:,1]])

    generator = CoulombGenerator(ffinfo, paramset)
    pot_func = generator.createPotential(mol, nonbondedMethod=app.PME, nonbondedCutoff=0.9, args={})
    print("E_coul", pot_func(pos, box, pairs, paramset))

    generator_lj = LennardJonesGenerator(ffinfo, paramset)
    pot_func_lj = generator_lj.createPotential(mol, nonbondedMethod=app.PME, nonbondedCutoff=0.9, args={})
    print("E_LJ", pot_func_lj(pos, box, pairs, paramset))

def test_eqv_list():
    mol = build_test_C4H10()
    eq_info = mol.getEquivalentAtoms()
    print(eq_info)


if __name__ == "__main__":
    # test_cov_mat()
    # test_eqv_list()
    # 
    test_run_coul_and_lj_nocutoff()
    test_run_coul_and_lj_pme()