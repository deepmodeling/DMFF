import openmm.app as app
import openmm.unit as unit
from dmff.api.topology import DMFFTopology
from dmff.api.paramset import ParamSet
from dmff.operators import TemplateVSiteOperator, SMARTSVSiteOperator, TemplateATypeOperator, SMARTSATypeOperator, AM1ChargeOperator, GAFFTypeOperator
from dmff.api.xmlio import XMLIO
from dmff.generators.classical import CoulombGenerator, LennardJonesGenerator
import numpy as np
import jax.numpy as jnp
from dmff import NeighborListFreud
from dmff.settings import update_jax_precision
update_jax_precision("double")


def dist_pbc(vi, vj, box):
    box_inv = np.linalg.inv(box)
    drvec = (vi - vj).reshape((1, 3))
    unshifted_dsvecs = drvec.dot(box_inv)
    dsvecs = unshifted_dsvecs - np.floor(unshifted_dsvecs + 0.5)
    dsret = dsvecs.dot(box)
    return np.linalg.norm(dsret, axis=1)[0]


def run_water_cutoff():
    pdb = app.PDBFile("tests/data/waterbox.pdb")
    print(pdb.topology)
    pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    pos = jnp.array(pos)
    dmfftop = DMFFTopology(from_top=pdb.topology)
    print(dmfftop)
    cov_mat = dmfftop.buildCovMat()
    atoms = [a for a in dmfftop.atoms()]
    print(cov_mat)

    xmlio = XMLIO()
    xmlio.loadXML("tests/data/water4.xml")
    ffinfo = xmlio.parseXML()
    tempOP = TemplateATypeOperator(ffinfo)
    top_atype = tempOP(dmfftop)
    for atom in top_atype.atoms():
        # print(atom.meta)
        pass

    cutoff = 1.0

    paramset = ParamSet()
    box = top_atype.getPeriodicBoxVectors(use_jax=True)

    nblist = NeighborListFreud(box, cutoff, cov_map=cov_mat)
    pairs = nblist.allocate(pos)

    generator = CoulombGenerator(ffinfo, paramset)
    pot_func = generator.createPotential(
        top_atype, nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=cutoff, args={})
    print("E_coul", pot_func(pos, box, pairs, paramset))

    generator_lj = LennardJonesGenerator(ffinfo, paramset)
    pot_func_lj = generator_lj.createPotential(
        top_atype, nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=cutoff, args={})
    print("E_LJ", pot_func_lj(pos, box, pairs, paramset))


if __name__ == "__main__":
    run_water_cutoff()
