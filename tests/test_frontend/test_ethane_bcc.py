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


def test_ethane():
    pdb = app.PDBFile("tests/data/ethane.pdb")
    pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    pos = jnp.array(pos)
    topdata = DMFFTopology(from_top=pdb.topology)

    xmlio = XMLIO()
    xmlio.loadXML("tests/data/ethane_smarts_newapi.xml")
    ffinfo = xmlio.parseXML()
    smartsATOP = SMARTSATypeOperator(ffinfo)
    templateOP = TemplateATypeOperator(ffinfo)
    top_mod = smartsATOP(templateOP(topdata))
    top_mod.setPeriodicBoxVectors(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) * 4.0)
    for atom in top_mod.atoms():
        print(atom.meta)
        pass
    cov_mat = top_mod.buildCovMat()
    cutoff = 1.0

    paramset = ParamSet()
    box = top_mod.getPeriodicBoxVectors(use_jax=True)

    nblist = NeighborListFreud(box, cutoff, cov_map=cov_mat)
    pairs = nblist.allocate(pos)

    generator = CoulombGenerator(ffinfo, paramset)
    pot_func = generator.createPotential(
        top_mod, nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=cutoff, args={})
    print("E_coul", pot_func(pos, box, pairs, paramset))

    generator_lj = LennardJonesGenerator(ffinfo, paramset)
    pot_func_lj = generator_lj.createPotential(
        top_mod, nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=cutoff, args={})
    print("E_LJ", pot_func_lj(pos, box, pairs, paramset))



if __name__ == "__main__":
    test_ethane()