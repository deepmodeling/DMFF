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


pdb = app.PDBFile("structure.pdb")
print(pdb.topology)
pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
pos = jnp.array(pos)
dmfftop = DMFFTopology(from_top=pdb.topology)
print(dmfftop)
cov_mat = dmfftop.buildCovMat()
atoms = [a for a in dmfftop.atoms()]
print(cov_mat)

xmlio = XMLIO()
xmlio.loadXML("param.xml")
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

generator_lj = LennardJonesGenerator(ffinfo, paramset)
pot_func_lj = generator_lj.createPotential(
    top_atype, nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=cutoff, args={})
print("E_LJ", pot_func_lj(pos, box, pairs, paramset))