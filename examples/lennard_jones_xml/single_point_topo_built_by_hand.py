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
pos = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
pos = jnp.array(pos)
dmfftop = DMFFTopology()
box_from_pdb = pdb.topology.getPeriodicBoxVectors()
box = np.array([v.value_in_unit(unit.nanometer) for v in box_from_pdb])
dmfftop.setPeriodicBoxVectors(box)
chain = dmfftop.addChain()
# water 1
res1 = dmfftop.addResidue("HOH", chain)
atom1 = dmfftop.addAtom("O", "O", res1)
atom1.meta["type"] = "OT"
atom2 = dmfftop.addAtom("H1", "H", res1)
atom2.meta["type"] = "HT"
atom3 = dmfftop.addAtom("H2", "H", res1)
atom3.meta["type"] = "HT"
dmfftop.addBond(atom1, atom2)
dmfftop.addBond(atom1, atom3)

# water 2
res2 = dmfftop.addResidue("HOH", chain)
atom1 = dmfftop.addAtom("O", "O", res2)
atom1.meta["type"] = "OT"
atom2 = dmfftop.addAtom("H1", "H", res2)
atom2.meta["type"] = "HT"
atom3 = dmfftop.addAtom("H2", "H", res2)
atom3.meta["type"] = "HT"
dmfftop.addBond(atom1, atom2)
dmfftop.addBond(atom1, atom3)

# Na+
res3 = dmfftop.addResidue("SOD", chain)
atom1 = dmfftop.addAtom("SOD", "Na", res3)
atom1.meta["type"] = "SOD"

print(dmfftop)
cov_mat = dmfftop.buildCovMat()
atoms = [a for a in dmfftop.atoms()]
print(cov_mat)

xmlio = XMLIO()
xmlio.loadXML("param.xml")
ffinfo = xmlio.parseXML()
for atom in dmfftop.atoms():
    # print(atom.meta)
    pass

cutoff = 1.0

paramset = ParamSet()
box = dmfftop.getPeriodicBoxVectors(use_jax=True)

nblist = NeighborListFreud(box, cutoff, cov_map=cov_mat)
pairs = nblist.allocate(pos)

generator_lj = LennardJonesGenerator(ffinfo, paramset)
pot_func_lj = generator_lj.createPotential(
    dmfftop, nonbondedMethod=app.CutoffPeriodic, nonbondedCutoff=cutoff, args={})
print("E_LJ", pot_func_lj(pos, box, pairs, paramset))