import openmm.app as app
from .xmlio import XMLIO
from .paramset import ParamSet
from .topology import TopologyData
from ..utils import DMFFException
import jax
import jax.numpy as jnp
import numpy as np

dmff_generators = {}
dmff_operators = {}


def build_covalent_map(data, max_neighbor):
    n_atoms = len(data.atoms)
    covalent_map = np.zeros((n_atoms, n_atoms), dtype=int)
    for bond in data.bonds:
        covalent_map[bond.atom1, bond.atom2] = 1
        covalent_map[bond.atom2, bond.atom1] = 1
    for n_curr in range(1, max_neighbor):
        for i in range(n_atoms):
            # current neighbors
            j_list = np.where(
                np.logical_and(covalent_map[i] <= n_curr,
                               covalent_map[i] > 0))[0]
            for j in j_list:
                k_list = np.where(covalent_map[j] == 1)[0]
                for k in k_list:
                    if k != i and k not in j_list:
                        covalent_map[i, k] = n_curr + 1
                        covalent_map[k, i] = n_curr + 1
    return jnp.array(covalent_map)


class Hamiltonian:
    # 存Residue templates
    # 存Generators

    def __init__(self, xmlfiles, operators=["res_template"]):
        self._xmlio = XMLIO()
        self.generators = {}
        self.templates = []
        self.paramset = ParamSet()
        self.operators = []

        for xml in xmlfiles:
            self._xmlio.loadXML(xml)
        ffinfo = self._xmlio.parseXML()
        self.ffinfo = ffinfo

        # 处理Forces
        for key in ffinfo["Forces"].keys():
            if key not in dmff_generators:
                raise BaseException(f"Generator {key} is not implemented.")
            self.generators[key] = dmff_generators[key](
                ffinfo["Forces"], self.paramset)

        self.parameters = self.paramset.parameters

        # initialize operators
        for tp in operators:
            if tp not in dmff_operators:
                raise DMFFException(f"Operator {tp} is not loaded.")
            self.operators.append(dmff_operators[tp](self.ffinfo))

    def prepTopData(self, topo: app.Topology, operators=["template"]) -> TopologyData:
        # prepare TopData
        if isinstance(operators, str):
            topdata = TopologyData(topo, default_typifier=operators)
        elif isinstance(operators, dict):
            topdata = TopologyData(topo)
            for key in operators.keys():
                for rid in operators[key]:
                    topdata.setOperatorToResidue(rid, key)

        # patch atoms using typifiers
        for operator in self.operators:
            if operator.name not in dmff_operators:
                raise DMFFException(f"Typifier {operator.name} is not loaded.")
            operator.typification(topdata)
        return topdata

    def buildJaxPotential(self, topdata: TopologyData, forces=None):
        efuncs = {}
        for gen in self.generators:
            if forces is not None and gen.getName() not in forces:
                continue
            efuncs[gen.getName()] = gen.createJaxFunc(topdata)

        def efunc_total(pos, box, pairs, prms):
            return sum([e(pos, box, pairs, prms) for e in efuncs.items()])

        return efunc_total

    def createJaxPotential(self, topo: app.Topology, operators=["template"], forces=None):
        topdata = self.prepTopData(topo, operators)
        efunc = self.buildJaxPotential(topdata, forces)
        return efunc

    def renderXML(self, out: str, residues=True, atomtypes=True, forces=True):
        for key in self.generators.keys():
            self.generators[key].updateParameters(self.ffinfo, self.paramset)

        self._xmlio.writeXML(
            out,
            residues=self.residues if residues else [],
            atomtypes=self.atomtypes if atomtypes else [],
            forces=self.ffinfo["Forces"] if forces else {}
        )
