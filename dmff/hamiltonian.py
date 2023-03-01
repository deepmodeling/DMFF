import openmm.app as app
from dmff_new.xmlio import XMLIO
from dmff_new.paramset import ParamSet
from dmff_new.topology import TopologyData
from dmff_new.utils import DMFFException
import jax

dmff_generators = {}
dmff_operators = {}


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
