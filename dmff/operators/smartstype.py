from dmff.operators.base import BaseOperator
from dmff.topology import TopologyData, top2graph, decompgraph, graph2top, top2rdmol
from dmff.hamiltonian import dmff_operators
from dmff.utils import DMFFException
from openmm.app import Topology
from typing import List
from rdkit import Chem


class SMARTSOperator(BaseOperator):

    def __init__(self, ffinfo):
        self.name = "smarts"
        self.parsers = []
        self.atypes = []
        for atom in ffinfo["AtomTypes"]:
            if "smarts" in atom:
                parser = atom["smarts"]
                atype = (atom["name"], atom["class"], atom["element"])
                self.parsers.append(parser)
                self.atypes.append(atype)

    def operate(self, topdata: TopologyData):
        for rdmol in topdata.rdmols:
            is_smarts = True
            node_idx = []
            for atom in rdmol.GetAtoms():
                idx = int(atom.GetProp("_Index"))
                node_idx.append(idx)
                if self.name not in topdata.atom_meta[idx]["operator"]:
                    is_smarts = False
                    break
            if not is_smarts:
                continue
            Chem.SanitizeMol(rdmol)
            
            for nparser, parser in enumerate(self.parsers):
                name, cls, elem = self.atypes[nparser]
                par = Chem.MolFromSmarts(parser)
                matches = rdmol.GetSubstructMatches(par)
                for match in matches:
                    matchidx = node_idx[match[0]]
                    topdata.atom_meta[matchidx]["type"] = name
                    topdata.atom_meta[matchidx]["class"] = cls


dmff_operators["smarts"] = SMARTSOperator
