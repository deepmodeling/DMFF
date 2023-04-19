from .base import BaseOperator
from ..api.xmlio import XMLIO
from ..api.topology import DMFFTopology
from ..utils import DMFFException
from openmm.app import Topology
from typing import List
from rdkit import Chem


class SMARTSATypeOperator(BaseOperator):

    def __init__(self, ffinfo):
        self.name = "smarts"
        self.parsers = []
        self.atypes = []
        for atom in ffinfo["AtomTypes"]:
            if "smarts" in atom:
                key = "smarts"
            elif "smirks" in atom:
                key = "smirks"
            else:
                continue
            parser = atom[key]
            nm = atom["name"]
            cls = atom["class"] if "class" in atom else nm
            atype = (nm, cls, atom["element"])
            self.parsers.append(parser)
            self.atypes.append(atype)

    def operate(self, topdata: DMFFTopology, resname: List[str] = [], **kwargs) -> DMFFTopology:
        atoms = [a for a in topdata.atoms()]
        for rdmol in topdata.molecules():
            Chem.SanitizeMol(rdmol)
        for nparser, parser in enumerate(self.parsers):
            name, cls, elem = self.atypes[nparser]
            matches = topdata.parseSMARTS(parser, resname=resname)
            for match in matches:
                atoms[match[0]].meta["type"] = name
                atoms[match[0]].meta["class"] = cls
        for atom in topdata.atoms():
            if "type" not in atom.meta:
                atom.meta["type"] = None
                atom.meta["class"] = None
        return topdata
                