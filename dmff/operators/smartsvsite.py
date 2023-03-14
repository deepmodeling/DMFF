from ..utils import DMFFException
from .base import BaseOperator
import xml.etree.ElementTree as ET
from ..api.topology import DMFFTopology
from ..api.vsite import VirtualSite
from ..api.vstools import insertVirtualSites
from rdkit import Chem


class SMARTSVSiteOperator(BaseOperator):
    
    def __init__(self, filename):
        self.infos = []
        if isinstance(filename, str):
            flist = [filename]
        else:
            flist = filename
        for fname in flist:
            root = ET.parse(fname).getroot()
            for child in root:
                if child.tag == "VirtualSite":
                    self.infos.append(child.attrib)

    def operate(self, topdata: DMFFTopology, **kwargs) -> DMFFTopology:
        vslist = []
        topatoms = [a for a in topdata.atoms()]
        for rdmol in topdata.molecules():
            Chem.SanitizeMol(rdmol)
            atoms = rdmol.GetAtoms()
            for info in self.infos:
                parser = info["smarts"]
                par = Chem.MolFromSmarts(parser)
                matches = rdmol.GetSubstructMatches(par)
                for match in matches:
                    if info["vtype"] == "average2":
                        a1idx, a2idx = match[0], match[1]
                        idx1 = int(atoms[a1idx].GetProp("_Index"))
                        idx2 = int(atoms[a2idx].GetProp("_Index"))
                        a1 = topatoms[idx1]
                        a2 = topatoms[idx2]
                        w1, w2 = float(info["weight1"]), float(info["weight2"])
                        meta = {}
                        meta["type"] = info["type"]
                        meta["class"] = info["class"]
                        vsite = VirtualSite(info["vtype"], [a1, a2], [w1, w2], meta=meta)
                        vslist.append(vsite)
        return insertVirtualSites(topdata, vslist)