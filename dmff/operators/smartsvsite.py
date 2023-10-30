from ..utils import DMFFException
from .base import BaseOperator
import xml.etree.ElementTree as ET
from ..api.topology import DMFFTopology
from ..api.vsite import VirtualSite
from ..api.vstools import insertVirtualSites
try:
    from rdkit import Chem
except ImportError:
    import warnings
    warnings.warn("WARNING: rdkit not installed, SMARTSVSiteOperator will not work.")


class SMARTSVSiteOperator(BaseOperator):

    def __init__(self, ffinfo):
        self.infos = []
        for vsite in ffinfo["Operators"]["SMARTSVSiteOperator"]:
            self.infos.append(vsite["attrib"])

    def operate(self, topdata: DMFFTopology, **kwargs) -> DMFFTopology:
        vslist = []
        topatoms = [a for a in topdata.atoms()]
        for rdmol in topdata.molecules():
            atoms = rdmol.GetAtoms()
            for info in self.infos:
                parser = info["smarts"] if "smarts" in info else info["smirks"]
                par = Chem.MolFromSmarts(parser)
                matches = rdmol.GetSubstructMatches(par)
                for match in matches:
                    alist = []
                    for molidx in match:
                        idx = int(atoms[molidx].GetProp("_Index"))
                        atom = topatoms[idx]
                        alist.append(atom)
                    wlist = []
                    widx = 1
                    while True:
                        key = f"weight{widx}"
                        if key not in info:
                            break
                        wlist.append(float(info[key]))
                        widx += 1
                    meta = {}
                    meta["type"] = info["type"]
                    meta["class"] = info["class"]
                    meta["charge"] = 0.0
                    vsite = VirtualSite(
                        info["vtype"], alist, wlist, meta=meta)
                    vslist.append(vsite)
        return insertVirtualSites(topdata, vslist)
