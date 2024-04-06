from ..utils import DMFFException
from .base import BaseOperator
import xml.etree.ElementTree as ET
from ..api.topology import DMFFTopology
from ..api.vsite import VirtualSite
from ..api.vstools import insertVirtualSites
<<<<<<< HEAD
from rdkit import Chem
import parmed
=======
try:
    from rdkit import Chem
    import parmed
except ImportError:
    import warnings
    warnings.warn("Parmed is not installed. Parmed optimization cannot be used.")
>>>>>>> upstream/devel


class ParmedLennardJonesOperator(BaseOperator):
    def __init__(self, ffinfo = None):
        self.infos = []
        self.atypes = []
        self.params = []

    def operate(self, topdata: DMFFTopology, gmx_top = None, **kwargs) -> DMFFTopology:
        prm_top = gmx_top
        atoms = [a for a in topdata.atoms()]
        for natom, atom in enumerate(prm_top.atoms):
            atype = atom.atom_type
            aname, sig, eps = atype.name, atype.sigma * 0.1, atype.epsilon * 4.184
            # print(aname, sig, eps)
            if aname not in self.atypes:
                self.atypes.append(aname)
                self.params.append((sig, eps))
            atoms[natom].meta["type"] = aname
            atoms[natom].meta["class"] = aname
        return topdata

    def getInitParameters(self):
        ret = {}
        for na in range(len(self.atypes)):
            ret[self.atypes[na]] = (self.params[na][0], self.params[na][1])
        return ret
    
    def renderLennardJonesXML(self, filename):
        prms = self.getInitParameters()
        with open(filename, "w") as f:
            f.write("<ForceField>\n")
            f.write('    <LennardJonesForce lj14scale="0.50000">\n')
            for atype in prms.keys():
                sig, eps = prms[atype]
                f.write(f'        <Atom epsilon="{eps}" sigma="{sig}" type="{atype}"/>\n')
            f.write("    </LennardJonesForce>\n")
            f.write("</ForceField>\n")

    @classmethod
<<<<<<< HEAD
    def overwriteLennardJones(cls, top: parmed.gromacs.GromacsTopologyFile, ffinfo: dict):
=======
    def overwriteLennardJones(cls, top, ffinfo: dict):
>>>>>>> upstream/devel
        nodes = [n for n in ffinfo["Forces"]["LennardJonesForce"]["node"]]
        prm = {}
        for node in nodes:
            if not node["name"] == "Atom":
                continue
            sig = float(node["attrib"]["sigma"])
            eps = float(node["attrib"]["epsilon"])
            prm[node["attrib"]["type"]] = (sig, eps)
        # 把parameter写到parmed top里去
        for atom in top.atoms:
            atype = atom.atom_type
            sig, eps = prm[atype.name]
            atype.sigma = sig * 10.
            atype.epsilon = eps / 4.184
            atype.sigma_14 = sig * 10.
            atype.epsilon_14 = eps / 4.184
        pass
