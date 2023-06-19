from ..utils import DMFFException
from .base import BaseOperator
import xml.etree.ElementTree as ET
from ..api.topology import DMFFTopology
from ..api.vsite import VirtualSite
from ..api.vstools import insertVirtualSites
from rdkit import Chem


class ParmedOperator(BaseOperator):

    def __init__(self, ffinfo):
        self.infos = []
        self.atypes = []
        self.params = []

    def operate(self, topdata: DMFFTopology, **kwargs) -> DMFFTopology:
        gmx_top = kwargs["gmx_top"]
        # 从parmed对象里拿到atomtype
        # 交给topology
        # 存在里边
        return topdata
    
    def getInitParameters(self):
        ret = []
        for na in range(len(self.atypes)):
            ret.append([self.atypes[na], self.params[na][0], self.params[na][1]])

    @classmethod
    def overwriteLennardJones(cls, top, prm):
        # 把parameter写到parmed top里去
        pass