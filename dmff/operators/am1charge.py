from ..utils import DMFFException
from .base import BaseOperator
from ..api.topology import DMFFTopology
import os


def cmd_exists(x):
    return any(os.access(os.path.join(path, x), os.X_OK)
               for path in os.environ["PATH"].split(os.pathsep))


class AM1ChargeOperator(BaseOperator):

    def __init__(self, sqm="sqm"):
        self.find_sqm = cmd_exists(sqm)
        self.sqm = sqm

    def operate(self, topdata: DMFFTopology) -> DMFFTopology:
        for atom in topdata.atoms():
            atom.meta["charge"] = 0.0

        for rdmol in topdata.molecules():
            pass
        return topdata
