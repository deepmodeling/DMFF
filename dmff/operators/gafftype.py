from ..utils import DMFFException
from .base import BaseOperator
from ..api.topology import DMFFTopology
import subprocess
<<<<<<< HEAD
from rdkit.Chem import AllChem
from rdkit import Chem
=======
try:
    from rdkit.Chem import AllChem
    from rdkit import Chem
except ImportError:
    import warnings
    warnings.warn("GAFFTypeOperator requires rdkit and cannot be used.")
>>>>>>> upstream/devel
from .am1charge import cmd_exists
import shutil
import os


class GAFFTypeOperator(BaseOperator):

    def __init__(self, ffinfo: dict):
        print(ffinfo["Operators"]["GAFFTypeOperator"])
        antechamber = ffinfo["Operators"]["GAFFTypeOperator"][0]["attrib"]["path"]
        self.find_antechamber = cmd_exists(antechamber)
        self.antechamber = antechamber

    def operate(self, topdata: DMFFTopology, **kwargs) -> DMFFTopology:
        atoms = [a for a in topdata.atoms()]

        for rdmol in topdata.molecules():
            aidx = {}
            for na, atom in enumerate(rdmol.GetAtoms()):
                aidx[na] = int(atom.GetProp("_Index"))
            topdata.regularize_aromaticity(rdmol)
            AllChem.EmbedMolecule(rdmol, randomSeed=1)
            conf = rdmol.GetConformer()
            Chem.MolToMolFile(rdmol, "_tmp.mol")
            subprocess.run([self.antechamber, '-i', '_tmp.mol', "-fi", "sdf", '-o', '_tmp.mol2', "-fo", "mol2", "-at", "gaff2", "-pf", "y"])
            with open("_tmp.mol2", "r") as f:
                text = [l.strip() for l in f if len(l.strip()) > 0]
                patom, pbond, pstruct = -1, -1, -1
                for nline, line in enumerate(text):
                    if "@<TRIPOS>ATOM" in line:
                        patom = nline
                    elif "@<TRIPOS>BOND" in line:
                        pbond = nline
                    elif "@<TRIPOS>SUBSTRUCTURE" in line:
                        pstruct = nline
                atypes = [l.strip().split()[5] for l in text[patom+1:pbond]]
                borders = []
                for line in text[pbond+1:pstruct]:
                    _, ii, jj, oo = line.strip().split()
                    borders.append([aidx[int(ii)-1], aidx[int(jj)-1], int(oo)])
                for nat, at in enumerate(atypes):
                    atoms[aidx[nat]].meta["type"] = at
                    atoms[aidx[nat]].meta["class"] = at
            os.system("rm _tmp.mol _tmp.mol2")
        for vsite in topdata.vsites():
            vidx = vsite.vatom.index
            atoms[vidx].meta["type"] = "vs"
            atoms[vidx].meta["class"] = "vs"
        return topdata