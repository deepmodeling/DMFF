from ..utils import DMFFException
from .base import BaseOperator
from ..api.topology import DMFFTopology
import subprocess
try:
    from rdkit.Chem import AllChem
    from rdkit import Chem
except ImportError:
    import warnings
    warnings.warn("AM1ChargeOperator requires rdkit and cannot be used.")
import os


def cmd_exists(x):
    return any(os.access(os.path.join(path, x), os.X_OK)
               for path in os.environ["PATH"].split(os.pathsep))


class AM1ChargeOperator(BaseOperator):

    def __init__(self, ffinfo: dict):
        sqm = ffinfo["Operators"]["AM1ChargeOperator"][0]["attrib"]["path"]
        self.find_sqm = cmd_exists(sqm)
        self.sqm = sqm

    def operate(self, topdata: DMFFTopology, tot_charge=0, **kwargs) -> DMFFTopology:
        atoms = [a for a in topdata.atoms()]
        for atom in atoms:
            atom.meta["charge"] = 0.0

        for rdmol in topdata.molecules():
            topdata.regularize_aromaticity(rdmol)
            AllChem.EmbedMolecule(rdmol, randomSeed=10)
            conf = rdmol.GetConformer()
            pos = conf.GetPositions()
            with open("tmp.in", "w") as f:
                f.write(f" Run semi-empirical minimization\n")
                f.write(f" &qmmm\n")
                f.write(f" qm_theory='AM1', qmcharge={tot_charge},\n")
                f.write(f" /\n")
                real_indices = []
                for natom, atom in enumerate(rdmol.GetAtoms()):
                    anum = atom.GetAtomicNum()
                    aname = atom.GetProp("_Name")
                    if anum > 0:
                        x, y, z = pos[natom]
                        f.write(f" {anum} {aname} {x:16.8f} {y:16.8f} {z:16.8f}\n")
                        real_indices.append(natom)
            subprocess.run([self.sqm, "-O", '-i', 'tmp.in', '-o', 'tmp.out'])
            with open('tmp.out', "r") as f:
                text = f.readlines()
            line_with_text = []
            for nline, line in enumerate(text):
                if "Mulliken Charge" in line:
                    line_with_text.append(nline)
            if len(line_with_text) != 2:
                raise DMFFException("AM1 charge error.")
            start, end = line_with_text
            ratoms = [a for a in rdmol.GetAtoms()]
            for nreal, ireal in enumerate(real_indices):
                charge = float(text[start+nreal+1].strip().split()[-1])
                idx = int(ratoms[ireal].GetProp("_Index"))
                atoms[idx].meta["charge"] = charge
            eqv_info = topdata.getEquivalentAtoms()
            finished_atoms = []
            for atom in topdata.atoms():
                aidx = atom.index
                if aidx in finished_atoms:
                    continue
                aeq = eqv_info[aidx]
                average_charge = sum([atoms[i].meta["charge"] for i in aeq]) / len(aeq)
                for i in aeq:
                    atoms[i].meta["charge"] = average_charge
                    finished_atoms.append(i)
            os.system("rm tmp.in tmp.out")
        return topdata
