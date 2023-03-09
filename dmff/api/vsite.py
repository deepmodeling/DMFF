import openmm.app as app
from typing import List, Union, Tuple


class VirtualSite:

    def __init__(self, vtype, atoms, weights, vatom=None, meta={}):
        self.type = vtype
        self.atoms = atoms
        self.weights = weights
        self.vatom = vatom
        self.meta = meta

    def __deepcopy__(self, memo):
        return VirtualSite(self.atoms, self.weights, vatom=self.vatom, meta=self.meta)

    def __repr__(self):
        s = f"Virtual site type: {self.type} with \natoms: "
        for atom in self.atoms:
            s += f"{atom} "
        s += f"\nweights: "
        for weight in self.weights:
            s += f"{weight} "
        return s