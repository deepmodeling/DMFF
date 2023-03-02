import openmm.app as app
from typing import List, Union, Tuple


class VSite:

    def __init__(self, atoms: List[app.topology.Atom], weights: List[float], vatom: Union[app.topology.Atom, None] = None, name: str = None):
        self.atoms = atoms
        self.weights = weights
        self.vatom = vatom
        self.vsname = name


class TwoParticleAverageSite(VSite):

    def __init__(self, atoms: List[app.topology.Atom], weights: List[float], vatom: Union[app.topology.Atom, None] = None, name: str = None):
        super().__init__(atoms, weights, vatom)
        self.name = "two-particle-average"
        self.vsname = name


class ThreeParticleAverageSite(VSite):

    def __init__(self, atoms: List[app.topology.Atom], weights: List[float], vatom: Union[app.topology.Atom, None] = None, name: str = None):
        super().__init__(atoms, weights, vatom)
        self.name = "three-particle-average"
        self.vsname = name


class OutOfPlaneSite(VSite):

    def __init__(self, atoms: List[app.topology.Atom], weights: List[float], vatom: Union[app.topology.Atom, None] = None, name: str = None):
        super().__init__(atoms, weights, vatom)
        self.name = "out-of-plane"
        self.vsname = name
