import openmm.app as app
from typing import List, Union, Tuple


class VSite:

    def __init__(self, atoms: List[app.topology.Atom], weights: List[float], vatom: Union[app.topology.Atom, None] = None):
        self.atoms = atoms
        self.weights = weights
        self.vatom = vatom


class TwoParticleAverageSite(VSite):

    def __init__(self, atoms: List[app.topology.Atom], weights: List[float], vatom: Union[app.topology.Atom, None] = None):
        super().__init__(atoms, weights, vatom)
        self.name = "two-particle-average"


class ThreeParticleAverageSite(VSite):

    def __init__(self, atoms: List[app.topology.Atom], weights: List[float], vatom: Union[app.topology.Atom, None] = None):
        super().__init__(atoms, weights, vatom)
        self.name = "three-particle-average"


class OutOfPlaneSite(VSite):

    def __init__(self, atoms: List[app.topology.Atom], weights: List[float], vatom: Union[app.topology.Atom, None] = None):
        super().__init__(atoms, weights, vatom)
        self.name = "out-of-plane"