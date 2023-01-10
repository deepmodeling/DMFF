try:
    import openmm.app as app
    import openmm.unit as unit
except ImportError as e:
    import simtk.openmm.app as app
    import simtk.unit as unit
from typing import Dict, Tuple, List
import numpy as np


class AtomType:

    def __init__(self, atype, aclass):
        self.atype = atype
        self.aclass = aclass


class TopologyData:

    class _Bond:

        def __init__(self, atom1, atom2):
            if atom1 < atom2:
                self.atom1 = atom1
                self.atom2 = atom2
            else:
                self.atom1 = atom2
                self.atom2 = atom1

        def get_another(self, atom):
            if self.atom1 == atom:
                return self.atom2
            elif self.atom2 == atom:
                return self.atom1
            return None

        def __hash__(self):
            return hash((self.atom1, self.atom2))

        def __eq__(self, other):
            return other.atom1 == self.atom1 and other.atom2 == self.atom2

        @classmethod
        def generate_indices(cls, bonds):
            return np.array([[a.atom1, a.atom2] for a in bonds])

    class _Angle:

        def __init__(self, atom1, atom2, atom3):
            self.atom2 = atom2
            if atom1 < atom3:
                self.atom1 = atom1
                self.atom3 = atom3
            else:
                self.atom1 = atom3
                self.atom3 = atom1

        def __hash__(self):
            return hash((self.atom1, self.atom2, self.atom3))

        def __eq__(self, other):
            return other.atom1 == self.atom1 and other.atom2 == self.atom2 and other.atom3 == self.atom3

        @classmethod
        def generate_indices(cls, angles):
            return np.array([[a.atom1, a.atom2, a.atom3] for a in angles])

    class _Proper:

        def __init__(self, atom1, atom2, atom3, atom4):
            if atom2 < atom3:
                self.atom1 = atom1
                self.atom2 = atom2
                self.atom3 = atom3
                self.atom4 = atom4
            else:
                self.atom1 = atom4
                self.atom2 = atom3
                self.atom3 = atom2
                self.atom4 = atom1

        def __hash__(self):
            return hash((self.atom1, self.atom2, self.atom3, self.atom4))

        def __eq__(self, other):
            return other.atom1 == self.atom1 and other.atom2 == self.atom2 and other.atom3 == self.atom3 and other.atom4 == self.atom4

        @classmethod
        def generate_indices(cls, propers):
            return np.array([[a.atom1, a.atom2, a.atom3, a.atom4] for a in propers])

    def __init__(self, topology: app.Topology) -> None:
        self.topo = topology
        self.atomtypes = []
        self.bonds = []
        self._bondOnAtom = []
        for na in range(topology.getNumAtoms()):
            self._bondOnAtom.append([])
        self.bond_indices = None
        self.angles = []
        self.angle_indices = None

        # initialize bond
        for nbond, bond in enumerate(topology.bonds()):
            i1, i2 = bond[0].index, bond[1].index
            self.bonds.append(self._Bond(i1, i2))
            self._bondOnAtom[i1].append(nbond)
            self._bondOnAtom[i2].append(nbond)
        self.bond_indices = self._Bond.generate_indices(self.bonds)

        # initialize angle
        unique_angles = set()
        for iatom in range(topology.getNumAtoms()):
            bonds_on_iatom = self._bondOnAtom[iatom]
            if len(bonds_on_iatom) > 1:
                for n1 in range(len(bonds_on_iatom)):
                    b1 = self.bonds[bonds_on_iatom[n1]]
                    for n2 in range(n1+1, len(bonds_on_iatom)):
                        b2 = self.bonds[bonds_on_iatom[n2]]
                        angle_i1 = b1.get_another(iatom)
                        angle_i2 = iatom
                        angle_i3 = b2.get_another(iatom)
                        if angle_i1 >= 0 and angle_i3 >= 0:
                            unique_angles.add(self._Angle(
                                angle_i1, angle_i2, angle_i3))
        self.angles = list(unique_angles)
        self.angle_indices = self._Angle.generate_indices(self.angles)

        # initialize proper

    def detect_improper(self):
        pass
