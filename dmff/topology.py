try:
    import openmm.app as app
    import openmm.unit as unit
except ImportError as e:
    import simtk.openmm.app as app
    import simtk.unit as unit
from typing import Dict, Tuple, List
import numpy as np
import itertools
import networkx as nx
from networkx.algorithms import isomorphism
from .utils import DMFFException


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

        def __contains__(self, item):
            return item == self.atom1 or item == self.atom2 or item == self.atom3

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

    class _Improper:

        def __init__(self, atom1, atom2, atom3, atom4):
            self.atom1 = atom1
            a2, a3, a4 = sorted([atom2, atom3, atom4])
            self.atom2 = a2
            self.atom3 = a3
            self.atom4 = a4

        def __hash__(self):
            return hash((self.atom1, self.atom2, self.atom3, self.atom4))

        def __eq__(self, other):
            return other.atom1 == self.atom1 and other.atom2 == self.atom2 and other.atom3 == self.atom3 and other.atom4 == self.atom4

        @classmethod
        def generate_indices(cls, imprs):
            return np.array([[a.atom1, a.atom2, a.atom3, a.atom4] for a in imprs])

    def __init__(self, topology: app.Topology) -> None:
        self.topo = topology
        self.natoms = topology.getNumAtoms()
        self.atoms = [a for a in topology.atoms()]

        self.residues = [r for r in topology.residues()]
        self.residue_indices = []
        for r in self.residues:
            self.residue_indices.append([a.index for a in r.atoms()])

        self._bondedAtom = []
        for na in range(topology.getNumAtoms()):
            self._bondedAtom.append([])

        self.properties = []
        for na in range(topology.getNumAtoms()):
            self.properties.append(None)

        # initialize bond
        unique_bonds = set()
        for nbond, bond in enumerate(topology.bonds()):
            i1, i2 = bond[0].index, bond[1].index
            unique_bonds.add(self._Bond(i1, i2))
            self._bondedAtom[i1].append(i2)
            self._bondedAtom[i2].append(i1)
        self.bonds = list(unique_bonds)
        self.bond_indices = self._Bond.generate_indices(self.bonds)

        # initialize angle
        unique_angles = set()
        for iatom in range(topology.getNumAtoms()):
            bonded_atoms = self._bondedAtom[iatom]
            angle_i2 = iatom
            if len(bonded_atoms) > 1:
                for n1 in range(len(bonded_atoms)):
                    angle_i1 = bonded_atoms[n1]
                    for n2 in range(n1+1, len(bonded_atoms)):
                        angle_i3 = bonded_atoms[n2]
                        unique_angles.add(self._Angle(
                            angle_i1, angle_i2, angle_i3))
        self.angles = list(unique_angles)
        self.angle_indices = self._Angle.generate_indices(self.angles)

        # initialize proper
        unique_propers = set()
        for angle in self.angles:
            for atom in self._bondedAtom[angle.atom1]:
                if atom not in angle:
                    unique_propers.add(self._Proper(
                        atom, angle.atom1, angle.atom2, angle.atom3))
            for atom in self._bondedAtom[angle.atom3]:
                if atom not in angle:
                    unique_propers.add(self._Proper(
                        atom, angle.atom3, angle.atom2, angle.atom1))
        self.propers = list(unique_propers)
        self.proper_indices = self._Proper.generate_indices(self.propers)

        # initialize improper
        self.impropers = []
        self.improper_indices = np.ndarray([], dtype=int)

    def detect_impropers(self):
        unique_impropers = set()
        for atom in range(self.natoms):
            bonded_atoms = self._bondedAtom[atom]
            if len(bonded_atoms) > 2:
                for subset in itertools.combinations(bonded_atoms, 3):
                    unique_impropers.add(self._Improper(atom, *subset))
        self.impropers = list(unique_impropers)
        self.improper_indices = self._Improper.generate_indices(self.impropers)

    def generate_residue_graph(self, resid):
        residue_indices = self.residue_indices[resid]
        graph = nx.Graph()
        # add nodes
        for atom in residue_indices:
            external_bond = False
            bonded_atoms = self._bondedAtom[atom]
            for bonded in bonded_atoms:
                if bonded not in residue_indices:
                    external_bond = True
            graph.add_node(
                atom, name=self.atoms[atom].name, element=self.atoms[atom].element.symbol, external_bond=external_bond)
            for bonded in bonded_atoms:
                if bonded < atom and bonded in residue_indices:
                    graph.add_edge(atom, bonded)
        return graph

    @classmethod
    def matchTemplate(cls, graph, template):
        if graph.number_of_nodes() != template.number_of_nodes():
            return False, {}

        def match_func(n1, n2):
            return n1["element"] == n2["element"] and n1["external_bond"] == n2["external_bond"]
        matcher = isomorphism.GraphMatcher(
            graph, template, node_match=match_func)
        is_matched = matcher.is_isomorphic()
        if is_matched:
            match_dict = [i for i in matcher.match()][0]
            atype_dict = {}
            for key in match_dict.keys():
                attrib = {k: v for k, v in template.nodes(
                )[match_dict[key]].items() if k != "name"}
                atype_dict[key] = attrib
        else:
            atype_dict = {}
        return is_matched, atype_dict

    def match_all(self, templates):
        for ires in range(len(self.residues)):
            graph = self.generate_residue_graph(ires)
            all_fail = True
            for template in templates:
                is_matched, atype_dict = self.matchTemplate(graph, template)
                if is_matched:
                    all_fail = False
                    for key in atype_dict.keys():
                        if self.properties[key] is not None:
                            raise DMFFException(
                                f"Atom {key} is matched twice.")
                        self.properties[key] = atype_dict[key]
                    break
            if all_fail:
                raise DMFFException(
                    f"No template can match residue {self.residues[ires]}.")
