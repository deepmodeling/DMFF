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
from ..utils import DMFFException, _standardResidues, dict_to_jnp
from rdkit import Chem
from rdkit.Chem import AllChem
import jax.numpy as jnp


def top2graph(top: app.Topology) -> nx.Graph:
    g = nx.Graph()
    for na, a in enumerate(top.atoms()):
        elem = a.element.symbol if a.element is not None else "none"
        g.add_node(a.index, index=a.index, name=a.name, element=elem,
                   residx=a.residue.index, resname=a.residue.name)
    for nb, b in enumerate(top.bonds()):
        g.add_edge(b.atom1.index, b.atom2.index)
    return g


def decompgraph(graph: nx.Graph) -> List[nx.Graph]:
    nsub = [graph.subgraph(indices)
            for indices in nx.connected_components(graph)]
    return nsub


def graph2top(graph: nx.Graph) -> app.Topology:
    nodes = [graph.nodes[n] for n in graph.nodes]
    nodes = sorted(nodes, key=lambda x: x["index"])
    atoms = []
    node2atom = {}

    top = app.Topology()
    chain = top.addChain("A")
    rid = nodes[0]["residx"]
    rname = nodes[0]["resname"]
    res = top.addResidue(rname, chain)
    for node in nodes:
        if node["residx"] != rid:
            rname = node["resname"]
            res = top.addResidue(rname, chain)
            rid = node["residx"]
        elem = app.element.get_by_symbol(node["element"]) if node["element"] != "none" else None
        atom = top.addAtom(
            node["name"], elem, res)
        atoms.append(atom)
        node2atom[node["index"]] = atom

    for b1, b2 in graph.edges.keys():
        n1, n2 = graph.nodes[b1]["index"], graph.nodes[b2]["index"]
        a1, a2 = node2atom[n1], node2atom[n2]
        top.addBond(a1, a2)

    return top


def top2rdmol(top: app.Topology) -> Chem.rdchem.Mol:
    rdmol = Chem.Mol()
    emol = Chem.EditableMol(rdmol)
    idx2ridx = {}
    for na, atm in enumerate(top.atoms()):
        if atm.element is None:
            continue
        ratm = Chem.Atom(atm.element.atomic_number)
        ratm.SetProp("_Name", atm.name)
        ratm.SetProp("_Index", f"{atm.index}")
        ratm.SetProp("_ResIndex", f"{atm.residue.index}")
        ratm.SetProp("_ResName", atm.residue.name)
        emol.AddAtom(ratm)
        idx2ridx[atm.index] = na
    for bnd in top.bonds():
        if bnd.type is None:
            if bnd.order is None:
                order = 1
            else:
                order = bnd.order
        else:
            if isinstance(bnd.type, app.topology.Single):
                order = 1
            elif isinstance(bnd.type, app.topology.Double):
                order = 2
            elif isinstance(bnd.type, app.topology.Triple):
                order = 3
            elif isinstance(bnd.type, app.topology.Aromatic) or isinstance(bnd.type, app.topology.Amide):
                order = 1.5
        emol.AddBond(idx2ridx[bnd.atom1.index],
                     idx2ridx[bnd.atom2.index], Chem.BondType(order))
    rdmol = emol.GetMol()
    # rdmol.UpdatePropertyCache()
    #AllChem.EmbedMolecule(rdmol, randomSeed=1)
    return rdmol


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


class TopologyData:

    def __init__(self, topology: app.Topology) -> None:
        self.topology = topology
        self.natoms = topology.getNumAtoms()
        self.atoms = [a for a in topology.atoms()]
        self.atom_meta = []
        for atom in self.atoms:
            self.atom_meta.append({
                "element": atom.element.symbol if atom.element is not None else "none",
                "operator": []
            })

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
            unique_bonds.add(_Bond(i1, i2))
            self._bondedAtom[i1].append(i2)
            self._bondedAtom[i2].append(i1)
        self.bonds = list(unique_bonds)
        self.bond_indices = _Bond.generate_indices(self.bonds)

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
                        unique_angles.add(_Angle(
                            angle_i1, angle_i2, angle_i3))
        self.angles = list(unique_angles)
        self.angle_indices = _Angle.generate_indices(self.angles)

        # initialize proper
        unique_propers = set()
        for angle in self.angles:
            for atom in self._bondedAtom[angle.atom1]:
                if atom not in angle:
                    unique_propers.add(_Proper(
                        atom, angle.atom1, angle.atom2, angle.atom3))
            for atom in self._bondedAtom[angle.atom3]:
                if atom not in angle:
                    unique_propers.add(_Proper(
                        atom, angle.atom3, angle.atom2, angle.atom1))
        self.propers = list(unique_propers)
        self.proper_indices = _Proper.generate_indices(self.propers)

        # initialize improper
        self.impropers = []
        self.improper_indices = np.ndarray([], dtype=int)
        self.detect_impropers()

        self.updateRDMol()

        self.vsite = {}
        self.vsite["two_point_average"] = {
            "index": [],
            "weight": [],
            "vsite": []
        }
        self.vsite["three_point_average"] = {
            "index": [],
            "weight": [],
            "vsite": []
        }
        self.vsite["out_of_plane"] = {
            "index": [],
            "weight": [],
            "vsite": []
        }

    def updateRDMol(self):
        # decompose topology to rdmol
        self.rdmols = [top2rdmol(graph2top(g))
                       for g in decompgraph(top2graph(self.topology))]

    def detect_impropers(self, detect_aromatic_only=False):
        unique_impropers = set()
        for atom in range(self.natoms):
            bonded_atoms = self._bondedAtom[atom]
            if not detect_aromatic_only:
                if len(bonded_atoms) > 2:
                    for subset in itertools.combinations(bonded_atoms, 3):
                        unique_impropers.add(_Improper(atom, *subset))
            else:
                if len(bonded_atoms) == 3:
                    unique_impropers.add(_Improper(atom, *bonded_atoms))
        self.impropers = list(unique_impropers)
        self.improper_indices = _Improper.generate_indices(self.impropers)

    def setOperatorToResidue(self, residue_id, operator_name):
        for iatom in self.residue_indices[residue_id]:
            self.atom_meta[iatom]["operator"].append(operator_name)

    def addVirtualSiteList(self, vslist: List):
        two_ave_idx, two_ave_weight = [], []
        two_ave_vsites = []
        three_ave_idx, three_ave_weight = [], []
        three_ave_vsites = []
        out_of_plane_idx, out_of_plane_weight = [], []
        out_of_plane_vsites = []

        from .vstools import TwoParticleAverageSite, ThreeParticleAverageSite, OutOfPlaneSite, VSite
        for vsite in vslist:
            if isinstance(vsite, TwoParticleAverageSite):
                a1, a2 = vsite.atoms
                w1, w2 = vsite.weights
                two_ave_idx.append([a1.index, a2.index])
                two_ave_weight.append([w1, w2])
            elif isinstance(vsite, ThreeParticleAverageSite):
                a1, a2, a3 = vsite.atoms
                w1, w2, w3 = vsite.weights
                three_ave_idx.append([a1.index, a2.index, a3.index])
                three_ave_weight.append([w1, w2, w3])
            elif isinstance(vsite, OutOfPlaneSite):
                a1, a2, a3 = vsite.atoms
                w1, w2, w3 = vsite.weights
                out_of_plane_idx.append([a1.index, a2.index, a3.index])
                out_of_plane_weight.append([w1, w2, w3])

        self.vsite["two_point_average"]["index"] = np.array(
            two_ave_idx, dtype=int)
        self.vsite["two_point_average"]["weight"] = np.array(
            two_ave_weight, dtype=float)
        self.vsite["two_point_average"]["vsite"] = np.array(
            two_ave_vsites, dtype=int)
        self.vsite["three_point_average"]["index"] = np.array(
            three_ave_idx, dtype=int)
        self.vsite["three_point_average"]["weight"] = np.array(
            three_ave_weight, dtype=float)
        self.vsite["three_point_average"]["vsite"] = np.array(
            three_ave_vsites, dtype=int)
        self.vsite["out_of_plane"]["index"] = np.array(
            out_of_plane_idx, dtype=int)
        self.vsite["out_of_plane"]["weight"] = np.array(
            out_of_plane_weight, dtype=float)
        self.vsite["out_of_plane"]["vsite"] = np.array(
            out_of_plane_vsites, dtype=int)

        self.vsite = dict_to_jnp(self.vsite)

    def generateVSiteUpdateFunction(self):

        def vsite_update(self, positions):
            # 2 site
            newpos_2_site_p1 = positions[self.vsite["two_point_average"]
                                         ["index"][:, 0], :]
            newpos_2_site_p2 = positions[self.vsite["two_point_average"]
                                         ["index"][:, 1], :]
            newpos = newpos_2_site_p1 * self.vsite["two_point_average"]["weight"][:,
                                                                                  0] + newpos_2_site_p2 * self.vsite["two_point_average"]["weight"][:, 1]
            positions.at[self.vsite["two_point_average"]
                         ["vsite"], :].set(newpos)
            # 3 site
            newpos_3_site_p1 = positions[self.vsite["three_point_average"]
                                         ["index"][:, 0], :]
            newpos_3_site_p2 = positions[self.vsite["three_point_average"]
                                         ["index"][:, 1], :]
            newpos_3_site_p3 = positions[self.vsite["three_point_average"]
                                         ["index"][:, 2], :]
            newpos = newpos_3_site_p1 * self.vsite["three_point_average"]["weight"][:, 0] + newpos_3_site_p2 * \
                self.vsite["three_point_average"]["weight"][:, 1] + \
                newpos_3_site_p3 * \
                self.vsite["three_point_average"]["weight"][:, 2]
            positions.at[self.vsite["three_point_average"]
                         ["vsite"], :].set(newpos)
            # out of plane
            newpos_op_p1 = positions[self.vsite["out_of_plane"]
                                     ["index"][:, 0], :]
            newpos_op_p2 = positions[self.vsite["out_of_plane"]
                                     ["index"][:, 1], :]
            newpos_op_p3 = positions[self.vsite["out_of_plane"]
                                     ["index"][:, 2], :]
            r12 = newpos_op_p2 - newpos_op_p1
            r13 = newpos_op_p3 - newpos_op_p1
            rcross = jnp.cross(r12, r13, axisa=1, axisb=1, axisc=1)
            newpos = newpos_op_p1 + self.vsite["out_of_plane"]["weight"][:, 0] * r12 + \
                self.vsite["out_of_plane"]["weight"][:, 1] * r13 + \
                self.vsite["out_of_plane"]["weight"][:, 2] * rcross
            positions.at[self.vsite["out_of_plane"]
                         ["vsite"], :].set(newpos)

            return positions

        return vsite_update

    def getAtomIndices(self):
        return [a.index for a in self.topology.atoms() if a.element is not None]