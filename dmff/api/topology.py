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
from .vsite import TwoParticleAverageSite, ThreeParticleAverageSite, OutOfPlaneSite, VSite
from .graph import top2graph, top2rdmol, graph2top, decompgraph, decomptop


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
        return np.array([[a.atom1, a.atom2, a.atom3, a.atom4]
                         for a in propers])


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
                "element":
                atom.element.symbol if atom.element is not None else "none",
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
                    for n2 in range(n1 + 1, len(bonded_atoms)):
                        angle_i3 = bonded_atoms[n2]
                        unique_angles.add(_Angle(angle_i1, angle_i2, angle_i3))
        self.angles = list(unique_angles)
        self.angle_indices = _Angle.generate_indices(self.angles)

        # initialize proper
        unique_propers = set()
        for angle in self.angles:
            for atom in self._bondedAtom[angle.atom1]:
                if atom not in angle:
                    unique_propers.add(
                        _Proper(atom, angle.atom1, angle.atom2, angle.atom3))
            for atom in self._bondedAtom[angle.atom3]:
                if atom not in angle:
                    unique_propers.add(
                        _Proper(atom, angle.atom3, angle.atom2, angle.atom1))
        self.propers = list(unique_propers)
        self.proper_indices = _Proper.generate_indices(self.propers)

        # initialize improper
        self.impropers = []
        self.improper_indices = np.ndarray([], dtype=int)
        self.detect_impropers()

        indices_decomp = decomptop(self.topology)
        self.rdmols = [top2rdmol(self.topology, ind) for ind in indices_decomp]
        self.atom2mol = {}
        self.res2mol = {}
        self.mol2res = {}
        self.mol2atom = {}
        for nmol, mol in enumerate(self.rdmols):
            resid = []
            atomid = []
            for atom in mol.GetAtoms():
                aid = int(atom.GetProp("_Index"))
                rid = int(atom.GetProp("_ResIndex"))
                self.atom2mol[aid] = nmol
                self.res2mol[rid] = nmol
                resid.append(rid)
                atomid.append(aid)
            self.mol2res[nmol] = list(set(resid))
            self.mol2atom[nmol] = list(set(atomid))

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
        self.vsite["out_of_plane"] = {"index": [], "weight": [], "vsite": []}

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

        for vsite in vslist:
            if isinstance(vsite, TwoParticleAverageSite):
                a1, a2 = vsite.atoms
                w1, w2 = vsite.weights
                two_ave_idx.append([a1.index, a2.index])
                two_ave_weight.append([w1, w2])
                two_ave_vsites.append(vsite.vatom.index)
            elif isinstance(vsite, ThreeParticleAverageSite):
                a1, a2, a3 = vsite.atoms
                w1, w2, w3 = vsite.weights
                three_ave_idx.append([a1.index, a2.index, a3.index])
                three_ave_weight.append([w1, w2, w3])
                three_ave_vsites.append(vsite.vatom.index)
            elif isinstance(vsite, OutOfPlaneSite):
                a1, a2, a3 = vsite.atoms
                w1, w2, w3 = vsite.weights
                out_of_plane_idx.append([a1.index, a2.index, a3.index])
                out_of_plane_weight.append([w1, w2, w3])
                out_of_plane_vsites.append(vsite.vatom.index)

        self.vsite["two_point_average"]["index"] = np.array(two_ave_idx,
                                                            dtype=int)
        self.vsite["two_point_average"]["weight"] = np.array(two_ave_weight,
                                                             dtype=float)
        self.vsite["two_point_average"]["vsite"] = np.array(two_ave_vsites,
                                                            dtype=int)
        self.vsite["three_point_average"]["index"] = np.array(three_ave_idx,
                                                              dtype=int)
        self.vsite["three_point_average"]["weight"] = np.array(
            three_ave_weight, dtype=float)
        self.vsite["three_point_average"]["vsite"] = np.array(three_ave_vsites,
                                                              dtype=int)
        self.vsite["out_of_plane"]["index"] = np.array(out_of_plane_idx,
                                                       dtype=int)
        self.vsite["out_of_plane"]["weight"] = np.array(out_of_plane_weight,
                                                        dtype=float)
        self.vsite["out_of_plane"]["vsite"] = np.array(out_of_plane_vsites,
                                                       dtype=int)

        self.vsite = dict_to_jnp(self.vsite)

    def generateVSiteUpdateFunction(self):
        def vsite_update(self, positions):
            # 2 site
            newpos_2_site_p1 = positions[
                self.vsite["two_point_average"]["index"][:, 0], :]
            newpos_2_site_p2 = positions[
                self.vsite["two_point_average"]["index"][:, 1], :]
            newpos = newpos_2_site_p1 * self.vsite["two_point_average"][
                "weight"][:, 0] + newpos_2_site_p2 * self.vsite[
                    "two_point_average"]["weight"][:, 1]
            positions.at[self.vsite["two_point_average"]["vsite"], :].set(
                newpos)
            # 3 site
            newpos_3_site_p1 = positions[
                self.vsite["three_point_average"]["index"][:, 0], :]
            newpos_3_site_p2 = positions[
                self.vsite["three_point_average"]["index"][:, 1], :]
            newpos_3_site_p3 = positions[
                self.vsite["three_point_average"]["index"][:, 2], :]
            newpos = newpos_3_site_p1 * self.vsite["three_point_average"]["weight"][:, 0] + newpos_3_site_p2 * \
                self.vsite["three_point_average"]["weight"][:, 1] + \
                newpos_3_site_p3 * \
                self.vsite["three_point_average"]["weight"][:, 2]
            positions.at[self.vsite["three_point_average"]["vsite"], :].set(
                newpos)
            # out of plane
            newpos_op_p1 = positions[self.vsite["out_of_plane"]["index"][:,
                                                                         0], :]
            newpos_op_p2 = positions[self.vsite["out_of_plane"]["index"][:,
                                                                         1], :]
            newpos_op_p3 = positions[self.vsite["out_of_plane"]["index"][:,
                                                                         2], :]
            r12 = newpos_op_p2 - newpos_op_p1
            r13 = newpos_op_p3 - newpos_op_p1
            rcross = jnp.cross(r12, r13, axisa=1, axisb=1, axisc=1)
            newpos = newpos_op_p1 + self.vsite["out_of_plane"]["weight"][:, 0] * r12 + \
                self.vsite["out_of_plane"]["weight"][:, 1] * r13 + \
                self.vsite["out_of_plane"]["weight"][:, 2] * rcross
            positions.at[self.vsite["out_of_plane"]["vsite"], :].set(newpos)

            return positions

        return vsite_update

    def getAtomIndices(self, include_vsite=False):
        if include_vsite:
            return [a.index for a in self.topology.atoms()]
        return [
            a.index for a in self.topology.atoms() if a.element is not None
        ]

    def getNumResidues(self):
        return len(self.residues)

    def getNumAtoms(self, include_vsite=False):
        if include_vsite:
            return len(self.atoms)
        return len(
            [a.index for a in self.topology.atoms() if a.element is not None])

from typing import Dict, Tuple, List
from collections import namedtuple
import networkx as nx
from rdkit import Chem


def top2graph(top: BetterTopology) -> nx.Graph:
    g = nx.Graph()
    print(top)
    for na, a in enumerate(top.atoms()):
        g.add_node(a.index, index=a.index)
    for nb, b in enumerate(top.bonds()):
        g.add_edge(b.atom1.index, b.atom2.index)
    return g

def decompgraph(graph: nx.Graph) -> List[nx.Graph]:
    nsub = [graph.subgraph(indices)
            for indices in nx.connected_components(graph)]
    return nsub

def decomptop(top: BetterTopology) -> List[List[int]]:
    graph = top2graph(top)
    graphs_dec = decompgraph(graph)
    indices = []
    for g in graphs_dec:
        index = []
        for n in g.nodes():
            index.append(g.nodes()[n]["index"])
        indices.append(index)
    return indices

def top2rdmol(top: BetterTopology, indices: List[int]) -> Chem.rdchem.Mol:
    rdmol = Chem.Mol()
    emol = Chem.EditableMol(rdmol)
    idx2ridx = {}
    na = 0
    for atm in top.atoms():
        if atm.element is None:
            continue
        if atm.element in ["none", "EP", "None", "NONE"]:
            continue
        if not atm.index in indices:
            continue
        ratm = Chem.Atom(atm.element)
        ratm.SetProp("_Index", f"{atm.index}")
        emol.AddAtom(ratm)
        idx2ridx[atm.index] = na
        na += 1
    for bnd in top.bonds():
        if bnd.atom1.index not in indices or bnd.atom2.index not in indices:
            continue
        if bnd.order is None:
            order = 1
        else:
            order = bnd.order
        emol.AddBond(idx2ridx[bnd.atom1.index],
                     idx2ridx[bnd.atom2.index], Chem.BondType(order))
    rdmol = emol.GetMol()
    # rdmol.UpdatePropertyCache()
    # AllChem.EmbedMolecule(rdmol, randomSeed=1)
    return rdmol

class BetterTopology:

    def __init__(self):
        self._chains = []
        self._numResidues = 0
        self._numAtoms = 0
        self._bonds = []
        self._molecules = []
        self._vsites = []

    def __repr__(self):
        nchains = len(self._chains)
        nres = self._numResidues
        natom = self._numAtoms
        nbond = len(self._bonds)
        return '<%s; %d chains, %d residues, %d atoms, %d bonds>' % (
                type(self).__name__, nchains, nres, natom, nbond)
    
    def add(self, other):
        offset = self.getNumAtoms()
        newatoms = []
        for chain in other.chains():
            newchain = self.addChain(id=chain.id)
            for res in chain.residues():
                newres = self.addResidue(res.name, newchain, id=res.id, insertionCode=res.insertionCode)
                for atom in res.atoms():
                    newatom = self.addAtom(atom.name, atom.element, newres, id=atom.id)
                    newatoms.append(newatom)
        for bond in other.bonds():
            a1, a2, order = bond.atom1, bond.atom2, bond.order
            self.addBond(newatoms[a1.index], newatoms[a2.index], order)
        # add molecules
        for mol in other.molecules():
            newmol = Chem.Mol()
            emol = Chem.EditableMol(rdmol)
            for atom in mol.GetAtoms():
                newatom = Chem.Atom(atom.GetSymbol())
                idx = atom.GetProp("_Index") + offset
                newatom.SetProp("_Index", f"{idx}")
                emol.AddAtom(newatom)
            for bond in mol.GetBonds():
                i1, i2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                emol.AddBond(i1, i2, bond.GetBondType())
            rdmol = emol.GetMol()
            self._molecules.append(rdmol)

    def updateMolecules(self):
        self._molecules = []
        decomp_indices = decomptop(self)
        for ind in decomp_indices:
            self._molecules.append(top2rdmol(self, ind))
            
    def sanitize(self):
        for mol in self._molecules:
            Chem.sanitize(mol)
            
    def parseSMARTS(self, parser):
        for mol in self._molecules:
            pass
            
    def getNumAtoms(self):
        return self._numAtoms

    def getNumResidues(self):
        return self._numResidues

    def getNumChains(self):
        return len(self._chains)

    def getNumBonds(self):
        return len(self._bonds)
    
    def getNumMolecules(self):
        return len(self._molecules)

    def addChain(self, id=None):
        if id is None:
            id = str(len(self._chains)+1)
        chain = Chain(len(self._chains), self, id)
        self._chains.append(chain)
        return chain

    def addResidue(self, name, chain, id=None, insertionCode=''):
        if len(chain._residues) > 0 and self._numResidues != chain._residues[-1].index+1:
            raise ValueError('All residues within a chain must be contiguous')
        if id is None:
            id = str(self._numResidues+1)
        residue = Residue(name, self._numResidues, chain, id, insertionCode)
        self._numResidues += 1
        chain._residues.append(residue)
        return residue

    def addAtom(self, name, element, residue, id=None, meta=None):
        if len(residue._atoms) > 0 and self._numAtoms != residue._atoms[-1].index+1:
            raise ValueError('All atoms within a residue must be contiguous')
        if id is None:
            id = str(self._numAtoms+1)
        if meta is None:
            meta = {}
            meta["element"] = element
        atom = Atom(name, element, self._numAtoms, residue, id, meta)
        self._numAtoms += 1
        residue._atoms.append(atom)
        return atom

    def addBond(self, atom1, atom2, order=None):
        self._bonds.append(Bond(atom1, atom2, order))
        
    def addVirtualSite(self, vsite):
        self._vsites.append(vsite)

    def chains(self):
        return iter(self._chains)

    def residues(self):
        for chain in self._chains:
            for residue in chain._residues:
                yield residue

    def atoms(self):
        for chain in self._chains:
            for residue in chain._residues:
                for atom in residue._atoms:
                    yield atom

    def bonds(self):
        return iter(self._bonds)
    
    def molecules(self):
        return iter(self._molecules)

class Chain(object):
    def __init__(self, index, topology, id):
        ## The index of the Chain within its Topology
        self.index = index
        ## The Topology this Chain belongs to
        self.topology = topology
        ## A user defined identifier for this Chain
        self.id = id
        self._residues = []

    def residues(self):
        return iter(self._residues)

    def atoms(self):
        for residue in self._residues:
            for atom in residue._atoms:
                yield atom

    def __len__(self):
        return len(self._residues)

    def __repr__(self):
        return "<Chain %d>" % self.index

class Residue(object):
    def __init__(self, name, index, chain, id, insertionCode):
        ## The name of the Residue
        self.name = name
        ## The index of the Residue within its Topology
        self.index = index
        ## The Chain this Residue belongs to
        self.chain = chain
        ## A user defined identifier for this Residue
        self.id = id
        ## A user defined insertion code for this Residue
        self.insertionCode = insertionCode
        self._atoms = []

    def atoms(self):
        return iter(self._atoms)

    def bonds(self):
        return ( bond for bond in self.chain.topology.bonds() if ((bond[0] in self._atoms) or (bond[1] in self._atoms)) )

    def internal_bonds(self):
        return ( bond for bond in self.chain.topology.bonds() if ((bond[0] in self._atoms) and (bond[1] in self._atoms)) )

    def external_bonds(self):
        return ( bond for bond in self.chain.topology.bonds() if ((bond[0] in self._atoms) != (bond[1] in self._atoms)) )

    def __len__(self):
        return len(self._atoms)

    def __repr__(self):
        return "<Residue %d (%s) of chain %d>" % (self.index, self.name, self.chain.index)

class Atom(object):

    def __init__(self, name, element, index, residue, id, meta):
        ## The name of the Atom
        self.name = name
        ## That Atom's element
        self.element = element
        ## The index of the Atom within its Topology
        self.index = index
        ## The Residue this Atom belongs to
        self.residue = residue
        ## A user defined identifier for this Atom
        self.id = id
        
        self.meta = meta

    def __repr__(self):
        return "<Atom %d (%s) of chain %d residue %d (%s)>" % (self.index, self.name, self.residue.chain.index, self.residue.index, self.residue.name)

class Bond(namedtuple('Bond', ['atom1', 'atom2'])):

    def __new__(cls, atom1, atom2, order=None):
        bond = super(Bond, cls).__new__(cls, atom1, atom2)
        bond.order = order
        return bond

    def __getnewargs__(self):
        return self[0], self[1], self.type, self.order

    def __getstate__(self):
        return self.__dict__

    def __deepcopy__(self, memo):
        return Bond(self[0], self[1], self.type, self.order)

    def __repr__(self):
        s = "Bond(%s, %s" % (self[0], self[1])
        if self.order is not None:
            s = "%s, order=%d" % (s, self.order)
        s += ")"
        return s

class VirtualSite:
    
    def __init__(self, type, atoms, weights):
        self.type = type
        self.atoms = atoms
        self.weights = weights