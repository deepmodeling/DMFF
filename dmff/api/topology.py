from typing import Dict, Tuple, List
from collections import namedtuple
from copy import deepcopy
from .vsite import VirtualSite
import networkx as nx
import openmm.app as app
from rdkit import Chem
import numpy as np
import jax.numpy as jnp


class DMFFTopology:
    def __init__(self, from_top=None):
        self._chains = []
        self._numResidues = 0
        self._numAtoms = 0
        self._bonds = []
        self._molecules = []
        self._vsites = []
        self._bondedAtom = {}

        if from_top is not None:
            self._load_omm_top(from_top)

    def __repr__(self):
        nchains = len(self._chains)
        nres = self._numResidues
        natom = self._numAtoms
        nbond = len(self._bonds)
        return '<%s; %d chains, %d residues, %d atoms, %d bonds>' % (
            type(self).__name__, nchains, nres, natom, nbond)

    def _load_from_top(self, top):
        pass

    def add(self, other, newchain=False):
        offset = self.getNumAtoms()
        newatoms = []
        for nchain, chain in enumerate(other.chains()):
            if nchain == 0 and not newchain and self.getNumChains() > 0:
                newchain = [c for c in self.chains()][-1]
            else:
                newchain = self.addChain(id=chain.id)
            for res in chain.residues():
                newres = self.addResidue(res.name,
                                         newchain,
                                         id=res.id,
                                         insertionCode=res.insertionCode)
                for atom in res.atoms():
                    newatom = self.addAtom(atom.name,
                                           atom.element,
                                           newres,
                                           id=atom.id,
                                           meta=deepcopy(atom.meta))
                    newatoms.append(newatom)
        for bond in other.bonds():
            a1, a2, order = bond.atom1, bond.atom2, bond.order
            self.addBond(newatoms[a1.index], newatoms[a2.index], order)

        for vsite in other.vsites():
            vtype = vsite.type
            aidx = [a.index for a in vsite.atoms]
            weights = vsite.weights
            vatom = newatoms[vsite.vatom.index]
            self._vsites.append(
                VirtualSite(vtype, [newatoms[i] for i in aidx],
                            weights,
                            vatom=vatom))

        # add molecules
        for mol in other.molecules():
            newmol = Chem.Mol()
            emol = Chem.EditableMol(newmol)
            for atom in mol.GetAtoms():
                newatom = Chem.Atom(atom.GetSymbol())
                idx = int(atom.GetProp("_Index")) + offset
                name = atom.GetProp("_Name")
                newatom.SetProp("_Index", f"{idx}")
                newatom.SetProp("_Name", name)
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
        parse = Chem.MolFromSmarts(parser)
        ret = []
        for mol in self._molecules:
            matches = mol.GetSubstructMatches(parse)
            for match in matches:
                ret.append([
                    int(mol.GetAtomWithIdx(idx).GetProp("_Index"))
                    for idx in match
                ])
        return ret

    def getProperty(self, property_name):
        data = []
        for atom in self.atoms():
            data.append(atom.meta[property_name])
        return np.array(data)

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
            id = str(len(self._chains) + 1)
        chain = Chain(len(self._chains), self, id)
        self._chains.append(chain)
        return chain

    def addResidue(self, name, chain, id=None, insertionCode=''):
        if len(chain._residues
               ) > 0 and self._numResidues != chain._residues[-1].index + 1:
            raise ValueError('All residues within a chain must be contiguous')
        if id is None:
            id = str(self._numResidues + 1)
        residue = Residue(name, self._numResidues, chain, id, insertionCode)
        self._numResidues += 1
        chain._residues.append(residue)
        return residue

    def addAtom(self, name, element, residue, id=None, meta=None):
        if isinstance(element, app.element.Element):
            element = element.symbol
        elif isinstance(element, str):
            element = element
        elif element is None:
            element = "none"
        if len(residue._atoms
               ) > 0 and self._numAtoms != residue._atoms[-1].index + 1:
            raise ValueError('All atoms within a residue must be contiguous')
        if id is None:
            id = str(self._numAtoms + 1)
        if meta is None:
            meta = {}
            meta["element"] = element
        atom = Atom(name, element, self._numAtoms, residue, id, meta)
        self._numAtoms += 1
        residue._atoms.append(atom)
        self._bondedAtom[atom.index] = []
        return atom

    def addBond(self, atom1, atom2, order=None):
        self._bonds.append(Bond(atom1, atom2, order))
        self._bondedAtom[atom1.index].append(atom2.index)
        self._bondedAtom[atom2.index].append(atom1.index)

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

    def vsites(self):
        return iter(self._vsites)

    def buildCovMat(self, nmax=6, use_jax=True):
        n_atoms = self.getNumAtoms()
        covalent_map = np.zeros((n_atoms, n_atoms), dtype=int)
        for bond in self.bonds():
            covalent_map[bond.atom1.index, bond.atom2.index] = 1
            covalent_map[bond.atom2.index, bond.atom1.index] = 1
        for n_curr in range(1, nmax):
            for i in range(n_atoms):
                # current neighbors
                j_list = np.where(
                    np.logical_and(covalent_map[i] <= n_curr,
                                   covalent_map[i] > 0))[0]
                for j in j_list:
                    k_list = np.where(covalent_map[j] == 1)[0]
                    for k in k_list:
                        if k != i and k not in j_list:
                            covalent_map[i, k] = n_curr + 1
                            covalent_map[k, i] = n_curr + 1
        if use_jax:
            return jnp.array(covalent_map)
        return covalent_map

    def buildVSiteUpdateFunction(self):
        pass

    def getEquivalentAtoms(self):
        graph = nx.Graph()
        for atom in self.atoms():
            elem = atom.meta["element"]
            if elem == "none":
                continue
            graph.add_node(atom.index, elem=elem)
        for bond in self.bonds():
            a1, a2 = bond.atom1, bond.atom2
            graph.add_edge(a1.index, a2.index, btype="bond")

        def match_node(n1, n2):
            return n1["elem"] == n2["elem"]

        ismags = nx.isomorphism.ISMAGS(graph, graph, node_match=match_node)
        isomorphisms = list(ismags.isomorphisms_iter(symmetry=False))
        eq_atoms = {}
        for atom in self.atoms():
            elem = atom.meta["element"]
            if elem == "none":
                eq_atoms[atom.index] = []
            eq_atoms[atom.index] = list(set([i[atom.index] for i in isomorphisms]))
        return eq_atoms




class Chain(object):
    def __init__(self, index, topology, id):
        # The index of the Chain within its Topology
        self.index = index
        # The Topology this Chain belongs to
        self.topology = topology
        # A user defined identifier for this Chain
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
        # The name of the Residue
        self.name = name
        # The index of the Residue within its Topology
        self.index = index
        # The Chain this Residue belongs to
        self.chain = chain
        # A user defined identifier for this Residue
        self.id = id
        # A user defined insertion code for this Residue
        self.insertionCode = insertionCode
        self._atoms = []

    def atoms(self):
        return iter(self._atoms)

    def bonds(self):
        return (bond for bond in self.chain.topology.bonds()
                if ((bond[0] in self._atoms) or (bond[1] in self._atoms)))

    def internal_bonds(self):
        return (bond for bond in self.chain.topology.bonds()
                if ((bond[0] in self._atoms) and (bond[1] in self._atoms)))

    def external_bonds(self):
        return (bond for bond in self.chain.topology.bonds()
                if ((bond[0] in self._atoms) != (bond[1] in self._atoms)))

    def __len__(self):
        return len(self._atoms)

    def __repr__(self):
        return "<Residue %d (%s) of chain %d>" % (self.index, self.name,
                                                  self.chain.index)


class Atom(object):
    def __init__(self, name, element, index, residue, id, meta):
        # The name of the Atom
        self.name = name
        # That Atom's element
        if isinstance(element, app.element.Element):
            self.element = element.symbol
        else:
            self.element = element
        # The index of the Atom within its Topology
        self.index = index
        # The Residue this Atom belongs to
        self.residue = residue
        # A user defined identifier for this Atom
        self.id = id

        self.meta = meta

    def __repr__(self):
        return "<Atom %d (%s) of chain %d residue %d (%s)>" % (
            self.index, self.name, self.residue.chain.index,
            self.residue.index, self.residue.name)


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


def top2graph(top) -> nx.Graph:
    g = nx.Graph()
    for na, a in enumerate(top.atoms()):
        g.add_node(a.index, index=a.index)
    for nb, b in enumerate(top.bonds()):
        g.add_edge(b.atom1.index, b.atom2.index)
    return g


def decompgraph(graph) -> List[nx.Graph]:
    nsub = [
        graph.subgraph(indices) for indices in nx.connected_components(graph)
    ]
    return nsub


def decomptop(top) -> List[List[int]]:
    graph = top2graph(top)
    graphs_dec = decompgraph(graph)
    indices = []
    for g in graphs_dec:
        index = []
        for n in g.nodes():
            index.append(g.nodes()[n]["index"])
        indices.append(index)
    return indices


def top2rdmol(top, indices) -> Chem.rdchem.Mol:
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
        ratm.SetProp("_Name", atm.name)
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
        emol.AddBond(idx2ridx[bnd.atom1.index], idx2ridx[bnd.atom2.index],
                     Chem.BondType(order))
    rdmol = emol.GetMol()
    # rdmol.UpdatePropertyCache()
    # AllChem.EmbedMolecule(rdmol, randomSeed=1)
    return rdmol
