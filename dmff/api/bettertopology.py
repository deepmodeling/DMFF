from typing import Dict, Tuple, List
from collections import namedtuple
import networkx as nx
from rdkit import Chem


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
            emol = Chem.EditableMol(newmol)
            for atom in mol.GetAtoms():
                newatom = Chem.Atom(atom.GetSymbol())
                idx = int(atom.GetProp("_Index")) + offset
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
        parse = Chem.MolFromSmarts(parser)
        ret = []
        for mol in self._molecules:
            matches = mol.GetSubstructMatches(parse)
            for match in matches:
                ret.append([mol.GetAtomWithIdx(idx).GetProp("_Index") for idx in match])
        return ret
            
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
        
    def addVirtualSites(self, vsite_list):
        pass

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

class TwoPointAverageSite:
    
    def __init__(self, atom1, atom2, weight1, weight2, vatom=None):
        self.atom1 = atom1
        self.atom2 = atom2
        self.weight1 = weight1
        self.weight2 = weight2
        self.vatom = vatom
        
    def __deepcopy__(self, memo):
        return TwoPointAverageSite(self.atom1, self.atom2, self.weight1, self.weight2)

    def __repr__(self):
        s = f"Two Point Average VSite: {self.weight1} x {self.atom1} + {self.weight2} x {self.atom2}"
        return s
        

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