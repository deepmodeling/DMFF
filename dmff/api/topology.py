from typing import Dict, Tuple, List
from collections import namedtuple
from copy import deepcopy
from .vsite import VirtualSite
import networkx as nx
import openmm.app as app
import openmm.unit as unit
from rdkit import Chem
import numpy as np
import jax.numpy as jnp


class DMFFTopology:
    def __init__(self, from_top=None, from_sdf=None, from_rdmol=None, residue_name="MOL"):
        self._chains = []
        self._numResidues = 0
        self._numAtoms = 0
        self._bonds = []
        self._molecules = []
        self._vsites = []
        self._bondedAtom = {}
        self.cell = None
        self._meta = {}

        if from_top is not None:
            self._load_omm_top(from_top)

        elif from_sdf is not None:
            self._load_sdf(from_sdf, residue_name)

        elif from_rdmol is not None:
            self._load_rdmol(from_rdmol, residue_name)

    def __repr__(self):
        nchains = len(self._chains)
        nres = self._numResidues
        natom = self._numAtoms
        nbond = len(self._bonds)
        return '<%s; %d chains, %d residues, %d atoms, %d bonds>' % (
            type(self).__name__, nchains, nres, natom, nbond)

    def _load_sdf(self, filename, residue_name):
        mol = Chem.MolFromMolFile(filename, removeHs=False)
        atoms = [a for a in mol.GetAtoms()]
        bonds = [b for b in mol.GetBonds()]
        chain = self.addChain()
        res = self.addResidue(residue_name, chain)
        no_symbol = {}
        top_atoms = []
        for atom in atoms:
            symbol = atom.GetSymbol()
            if symbol not in no_symbol:
                no_symbol[symbol] = 0
            no_symbol[symbol] += 1
            top_atom = self.addAtom(
                f"{symbol}{no_symbol[symbol]}", symbol, res)
            top_atoms.append(top_atom)
        for bond in bonds:
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            order = bond.GetBondType()
            self.addBond(top_atoms[idx1], top_atoms[idx2], order)
        self.updateMolecules()

    def _load_rdmol(self, mol, residue_name):
        atoms = [a for a in mol.GetAtoms()]
        bonds = [b for b in mol.GetBonds()]
        chain = self.addChain()
        res = self.addResidue(residue_name, chain)
        no_symbol = {}
        top_atoms = []
        for atom in atoms:
            symbol = atom.GetSymbol()
            if symbol not in no_symbol:
                no_symbol[symbol] = 0
            no_symbol[symbol] += 1
            top_atom = self.addAtom(
                f"{symbol}{no_symbol[symbol]}", symbol, res)
            top_atoms.append(top_atom)
        for bond in bonds:
            idx1 = bond.GetBeginAtomIdx()
            idx2 = bond.GetEndAtomIdx()
            order = bond.GetBondType()
            self.addBond(top_atoms[idx1], top_atoms[idx2], order)
        self._molecules.append(mol)

    def _load_omm_top(self, top: app.Topology):
        # add atom
        for omm_chain in top.chains():
            dmff_chain = self.addChain(id=omm_chain.id)
            for omm_res in omm_chain.residues():
                dmff_res = self.addResidue(
                    omm_res.name, dmff_chain, id=omm_res.id)
                for omm_atom in omm_res.atoms():
                    dmff_atom = self.addAtom(
                        omm_atom.name, omm_atom.element, dmff_res, omm_atom.id)
        atoms = [a for a in self.atoms()]

        # add bonds
        for bond in top.bonds():
            a1, a2, order = bond.atom1, bond.atom2, bond.order
            self.addBond(atoms[a1.index], atoms[a2.index], order)

        self.updateMolecules()

        cell_omm = top.getPeriodicBoxVectors()
        if cell_omm is not None:
            cell = cell_omm.value_in_unit(unit.nanometer)
            cellvec = np.array([
                [cell[0][0], cell[0][1], cell[0][2]],
                [cell[1][0], cell[1][1], cell[1][2]],
                [cell[2][0], cell[2][1], cell[2][2]]
            ])
            self.setPeriodicBoxVectors(cellvec)

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
            Chem.SanitizeMol(mol)

    def parseSMARTS(self, parser, resname=[]):
        atoms = [a for a in self.atoms()]
        if resname:
            aidx = [a.index for a in atoms if a.residue.name in resname]
        else:
            aidx = [a.index for a in atoms]
        parse = Chem.MolFromSmarts(parser)
        ret = []
        self.sanitize()
        for mol in self._molecules:
            matches = mol.GetSubstructMatches(parse)
            for match in matches:
                matched_idx = [
                    int(mol.GetAtomWithIdx(idx).GetProp("_Index"))
                    for idx in match
                ]
                if resname:
                    none_matched = True
                    for idx in matched_idx:
                        if idx in aidx:
                            none_matched = False
                            break
                    if none_matched:
                        continue
                ret.append(matched_idx)
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
            element = "EP"
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
        for vsite in self.vsites():
            self_idx = vsite.vatom.index
            parent = vsite.atoms[0].index
            covalent_map[self_idx,:] = covalent_map[parent,:]
            covalent_map[:,self_idx] = covalent_map[:,parent]
            covalent_map[self_idx,parent] = 1
            covalent_map[parent,self_idx] = 1
        if use_jax:
            return jnp.array(covalent_map)
        return covalent_map

    def buildVSiteUpdateFunction(self):
        # vtype: 2
        vsites_type_2 = [v for v in self.vsites() if v.type == "2"]
        if len(vsites_type_2) > 0:
            use_type_2 = True
            self_idx_type_2 = jnp.array(
                [v.vatom.index for v in vsites_type_2], dtype=int)
            a1_idx_type_2 = jnp.array(
                [v.atoms[0].index for v in vsites_type_2], dtype=int)
            a2_idx_type_2 = jnp.array(
                [v.atoms[1].index for v in vsites_type_2], dtype=int)
            w2_idx_type_2 = jnp.array([v.weights[0] for v in vsites_type_2])
            w1_idx_type_2 = jnp.ones(w2_idx_type_2.shape) - w2_idx_type_2
        else:
            use_type_2 = False

        # vtype: 3
        vsites_type_3 = [v for v in self.vsites() if v.type == "3"]
        if len(vsites_type_3) > 0:
            use_type_3 = True
            self_idx_type_3 = jnp.array(
                [v.vatom.index for v in vsites_type_3], dtype=int)
            a1_idx_type_3 = jnp.array(
                [v.atoms[0].index for v in vsites_type_3], dtype=int)
            a2_idx_type_3 = jnp.array(
                [v.atoms[1].index for v in vsites_type_3], dtype=int)
            a3_idx_type_3 = jnp.array(
                [v.atoms[2].index for v in vsites_type_3], dtype=int)
            w2_idx_type_3 = jnp.array([v.weights[0] for v in vsites_type_3])
            w3_idx_type_3 = jnp.array([v.weights[1] for v in vsites_type_3])
            w1_idx_type_3 = jnp.ones(w2_idx_type_3.shape) - \
                w2_idx_type_3 - w3_idx_type_3
        else:
            use_type_3 = False

        # vtype: 2fd
        vsites_type_2fd = [v for v in self.vsites() if v.type == "2fd"]
        if len(vsites_type_2fd) > 0:
            use_type_2fd = True
            self_idx_type_2fd = jnp.array(
                [v.vatom.index for v in vsites_type_2fd], dtype=int)
            a1_idx_type_2fd = jnp.array(
                [v.atoms[0].index for v in vsites_type_2fd], dtype=int)
            a2_idx_type_2fd = jnp.array(
                [v.atoms[1].index for v in vsites_type_2fd], dtype=int)
            dist_idx_type_2fd = jnp.array([v.weights[0] for v in vsites_type_2fd]).reshape((-1, 1))
        else:
            use_type_2fd = False

        # vtype: 3fd
        vsites_type_3fd = [v for v in self.vsites() if v.type == "3fd"]
        if len(vsites_type_3fd) > 0:
            use_type_3fd = True
            self_idx_type_3fd = jnp.array(
                [v.vatom.index for v in vsites_type_3fd], dtype=int)
            a1_idx_type_3fd = jnp.array(
                [v.atoms[0].index for v in vsites_type_3fd], dtype=int)
            a2_idx_type_3fd = jnp.array(
                [v.atoms[1].index for v in vsites_type_3fd], dtype=int)
            a3_idx_type_3fd = jnp.array(
                [v.atoms[2].index for v in vsites_type_3fd], dtype=int)
            dist_idx_type_3fd = jnp.array([v.weights[0] for v in vsites_type_3fd]).reshape((-1, 1))
        else:
            use_type_3fd = False

        def update_pos(pos):
            # vtype: 2
            if use_type_2:
                new_pos_type_2 = pos[a1_idx_type_2, :] * \
                    w1_idx_type_2 + pos[a2_idx_type_2, :] * w2_idx_type_2
                pos = pos.at[self_idx_type_2, :].set(new_pos_type_2)
            # vtype: 3
            if use_type_3:
                new_pos_type_3 = pos[a1_idx_type_3, :] * w1_idx_type_3 + \
                    pos[a2_idx_type_3, :] * w2_idx_type_3 + \
                    pos[a3_idx_type_3, :] * w3_idx_type_3
                pos = pos.at[self_idx_type_3, :].set(new_pos_type_3)
            # vtype: 2fd
            if use_type_2fd:
                vvec = pos[a1_idx_type_2fd, :] - pos[a2_idx_type_2fd]
                rvec = vvec / jnp.linalg.norm(vvec, axis=1).reshape((-1, 1))
                new_pos_type_2fd = pos[a1_idx_type_2fd,
                                    :] + rvec * dist_idx_type_2fd
                pos = pos.at[self_idx_type_2fd, :].set(new_pos_type_2fd)
            # vtype: 3fd
            if use_type_3fd:
                vji = pos[a1_idx_type_3fd, :] - pos[a2_idx_type_3fd, :]
                vki = pos[a1_idx_type_3fd, :] - pos[a3_idx_type_3fd, :]
                rji = vji / jnp.linalg.norm(vji, axis=1).reshape((-1, 1))
                rki = vki / jnp.linalg.norm(vki, axis=1).reshape((-1, 1))
                vmid = rji + rki
                rmid = vmid / jnp.linalg.norm(vmid, axis=1).reshape((-1, 1))

                new_pos_type_3fd = pos[self_idx_type_3fd,
                                    :] + rmid * dist_idx_type_3fd
                pos = pos.at[self_idx_type_3fd, :].set(new_pos_type_3fd)

            return pos

        return update_pos
    
    def addVSiteToPos(self, positions):
        new_pos = jnp.zeros((self.getNumAtoms(), 3))
        idx = [a.index for a in self.atoms() if a.meta["element"] != "EP"]
        new_pos = new_pos.at[idx,:].set(positions[:,:])
        update_func = self.buildVSiteUpdateFunction()
        return update_func(new_pos)

    def getEquivalentAtoms(self):
        graph = nx.Graph()
        for atom in self.atoms():
            elem = atom.meta["element"]
            if elem == "EP":
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
            if elem == "EP":
                eq_atoms[atom.index] = [atom.index]
            eq_atoms[atom.index] = list(
                set([i[atom.index] for i in isomorphisms]))
        return eq_atoms

    def getPeriodicBoxVectors(self, use_jax=True):
        if use_jax:
            return jnp.array(self.cell)
        return self.cell

    def setPeriodicBoxVectors(self, box):
        self.cell = np.zeros((3, 3))
        self.cell[:, :] = box[:, :]


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
        return self[0], self[1], self.order

    def __getstate__(self):
        return self.__dict__

    def __deepcopy__(self, memo):
        return Bond(self[0], self[1], self.order)

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
    for nvs, vs in enumerate(top.vsites()):
        g.add_edge(vs.vatom.index, vs.atoms[0].index)
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
        elif atm.element in ["none", "EP", "None", "NONE"]:
            ratm = Chem.Atom(0)
        elif not atm.index in indices:
            continue
        else:
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
    for vsite in top.vsites():
        vidx = vsite.vatom.index
        parent = vsite.atoms[0].index
        if vidx not in indices or parent not in indices:
            continue
        emol.AddBond(idx2ridx[vidx], idx2ridx[parent])
    rdmol = emol.GetMol()
    # rdmol.UpdatePropertyCache()
    # AllChem.EmbedMolecule(rdmol, randomSeed=1)
    return rdmol
