import openmm.app as app
from typing import List, Union, Tuple
import xml.etree.ElementTree as ET
import networkx as nx
from networkx.algorithms import isomorphism
from .vsite import VirtualSite
from .topology import DMFFTopology
from .graph import matchTemplate
from rdkit import Chem


def pickTheSame(obj, li) -> int:
    for no, o in enumerate(li):
        if o == obj:
            return no
    raise BaseException(f"No object found in list.")


def insertVirtualSites(topdata, vsite_list):
    parent2vsite = {}
    for vsite in vsite_list:
        parent = vsite.atoms[0].index
        if parent not in parent2vsite:
            parent2vsite[parent] = []
        parent2vsite[parent].append(vsite)
    tot_vsites = []
    vsite_and_vatom = []

    newatoms = []
    newtop = DMFFTopology()
    for chain in topdata.chains():
        newchain = newtop.addChain(id=chain.id)
        for residue in chain.residues():
            newres = newtop.addResidue(
                name=residue.name, chain=newchain, id=residue.id)
            nep = 1
            for atom in residue.atoms():
                newatom = newtop.addAtom(
                    atom.name, atom.element, newres, id=atom.id, meta=atom.meta)
                newatoms.append(newatom)
                if atom.element is None:
                    nep += 1

                # add new vsite
                if atom.index in parent2vsite:
                    for vsite in parent2vsite[atom.index]:
                        newvatom = newtop.addAtom(f"V{nep}", None, newres)
                        nep += 1
                        vsite_and_vatom.append((vsite, newvatom))

    for vs, va in vsite_and_vatom:
        aidx = [a.index for a in vs.atoms]
        weights = vs.weights
        vtype = vs.type
        vmeta = vs.meta
        newvsite = VirtualSite(vtype, [newatoms[i]
                                for i in aidx], weights, vatom=va, meta=vmeta)
        tot_vsites.append(newvsite)

    for bond in topdata.bonds():
        idx1 = bond.atom1.index
        idx2 = bond.atom2.index
        order = bond.order
        newtop.addBond(newatoms[idx1], newatoms[idx2], order=order)

    for vsite in topdata.vsites():
        vtype = vsite.type
        aidx = [a.index for a in vsite.atoms]
        weights = vsite.weights
        vidx = vsite.vatom.index
        vmeta = vsite.meta
        new_vsite = VirtualSite(
            vtype, [newatoms[i] for i in aidx], weights, vatom=newatoms[vidx], meta=vmeta)
        tot_vsites.append(new_vsite)

    newtop._vsites = sorted(tot_vsites, key=lambda x: x.atoms[0].index)
    for vsite in newtop.vsites():
        for k, v in vsite.meta.items():
            vsite.vatom.meta[k] = v

    for mol in topdata.molecules():
        # copy a molecule with new index
        newmol = Chem.Mol()
        emol = Chem.EditableMol(newmol)
        for atom in mol.GetAtoms():
            newatom = Chem.Atom(atom.GetSymbol())
            idx = int(atom.GetProp("_Index"))
            newatom.SetProp("_Index", f"{newatoms[idx].index}")
            emol.AddAtom(newatom)
        for bond in mol.GetBonds():
            i1, i2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            emol.AddBond(i1, i2, bond.GetBondType())
        rdmol = emol.GetMol()
        newtop._molecules.append(rdmol)
    return newtop

