import openmm.app as app
from typing import List, Union, Tuple
import xml.etree.ElementTree as ET
import networkx as nx
from networkx.algorithms import isomorphism


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


class TemplateVSitePatcher:

    def __init__(self):
        self.residue_templates = []
        self.atype_to_elem = {}
        self._residue = []
        self._atomtypes = []

    def loadFile(self, filename):
        root = ET.parse(filename).getroot()
        for child in root:
            if child.tag == "AtomTypes":
                for atom in child:
                    if atom.tag == "Type":
                        self._atomtypes.append(atom)
            if child.tag == "Residues":
                for residue in child:
                    if residue.tag == "Residue":
                        self._residue.append(residue)

    def parse(self):
        for atype in self._atomtypes:
            typename = atype.attrib["name"]
            if "element" not in atype.attrib:
                elem = "none"
            else:
                elem = atype.attrib["element"]
            self.atype_to_elem[typename] = elem

        for residue in self._residue["Residues"]:
            res = {
                "name": None, "particles": [], "bonds": [], "externals": [], "vsite": []
            }
            res["name"] = residue.attrib["name"]
            for item in residue:
                if item.tag == "Atom":
                    ainner = {}
                    for key in item.attrib:
                        if key in ["name", "type"]:
                            ainner[key] = item.attrib[key]
                        else:
                            ainner[key] = float(item.attrib[key])
                    res["particles"].append(ainner)
                if item.tag == "Bond":
                    res["bonds"].append(item.attrib)
                if item.tag == "ExternalBond":
                    res["externals"].append(item.attrib["atomName"])
                if item.tag == "VirtualSite":
                    res["vsite"].append(item.attrib)
            if len(res["vsite"]) > 0:
                self.residue_templates.append(res)

    def patch(self, topdata, resid=None):
        pass

    def res2graph(self, topdata, resid):
        residue_indices = topdata.residue_indices[resid]
        graph = nx.Graph()
        # add nodes
        for atom in residue_indices:
            external_bond = False
            bonded_atoms = topdata._bondedAtom[atom]
            for bonded in bonded_atoms:
                if bonded not in residue_indices:
                    external_bond = True
            elem = topdata.atoms[atom].element.symbol if topdata.atoms[atom].element is not None else "none"
            graph.add_node(
                atom, name=topdata.atoms[atom].name, element=elem, external_bond=external_bond)
            for bonded in bonded_atoms:
                if bonded < atom and bonded in residue_indices:
                    graph.add_edge(atom, bonded)
        return graph

    def template2graph(self, template):
        graph = nx.Graph()
        name2idx = {}
        for na, atom in enumerate(template["particles"]):
            name = atom["name"]
            atype = atom["type"]
            elem = self.atype_to_elem[atype]
            name2idx[name] = na
            external_bond = name in template["externals"]
            graph.add_node(atom["name"], element=elem,
                           external_bond=external_bond, **atom)
        for bond in template["bonds"]:
            a1, a2 = bond["atomName1"], bond["atomName2"]
            graph.add_edge(a1, a2)
        return graph


class SMARTSVSitePatcher:

    def __init__(self):
        pass

    def loadFile(self, filename):
        pass

    def parse(self):
        pass

    def patch(self, top, resid=None):
        pass


def pickTheSame(obj, li) -> int:
    for no, o in enumerate(li):
        if o == obj:
            return no
    raise BaseException(f"No object found in list.")


def loadVSiteTemplate(filename: Union[str, List[str]]) -> TemplateVSitePatcher:
    patcher = TemplateVSitePatcher()
    return _loadVSitePatcher(filename, patcher)


def loadVSiteSMARTS(filename: Union[str, List[str]]) -> SMARTSVSitePatcher:
    patcher = SMARTSVSitePatcher()
    return _loadVSitePatcher(filename, patcher)


def _loadVSitePatcher(filename: Union[str, List[str]], patcher: Union[TemplateVSitePatcher, SMARTSVSitePatcher]):
    if isinstance(filename, str):
        patcher.loadFile(filename)
    else:
        for fname in filename:
            patcher.loadFile(fname)
    patcher.parse()
    return patcher


def addVSiteToTopology(top: app.Topology, vslist: List[VSite]) -> Tuple[app.Topology, List[VSite]]:
    atom_per_residue = []
    vsite_per_residue = []
    atom_all = []
    for nres, res in enumerate(top.residues()):
        atom_per_residue.append([])
        vsite_per_residue.append([])
        for atom in res.atoms():
            atom_per_residue[nres].append(atom)
            atom_all.append(atom)

    for vsite in vslist:
        a1 = vsite.atoms[0]
        ridx = a1.residue.index
        vsite_per_residue[ridx].append(vsite)

    newtop = app.Topology()
    new_vslist = []
    new_atom_all = []
    for chain in top.chains():
        newchain = newtop.addChain()
        for residue in chain.residues():
            newatom_list = []
            newres = newtop.addResidue(residue.name, newchain)
            # add exist atoms
            for atom in atom_per_residue[residue.index]:
                newatom = newtop.addAtom(atom.name, atom.element, newres)
                newatom_list.append(newatom)
                new_atom_all.append(newatom)
            # add vsite
            for nv, vsite in enumerate(vsite_per_residue[residue.index]):
                vatom = newtop.addAtom(f"V{nv+1}", None, newres)
                atoms = []
                weights = vsite.weights
                for natom, atom in enumerate(vsite.atoms):
                    newa = newatom_list[pickTheSame(
                        atom, atom_per_residue[residue.index])]
                    atoms.append(newa)
                if isinstance(vsite, TwoParticleAverageSite):
                    new_vslist.append(
                        TwoParticleAverageSite(atoms, weights, vatom))
                elif isinstance(vsite, ThreeParticleAverageSite):
                    new_vslist.append(
                        ThreeParticleAverageSite(atoms, weights, vatom))
                elif isinstance(vsite, OutOfPlaneSite):
                    new_vslist.append(OutOfPlaneSite(atoms, weights, vatom))
                else:
                    raise BaseException(f"Virtual site type not supported.")
    for bond in top.bonds():
        a1, a2 = bond.atom1, bond.atom2
        na1 = new_atom_all[a1.index]
        na2 = new_atom_all[a2.index]
        newtop.addBond(na1, na2)
    return newtop, new_vslist


def insertVirtualSites(top: app.Topology, templatePatcher=Union[TemplateVSitePatcher, None], smartsPatcher=Union[SMARTSVSitePatcher, None]) -> Tuple[app.Topology, List[VSite]]:
    vslist = []
    from .topology import TopologyData
    topdata = TopologyData(top)
    # map virtual site from template
    if templatePatcher is not None:
        templatePatcher.patch(topdata, vslist)
    # map virtual site from smarts
    if smartsPatcher is not None:
        smartsPatcher.patch(topdata, vslist)
    # update topology
    if templatePatcher is None and smartsPatcher is None:
        return top, vslist
    return addVSiteToTopology(top, vslist)
