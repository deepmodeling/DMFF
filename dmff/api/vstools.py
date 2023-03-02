import openmm.app as app
from typing import List, Union, Tuple
import xml.etree.ElementTree as ET
import networkx as nx
from networkx.algorithms import isomorphism
from .vsite import VSite, TwoParticleAverageSite, ThreeParticleAverageSite, OutOfPlaneSite
from .topology import TopologyData
from .graph import matchTemplate
from rdkit import Chem


class TemplateVSitePatcher:

    def __init__(self):
        self.residue_templates = []
        self.residue_infos = []
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

        for residue in self._residue:
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
                self.residue_templates.append(self.template2graph(res))
                self.residue_infos.append(res)

    def patch(self, topdata: TopologyData, resid: Union[List[int], None], vslist: List[VSite]):
        if resid is None:
            resid = [_ for _ in range(topdata.getNumResidues())]
        for rid in resid:
            rgraph = self.res2graph(topdata, rid)
            for nt, tgraph in enumerate(self.residue_templates):
                is_matched, matched_dict, atype_dict = matchTemplate(
                    rgraph, tgraph)
                if is_matched:
                    name2idx_template = {}
                    for k,v in matched_dict.items():
                        name2idx_template[v] = k
                    for vs in self.residue_infos[nt]["vsite"]:
                        vtype = vs["type"]
                        idx = int(vs["index"])
                        if vtype == "average2":
                            a1i, a2i = int(vs["atom1"]), int(vs["atom2"])
                            a1_name = self.residue_infos[nt]["particles"][a1i]["name"]
                            a2_name = self.residue_infos[nt]["particles"][a2i]["name"]
                            vs_name = self.residue_infos[nt]["particles"][idx]["name"]
                            a1_idx = name2idx_template[a1_name]
                            a2_idx = name2idx_template[a2_name]
                            a1, a2 = topdata.atoms[a1_idx], topdata.atoms[a2_idx]
                            w1, w2 = float(vs["weight1"]), float(vs["weight2"])
                            vsite = TwoParticleAverageSite([a1, a2], [w1, w2], name=vs_name)
                            vslist.append(vsite)
                    break


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
            if elem == "none":
                continue
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
        self.infos = []

    def loadFile(self, filename):
        root = ET.parse(filename).getroot()
        for child in root:
            if child.tag == "VirtualSite":
                self.infos.append(child.attrib)

    def parse(self):
        pass

    def patch(self, topdata: TopologyData, resid: Union[List[int], None], vslist: List[VSite]):
        if resid is None:
            resid = [_ for _ in range(topdata.getNumResidues())]
        mols = list(set([topdata.res2mol[r] for r in resid]))
        for nmol in mols:
            rdmol = topdata.rdmols[nmol]
            Chem.SanitizeMol(rdmol)
            atoms = rdmol.GetAtoms()
            for info in self.infos:
                parser = info["smarts"]
                par = Chem.MolFromSmarts(parser)
                matches = rdmol.GetSubstructMatches(par)
                for match in matches:
                    if info["type"] == "average2":
                        a1idx, a2idx = match[0], match[1]
                        idx1 = int(atoms[a1idx].GetProp("_Index"))
                        idx2 = int(atoms[a2idx].GetProp("_Index"))
                        a1 = topdata.atoms[idx1]
                        a2 = topdata.atoms[idx2]
                        w1, w2 = info["weight1"], info["weight2"]
                        vsite = TwoParticleAverageSite([a1, a2], [w1, w2])
                        vslist.append(vsite)




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
                if vsite.vsname is None:
                    name = f"V{nv+1}"
                else:
                    name = vsite.vsname
                vatom = newtop.addAtom(name, None, newres)
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
