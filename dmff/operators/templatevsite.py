from ..utils import DMFFException
from .base import BaseOperator
from ..api.xmlio import XMLIO
from ..api.vsite import VirtualSite
from ..api.vstools import insertVirtualSites
from ..api.topology import DMFFTopology
from ..api.graph import matchTemplate
from typing import List, Union
import openmm.app as app
import networkx as nx


class TemplateVSiteOperator(BaseOperator):

    def __init__(self, filename: Union[str, List[str]]):
        xmlio = XMLIO()
        if isinstance(filename, str):
            xmlio.loadXML(filename)
        else:
            for fn in filename:
                xmlio.loadXML(fn)
        ffinfo = xmlio.parseXML()

        self.atype_to_elem = {}
        for atype in ffinfo["AtomTypes"]:
            atype, elem = atype["name"], atype["element"]
            self.atype_to_elem[atype] = elem

        self.residue_templates = []
        self.residue_infos = []
        for residue in ffinfo["Residues"]:
            if len(residue["vsites"]) > 0:
                self.residue_infos.append(residue)
                self.residue_templates.append(self.template2graph(residue))

    def template2graph(self, template):
        graph = nx.Graph()
        name2idx = {}
        for na, atom in enumerate(template["particles"]):
            name = atom["name"]
            atype = atom["type"]
            elem = self.atype_to_elem[atype]
            if elem is None:
                continue
            name2idx[name] = na
            external_bond = name in template["externals"]
            graph.add_node(atom["name"], element=elem,
                           external_bond=external_bond, **atom)
        for bond in template["bonds"]:
            a1, a2 = bond["atomName1"], bond["atomName2"]
            graph.add_edge(a1, a2)
        return graph

    def res2graph(self, topdata, residx):
        atoms = [a for a in topdata.atoms()]
        residue = [r for r in topdata.residues()][residx]
        residue_indices = [a.index for a in residue.atoms()]
        graph = nx.Graph()
        # add nodes
        for atom in residue_indices:
            external_bond = False
            bonded_atoms = topdata._bondedAtom[atom]
            for bonded in bonded_atoms:
                if bonded not in residue_indices:
                    external_bond = True
            if isinstance(atoms[atom].element, str):
                elem = atoms[atom].element
            elif isinstance(atoms[atom].element, app.element.Element):
                elem = atoms[atom].element.symbol
            elif atoms[atom].element is None:
                continue
            graph.add_node(
                atom, name=atoms[atom].name, element=elem, external_bond=external_bond)
            for bonded in bonded_atoms:
                if bonded < atom and bonded in residue_indices:
                    graph.add_edge(atom, bonded)
        return graph

    def operate(self, topdata: DMFFTopology, **kwargs) -> DMFFTopology:
        # get vslist
        vslist = []
        atoms = [a for a in topdata.atoms()]
        for residue in topdata.residues():
            rgraph = self.res2graph(topdata, residue.index)
            for nt, tgraph in enumerate(self.residue_templates):
                is_matched, matched_dict, atype_dict = matchTemplate(
                    rgraph, tgraph)
                if is_matched:
                    name2idx_template = {}
                    for k, v in matched_dict.items():
                        name2idx_template[v] = k
                    for vs in self.residue_infos[nt]["vsites"]:
                        vtype = vs["type"]
                        idx = int(vs["index"])
                        if vtype == "average2":
                            a1i, a2i = int(vs["atom1"]), int(vs["atom2"])
                            a1_name = self.residue_infos[nt]["particles"][a1i]["name"]
                            a2_name = self.residue_infos[nt]["particles"][a2i]["name"]
                            vs_name = self.residue_infos[nt]["particles"][idx]["name"]
                            a1_idx = name2idx_template[a1_name]
                            a2_idx = name2idx_template[a2_name]
                            a1, a2 = atoms[a1_idx], atoms[a2_idx]
                            w1, w2 = float(vs["weight1"]), float(vs["weight2"])
                            vsite = VirtualSite(vtype, [a1, a2], [w1, w2])
                            vslist.append(vsite)
                    break
        return insertVirtualSites(topdata, vslist)
