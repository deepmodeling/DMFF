import networkx as nx
from networkx.algorithms import isomorphism
from .base import BaseOperator
from ..api.graph import matchTemplate
from ..utils import DMFFException
from ..api.topology import DMFFTopology, Residue
import openmm.app as app


class TemplateATypeOperator(BaseOperator):

    def __init__(self, ffinfo):
        self.name = "template"
        self.atomtypes = {}
        for node in ffinfo["AtomTypes"]:
            if "name" in node and "class" in node:
                elem = node["element"] if "element" in node else "none"
                self.atomtypes[node["name"]] = (node["class"], elem)

        self.templates = []
        for resinfo in ffinfo["Residues"]:
            self.templates.append(self.generate_template(resinfo))
        self.residues = ffinfo["Residues"]

    def generate_template(self, resinfo):
        graph = nx.Graph()
        name2idx = {}
        names = []
        for na, atom in enumerate(resinfo["particles"]):
            name = atom["name"]
            names.append(name)
            atype = atom["type"]
            elem = self.atomtypes[atype][1]
            if elem is None:
                elem = "none"
            name2idx[name] = na
            external_bond = name in resinfo["externals"]
            graph.add_node(atom["name"], element=elem,
                           external_bond=external_bond, **atom)
        for bond in resinfo["bonds"]:
            try:
                a1, a2 = bond["atomName1"], bond["atomName2"]
            except KeyError:
                i1, i2 = int(bond["from"]), int(bond["to"])
                a1, a2 = names[i1], names[i2]
            graph.add_edge(a1, a2, btype="bond")
        # vsite
        idx2name = {v: k for k, v in name2idx.items()}
        for vsite in resinfo["vsites"]:
            iself = int(vsite["index"])
            for key in vsite.keys():
                if "atom" in key:
                    iatom = int(vsite[key])
                    graph.add_edge(idx2name[iself],
                                   idx2name[iatom], btype="vsite")
        return graph

    def generate_residue_graph(self, topdata: DMFFTopology, residue: Residue):
        residue_indices = [a.index for a in residue.atoms()]
        atoms = [a for a in residue.atoms()]
        graph = nx.Graph()
        # add nodes
        for atom in atoms:
            aidx = atom.index
            external_bond = False
            bonded_atoms = topdata._bondedAtom[aidx]
            for bonded in bonded_atoms:
                if bonded not in residue_indices:
                    external_bond = True
            if isinstance(atom.element, str):
                elem = atom.element
            elif isinstance(atom.element, app.element.Element):
                elem = atom.element.symbol
            elif atom.element is None:
                elem = "none"
            graph.add_node(
                aidx, name=atom.name, element=elem, external_bond=external_bond)
            for bonded in bonded_atoms:
                if bonded < aidx and bonded in residue_indices:
                    graph.add_edge(aidx, bonded, btype="bond")
        # vsite
        for nvsite, vsite in enumerate(topdata.vsites()):
            vidx = vsite.vatom.index
            aidx = [a.index for a in vsite.atoms]
            for a in aidx:
                graph.add_edge(vidx, a, btype="vsite")
        return graph

    def match_all(self, topdata: DMFFTopology, templates):
        residues = [r for r in topdata.residues()]
        atoms = [a for a in topdata.atoms()]
        for res in residues:
            graph = self.generate_residue_graph(topdata, res)
            all_fail = True
            for ntemp, template in enumerate(templates):
                # debug 
                # print(res)
                # print(template)
                # print(dir(template))
                # print('-------')
                is_matched, _, atype_dict = matchTemplate(graph, template)
                if is_matched:
                    all_fail = False
                    # move attribs
                    for key in atype_dict.keys():
                        for key2 in atype_dict[key]:
                            atoms[key].meta[key2] = atype_dict[key][key2]
                        if "type" in atoms[key].meta:
                            itype = atoms[key].meta["type"]
                            atoms[key].meta["class"] = self.atomtypes[itype][0]
                    break
            if all_fail:
                print(f"{res} is not patched.")

    def operate(self, topdata, **kwargs):
        self.match_all(topdata, self.templates)
        return topdata
