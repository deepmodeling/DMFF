import networkx as nx
from networkx.algorithms import isomorphism
from .base import BaseOperator
from ..api.hamiltonian import dmff_operators
from ..api.graph import matchTemplate
from ..utils import DMFFException
from ..api.topology import TopologyData


class TemplateOperator(BaseOperator):

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
        for na, atom in enumerate(resinfo["particles"]):
            name = atom["name"]
            atype = atom["type"]
            elem = self.atomtypes[atype][1]
            name2idx[name] = na
            external_bond = name in resinfo["externals"]
            graph.add_node(atom["name"], element=elem,
                           external_bond=external_bond, **atom)
        for bond in resinfo["bonds"]:
            a1, a2 = bond["atomName1"], bond["atomName2"]
            graph.add_edge(a1, a2, btype="bond")
        # vsite
        idx2name = {v:k for k,v in name2idx.items()}
        for vsite in resinfo["vsites"]:
            iself = int(vsite["index"])
            if vsite["type"] == "average2":
                iatom1 = int(vsite["atom1"])
                iatom2 = int(vsite["atom2"])
                graph.add_edge(idx2name[iself], idx2name[iatom1], btype="vsite")
                graph.add_edge(idx2name[iself], idx2name[iatom2], btype="vsite")
        return graph

    def generate_residue_graph(self, topdata: TopologyData, resid: int):
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
                    graph.add_edge(atom, bonded, btype="bond")
        # vsite
        for nvsite, vsite in enumerate(topdata.vsite["two_point_average"]["vsite"]):
            a1, a2 = topdata.vsite["two_point_average"]["index"][nvsite]
            vsite, a1, a2 = int(vsite), int(a1), int(a2)
            graph.add_edge(vsite, a1, btype="vsite")
            graph.add_edge(vsite, a2, btype="vsite")
        return graph

    def match_all(self, topdata: TopologyData, templates):
        for ires in range(len(topdata.residues)):
            if self.name not in topdata.atom_meta[topdata.residue_indices[ires][0]]["operator"]:
                continue
            graph = self.generate_residue_graph(topdata, ires)
            all_fail = True
            for ntemp, template in enumerate(templates):
                is_matched, _, atype_dict = matchTemplate(graph, template)
                if is_matched:
                    all_fail = False
                    # move attribs
                    for key in atype_dict.keys():
                        for key2 in atype_dict[key]:
                            topdata.atom_meta[key][key2] = atype_dict[key][key2]
                        if "type" in topdata.atom_meta[key]:
                            itype = topdata.atom_meta[key]["type"]
                            topdata.atom_meta[key]["class"] = self.atomtypes[itype][0]
                    break
            if all_fail:
                print(f"{topdata.residues[ires]} is not patched.")

    def operate(self, topdata):
        self.match_all(topdata, self.templates)


dmff_operators["template"] = TemplateOperator
