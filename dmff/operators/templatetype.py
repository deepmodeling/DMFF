import networkx as nx
from networkx.algorithms import isomorphism
from dmff.operators.base import BaseOperator
from dmff.hamiltonian import dmff_operators
from dmff.utils import DMFFException


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
            graph.add_edge(a1, a2)
        return graph

    def generate_residue_graph(self, topdata, resid):
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

    def matchTemplate(self, graph, template):
        if graph.number_of_nodes() != template.number_of_nodes():
            return False, {}

        def match_func(n1, n2):
            return n1["element"] == n2["element"] and n1["external_bond"] == n2["external_bond"]
        matcher = isomorphism.GraphMatcher(
            graph, template, node_match=match_func)
        is_matched = matcher.is_isomorphic()
        if is_matched:
            match_dict = [i for i in matcher.match()][0]
            atype_dict = {}
            for key in match_dict.keys():
                attrib = {k: v for k, v in template.nodes(
                )[match_dict[key]].items() if k != "name"}
                atype_dict[key] = attrib
        else:
            atype_dict = {}
        return is_matched, atype_dict

    def match_all(self, topdata, templates):
        for ires in range(len(topdata.residues)):
            if topdata.atom_meta[topdata.residue_indices[ires][0]]["operator"] != self.name:
                continue
            graph = self.generate_residue_graph(topdata, ires)
            all_fail = True
            for template in templates:
                is_matched, atype_dict = self.matchTemplate(graph, template)
                if is_matched:
                    all_fail = False
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
