import networkx as nx
from networkx.algorithms import isomorphism

def matchTemplate(graph, template):
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
        match_dict = {}
        atype_dict = {}
    return is_matched, match_dict, atype_dict