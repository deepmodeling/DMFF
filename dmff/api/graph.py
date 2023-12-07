import networkx as nx
from networkx.algorithms import isomorphism
try:
    import openmm.app as app
    import openmm.unit as unit
except ImportError as e:
    import simtk.openmm.app as app
    import simtk.unit as unit
from typing import Dict, Tuple, List
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError as e:
    import warnings
    warnings.warn("RDKit is not installed. SMIRKS pattern matching cannot be used.")

def is_same_list(l1, l2):
    if len(l1) != len(l2):
        return False
    for nn in range(len(l1)):
        if l1[nn] != l2[nn]:
            return False
    return True

def matchTemplate(graph, template):
    if graph.number_of_nodes() != template.number_of_nodes():
        # print("Node with different number of nodes.")
        return False, {}, {}

    name_graph = sorted([i[1]['name'] for i in graph.nodes.data()])
    name_template = sorted([i[1]['name'] for i in template.nodes.data()])

    if is_same_list(name_graph, name_template):
        def match_func(n1, n2):
            return n1["element"] == n2["element"] and n1["external_bond"] == n2["external_bond"] and n1['name'] == n2['name']
    else:
        def match_func(n1, n2):
            return n1["element"] == n2["element"] and n1["external_bond"] == n2["external_bond"]
    
    def edge_match(e1, e2):
        if len(e1) == 0 and len(e2) == 0:
            return True
        return e1["btype"] == e2["btype"]

    matcher = isomorphism.GraphMatcher(
        graph, template, node_match=match_func, edge_match=edge_match)
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


def top2graph(top: app.Topology) -> nx.Graph:
    g = nx.Graph()
    for na, a in enumerate(top.atoms()):
        elem = a.element.symbol if a.element is not None else "none"
        g.add_node(a.index, index=a.index, name=a.name, element=elem,
                   residx=a.residue.index, resname=a.residue.name)
    for nb, b in enumerate(top.bonds()):
        g.add_edge(b.atom1.index, b.atom2.index)
    return g


def decompgraph(graph: nx.Graph) -> List[nx.Graph]:
    nsub = [graph.subgraph(indices)
            for indices in nx.connected_components(graph)]
    return nsub


def decomptop(top: app.Topology) -> List[List[int]]:
    graph = top2graph(top)
    graphs_dec = decompgraph(graph)
    indices = []
    for g in graphs_dec:
        index = []
        for n in g.nodes():
            index.append(g.nodes()[n]["index"])
        indices.append(index)
    return indices


def graph2top(graph: nx.Graph) -> app.Topology:
    nodes = [graph.nodes[n] for n in graph.nodes]
    nodes = sorted(nodes, key=lambda x: x["index"])
    atoms = []
    node2atom = {}

    top = app.Topology()
    chain = top.addChain("A")
    rid = nodes[0]["residx"]
    rname = nodes[0]["resname"]
    res = top.addResidue(rname, chain)
    for node in nodes:
        if node["residx"] != rid:
            rname = node["resname"]
            res = top.addResidue(rname, chain)
            rid = node["residx"]
        elem = app.element.get_by_symbol(
            node["element"]) if node["element"] != "none" else None
        atom = top.addAtom(
            node["name"], elem, res)
        atoms.append(atom)
        node2atom[node["index"]] = atom

    for b1, b2 in graph.edges.keys():
        n1, n2 = graph.nodes[b1]["index"], graph.nodes[b2]["index"]
        a1, a2 = node2atom[n1], node2atom[n2]
        top.addBond(a1, a2)

    return top


def top2rdmol(top: app.Topology, indices: List[int]):
    rdmol = Chem.Mol()
    emol = Chem.EditableMol(rdmol)
    idx2ridx = {}
    na = 0
    for atm in top.atoms():
        if atm.element is None:
            continue
        if not atm.index in indices:
            continue
        ratm = Chem.Atom(atm.element.atomic_number)
        ratm.SetProp("_Name", atm.name)
        ratm.SetProp("_Index", f"{atm.index}")
        ratm.SetProp("_ResIndex", f"{atm.residue.index}")
        ratm.SetProp("_ResName", atm.residue.name)
        emol.AddAtom(ratm)
        idx2ridx[atm.index] = na
        na += 1
    for bnd in top.bonds():
        if bnd.atom1.index not in indices or bnd.atom2.index not in indices:
            continue
        if bnd.type is None:
            if bnd.order is None:
                order = 1
            else:
                order = bnd.order
        else:
            if isinstance(bnd.type, app.topology.Single):
                order = 1
            elif isinstance(bnd.type, app.topology.Double):
                order = 2
            elif isinstance(bnd.type, app.topology.Triple):
                order = 3
            elif isinstance(bnd.type, app.topology.Aromatic) or isinstance(bnd.type, app.topology.Amide):
                order = 1.5
        emol.AddBond(idx2ridx[bnd.atom1.index],
                     idx2ridx[bnd.atom2.index], Chem.BondType(order))
    rdmol = emol.GetMol()
    # rdmol.UpdatePropertyCache()
    # AllChem.EmbedMolecule(rdmol, randomSeed=1)
    return rdmol
