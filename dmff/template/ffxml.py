import xml.etree.ElementTree as ET
import networkx as nx


def generate_templates_from_xml(*files):
    type2element = {}
    for fname in files:
        tree = ET.parse(fname)
        root = tree.getroot()
        atypes = root.find("AtomTypes")
        if atypes is None:
            continue
        for atype in atypes:
            tname = atype.get("name")
            elem = atype.get("element")
            type2element[tname] = elem

    templates = []
    for fname in files:
        tree = ET.parse(fname)
        root = tree.getroot()

        residues = root.find("Residues")
        if residues is None:
            continue
        for residue in residues:
            graph = nx.Graph()
            name2idx = {}
            for na, atom in enumerate(residue.findall("Atom")):
                graph.add_node(
                    na, element=type2element[atom.attrib["type"]], external_bond=False, **atom.attrib)
                name2idx[atom.attrib["name"]] = na
            for bond in residue.findall("Bond"):
                attrib = bond.attrib
                if "from" in attrib:
                    n1 = attrib["from"]
                    n2 = attrib["to"]
                else:
                    n1 = attrib["atomName1"]
                    n2 = attrib["atomName2"]
                graph.add_edge(name2idx[n1], name2idx[n2])
            for ext in residue.findall("ExternalBond"):
                name = ext.attrib["atomName"]
                graph.nodes()[name]["external_bond"] = True
            templates.append(graph)
    return templates
