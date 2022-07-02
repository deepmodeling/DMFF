import xml.etree.ElementTree as ET
import xml.dom.minidom
from dmff.utils import convertStr2Float, DMFFException
from typing import List
from itertools import permutations


class SelectError(BaseException):
    pass


class Node:
    def __init__(self, tag, **attrs):

        self.tag = tag
        #TODO: self.parent = parent
        self.attrs = attrs
        self.children = []

    def add_child(self, child: 'Node'):

        self.children.append(child)

    def add_children(self, children: List['Node']):
        self.children.extend(children)

    def get_children(self, key):
        return [c for c in self.children if c.tag == key]

    def get_child(self, key):
        for child in self.children:
            if child.tag == key:
                return child

    def __getitem__(self, key):

        return self.attrs[key]

    def __repr__(self):
        return f'<{self.tag}: {self.attrs}, with {len(self.children)} subnodes>'

    def __iter__(self):
        return iter(self.children)

    def __contains__(self, key):
        if self.get_child(key):
            return True
        else:
            return False


class ForcefieldTree(Node):
    def __init__(self, tag, **attrs):

        super().__init__(tag, **attrs)

    def get_nodes(self, parser):
        steps = parser.split("/")
        val = self
        for nstep, step in enumerate(steps):
            name, index = step, -1
            if "[" in step:
                name, index = step.split("[")
                index = int(index[:-1])
            val = [c for c in val.children if c.tag == name]
            if index >= 0:
                val = val[index]
            elif nstep < len(steps) - 1:
                val = val[0]
        return val

    def get_attribs(self, parser, attrname):
        sel = self.get_nodes(parser)
        if isinstance(attrname, list):
            ret = []
            for item in sel:
                vals = [convertStr2Float(item.attrs[an]) if an in item.attrs else None for an in attrname]
                ret.append(vals)
            return ret
        else:
            attrs = [convertStr2Float(n.attrs[attrname]) if attrname in n.attrs else None for n in sel]
            return attrs

    def set_node(self, parser, values):
        nodes = self.get_nodes(parser)
        for nit in range(len(values)):
            for key in values[nit]:
                nodes[nit].attrs[key] = f"{values[nit][key]}"

    def set_attrib(self, parser, attrname, values):
        valdicts = [{attrname: i} for i in values]
        self.set_node(parser, valdicts)


class XMLParser:
    def __init__(self, ffTree: ForcefieldTree):

        self.ff = ffTree

    def parse_node(self, root):

        node = ForcefieldTree(tag=root.tag, **root.attrib)
        children = list(map(self.parse_node, root))
        if children:
            node.add_children(children)
        return node

    def parse(self, *xmls):
        for xml in xmls:
            root = ET.parse(xml).getroot()
            for leaf in root:
                n = self.parse_node(leaf)
                ifExist = False
                for nchild, child in enumerate(self.ff.children):
                    if child.tag == n.tag:
                        ifExist = True
                        break
                if ifExist:
                    self.ff.children[nchild].add_children(n.children)
                else:
                    self.ff.add_child(n)

    def write_node(self, parent, node):
        parent = ET.SubElement(parent, node.tag, node.attrs)
        for sibiling in node:
            tmp = ET.SubElement(parent, sibiling.tag, sibiling.attrs)
            for child in sibiling:
                self.write_node(tmp, child)

    @staticmethod
    def pretty_print(element):
        initstr = ET.tostring(element, "unicode")
        pretxml = xml.dom.minidom.parseString(initstr)
        pretstr = pretxml.toprettyxml()
        return pretstr

    def write(self, path):

        root = ET.Element('Forcefield')

        for child in self.ff:
            if child.tag == 'Residues':
                Residues = ET.SubElement(root, 'Residues')
                for residue in child:
                    self.write_node(Residues, residue)
            else:
                self.write_node(root, child)
        outstr = self.pretty_print(root)
        with open(path, "w") as f:
            f.write(outstr)


class TypeMatcher:
    def __init__(self, fftree: ForcefieldTree, parser):
        """
        Freeze type matching list.
        """
        atypes = fftree.get_attribs("AtomTypes/Type", "name")
        aclasses = fftree.get_attribs("AtomTypes/Type", "class")
        self.class2type = {}
        for nline in range(len(atypes)):
            if aclasses[nline] not in self.class2type:
                self.class2type[aclasses[nline]] = []
            self.class2type[aclasses[nline]].append(atypes[nline])
        self.class2type[""] = atypes
        funcs = fftree.get_nodes(parser)
        self.functions = []
        for node in funcs:
            tmp = []
            for key in node.attrs:
                if len(key) > 4 and "type" == key[:4]:
                    nit = int(key[4:])
                    if len(node.attrs[key]) == 0:
                        tmp.append((nit, atypes))
                    else:
                        tmp.append((nit, [node.attrs[key]]))
                elif key == "type":
                    tmp.append((1, [node.attrs[key]]))
                elif len(key) > 5 and "class" == key[:5]:
                    nit = int(key[5:])
                    tmp.append((nit, self.class2type[node.attrs[key]]))
                elif key == "class":
                    tmp.append((1, self.class2type[node.attrs[key]]))
            tmp = sorted(tmp, key=lambda x: x[0])
            self.functions.append([i[1] for i in tmp])

    def matchGeneral(self, types):
        matches = []
        for nterm, term in enumerate(self.functions):
            ifMatch, ifForward = self._match(types, term)
            if ifMatch:
                matches.append((ifMatch, ifForward, nterm))
        if len(matches) == 0:
            return False, False, -1
        return matches[-1]

    def _match(self, types, term):
        if len(types) != len(term):
            raise DMFFException(
                "The length of matching types is not equivalent to the forcefield term."
            )
        # Forward
        ifMatchForward = True
        for ntypes in range(len(types)):
            if len(types[ntypes]) == 0:
                continue
            if types[ntypes] not in term[ntypes]:
                ifMatchForward = False
                break
        ifMatchReverse = True
        for ntypes in range(len(types)):
            if len(types[len(types) - ntypes - 1]) == 0:
                continue
            if types[len(types) - ntypes - 1] not in term[ntypes]:
                ifMatchReverse = False
                break
        return ifMatchForward or ifMatchReverse, ifMatchForward

    def matchImproper(self, torsion, data, ordering="amber"):
        type1, type2, type3, type4 = [data.atomType[data.atoms[torsion[i]]] for i in range(4)]
        match = None
        for nterm, term in enumerate(self.functions):
            types1 = term[0]
            types2 = term[1]
            types3 = term[2]
            types4 = term[3]
            hasWildcard = (len(self.class2type[""])
                           in (len(types1), len(types2), len(types3),
                               len(types4)))
            if type1 in types1:
                for (t2, t3, t4) in permutations(((type2, 1), (type3, 2), (type4, 3))):
                    if t2[0] in types2 and t3[0] in types3 and t4[0] in types4:
                        if ordering == 'default':
                            # Workaround to be more consistent with AMBER.  It uses wildcards to define most of its
                            # impropers, which leaves the ordering ambiguous.  It then follows some bizarre rules
                            # to pick the order.
                            a1 = torsion[t2[1]]
                            a2 = torsion[t3[1]]
                            e1 = data.atoms[a1].element
                            e2 = data.atoms[a2].element
                            if e1 == e2 and a1 > a2:
                                (a1, a2) = (a2, a1)
                            elif e1.symbol != "C" and (e2.symbol == "C" or e1.mass < e2.mass):
                                (a1, a2) = (a2, a1)
                            match = (a1, a2, torsion[0], torsion[t4[1]], nterm)
                            break
                        elif ordering == 'charmm':
                            if hasWildcard:
                                # Workaround to be more consistent with AMBER.  It uses wildcards to define most of its
                                # impropers, which leaves the ordering ambiguous.  It then follows some bizarre rules
                                # to pick the order.
                                a1 = torsion[t2[1]]
                                a2 = torsion[t3[1]]
                                e1 = data.atoms[a1].element
                                e2 = data.atoms[a2].element
                                if e1 == e2 and a1 > a2:
                                    (a1, a2) = (a2, a1)
                                elif e1.symbol != "C" and (e2.symbol == "C" or e1.mass < e2.mass):
                                    (a1, a2) = (a2, a1)
                                match = (a1, a2, torsion[0], torsion[t4[1]], nterm)
                            else:
                                # There are no wildcards, so the order is unambiguous.
                                match = (torsion[0], torsion[t2[1]], torsion[t3[1]], torsion[t4[1]], nterm)
                            break
                        elif ordering == 'amber':
                            # topology atom indexes
                            a2 = torsion[t2[1]]
                            a3 = torsion[t3[1]]
                            a4 = torsion[t4[1]]
                            # residue indexes
                            r2 = data.atoms[a2].residue.index
                            r3 = data.atoms[a3].residue.index
                            r4 = data.atoms[a4].residue.index
                            # template atom indexes
                            ta2 = data.atomTemplateIndexes[data.atoms[a2]]
                            ta3 = data.atomTemplateIndexes[data.atoms[a3]]
                            ta4 = data.atomTemplateIndexes[data.atoms[a4]]
                            # elements
                            e2 = data.atoms[a2].element
                            e3 = data.atoms[a3].element
                            e4 = data.atoms[a4].element
                            if not hasWildcard:
                                if t2[0] == t4[0] and (r2 > r4 or (r2 == r4 and ta2 > ta4)):
                                    (a2, a4) = (a4, a2)
                                    r2 = data.atoms[a2].residue.index
                                    r4 = data.atoms[a4].residue.index
                                    ta2 = data.atomTemplateIndexes[data.atoms[a2]]
                                    ta4 = data.atomTemplateIndexes[data.atoms[a4]]
                                if t3[0] == t4[0] and (r3 > r4 or (r3 == r4 and ta3 > ta4)):
                                    (a3, a4) = (a4, a3)
                                    r3 = data.atoms[a3].residue.index
                                    r4 = data.atoms[a4].residue.index
                                    ta3 = data.atomTemplateIndexes[data.atoms[a3]]
                                    ta4 = data.atomTemplateIndexes[data.atoms[a4]]
                                if t2[0] == t3[0] and (r2 > r3 or (r2 == r3 and ta2 > ta3)):
                                    (a2, a3) = (a3, a2)
                            else:
                                if e2 == e4 and (r2 > r4 or (r2 == r4 and ta2 > ta4)):
                                    (a2, a4) = (a4, a2)
                                    r2 = data.atoms[a2].residue.index
                                    r4 = data.atoms[a4].residue.index
                                    ta2 = data.atomTemplateIndexes[data.atoms[a2]]
                                    ta4 = data.atomTemplateIndexes[data.atoms[a4]]
                                if e3 == e4 and (r3 > r4 or (r3 == r4 and ta3 > ta4)):
                                    (a3, a4) = (a4, a3)
                                    r3 = data.atoms[a3].residue.index
                                    r4 = data.atoms[a4].residue.index
                                    ta3 = data.atomTemplateIndexes[data.atoms[a3]]
                                    ta4 = data.atomTemplateIndexes[data.atoms[a4]]
                                if r2 > r3 or (r2 == r3 and ta2 > ta3):
                                    (a2, a3) = (a3, a2)
                            match = (a2, a3, torsion[0], a4, nterm)
                            break
                        elif ordering == 'smirnoff':
                            # topology atom indexes
                            a1 = torsion[0]
                            a2 = torsion[t2[1]]
                            a3 = torsion[t3[1]]
                            a4 = torsion[t4[1]]
                            # enforce exact match
                            match = (a1, a2, a3, a4, nterm)
                            break
        return match