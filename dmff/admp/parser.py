# jichen: deprecated
from xml.dom import minidom
import numpy as np
import warnings
from collections import defaultdict

def read_atom_line(line_full):
    """
    Read atom line from pdb format
    HETATM    1  H14 ORTE    0       6.301   0.693   1.919  1.00  0.00        H
    
    1-6 7-11 13-16 17 18-20 22 23-26 27 28-30 31-38 39-46 47-54 55-60 61-66 67-72 73-76 77-78 79-80
    ATOM serial name altLoc resName chainID resSeq iCode _ x y z occupancy tempFactor _ segID element charge
    """

    line = line_full.rstrip("\n")
    type_atm = line[0:6]
    if type_atm == "ATOM  " or type_atm == "HETATM":

        # Roy
        serial = line[7:12].strip()

        name = line[12:16].strip()

        altLoc = line[16]
        resName = line[17:21]
        chainID = line[21]  # Not used

        resSeq = int(line[22:26].split()[0])  # sequence identifier
        iCode = line[26]  # insertion code, not used

        # atomic coordinates
        try:
            coord = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=np.float64,
            )
        except ValueError:
            raise ValueError("Invalid or missing coordinate(s)")

        # occupancy & B factor
        try:
            occupancy = float(line[54:60])
        except ValueError:
            occupancy = None  # Rather than arbitrary zero or one

        if occupancy is not None and occupancy < 0:
            warnings.warn("Negative occupancy in one or more atoms")

        try:
            bfactor = float(line[60:66])
        except ValueError:
            # The PDB use a default of zero if the data is missing
            bfactor = 0.0

        segid = line[72:76]  # not used
        element = line[76:78].strip().upper()
        charge = line[79:81]

    else:
        raise ValueError("Only ATOM and HETATM supported")

    return (
        type_atm,
        serial,
        name,
        altLoc,
        resName.strip(),
        chainID,
        resSeq,
        iCode,
        coord,
        occupancy,
        bfactor,
        segid,
        element,
        charge,
    )

def read_pdb(file):
    """Read PDB files."""
    fileobj = open(file, 'r')
    orig = np.identity(3)
    trans = np.zeros(3)
    serials = []
    names = []
    altLocs = []
    resNames = []
    chainIDs = []
    resSeqs = []
    iCodes = []
    positions = []
    occupancies = []
    tempFactors = []
    segId = []
    elements = []
    charges = []
    cell = None
    pbc = None
    cellpar = []
    conects = {}
    # make sure that only one frame is read
    continue_read_atoms_flag = True
    # serial starts at 1 and we need to discard it and just keep align with positions
    id = 0
    
    for line in fileobj.readlines():
        if line.startswith('CRYST1'):
            cellpar = [float(line[6:15]),  # a
                       float(line[15:24]),  # b
                       float(line[24:33]),  # c
                       float(line[33:40]),  # alpha
                       float(line[40:47]),  # beta
                       float(line[47:54])]  # gamma

        for c in range(3):
            if line.startswith('ORIGX' + '123'[c]):
                orig[c] = [float(line[10:20]),
                           float(line[20:30]),
                           float(line[30:40])]
                trans[c] = float(line[45:55])

        if (
            line.startswith("ATOM")
            or line.startswith("HETATM")
            and continue_read_atoms_flag
        ):
            # Atom name is arbitrary and does not necessarily
            # contain the element symbol.  The specification
            # requires the element symbol to be in columns 77+78.
            # Fall back to Atom name for files that do not follow
            # the spec, e.g. packmol.

            # line_info = type_atm, serial, name, altLoc, resName, chainID, resSeq, iCode, coord, occupancy, tempFactor, segid, element, charge
            line_info = read_atom_line(line)

            # serials.append(int(line_info[1]))
            serials.append(id)
            id += 1
            names.append(line_info[2])
            resNames.append(line_info[4])
            resSeqs.append(line_info[6])
            position = np.dot(orig, line_info[8]) + trans
            positions.append(position)
            if line_info[9] is not None:
                occupancies.append(line_info[9])
            tempFactors.append(line_info[10])
            elements.append(line_info[-2])
            charges.append(line_info[-1] or 0)

        if line.startswith("END"):
            # End of configuration reached
            # According to the latest PDB file format (v3.30),
            # this line should start with 'ENDMDL' (not 'END'),
            # but in this way PDB trajectories from e.g. CP2K
            # are supported (also VMD supports this format).
            continue_read_atoms_flag = False
            pass

        if line.startswith("CONECT"):
            l = line.split()
            center_atom_idx = int(l[1])
            bonded_atom_idx = [int(i) for i in l[2:]]

            conects[center_atom_idx] = bonded_atom_idx
    fileobj.close()
    
    return {'serials': serials,
           'names': names,
           'resNames': resNames,
           'resSeqs': resSeqs,
           'positions': np.vstack(positions),
           'charges': charges,
           'connects': conects,
           'box': cellpar}

def set_axis_type(atoms):
    ZThenX = 0
    Bisector = 1
    ZBisect = 2
    ThreeFold = 3
    Zonly = 4
    NoAxisType = 5
    LastAxisTypeIndex = 6    
    kStrings = ['kz', 'kx', 'ky']
    
    for atom in atoms:
        atomType = atom['type']
        kIndices = [atomType]

        for kString in kStrings:
            
            if kString in atom and atom[kString] != '':
                kIndices.append(atom[kString])
        atom['axis_indices'] = kIndices

        # set axis type

        kIndicesLen = len(kIndices)

        if (kIndicesLen > 3):
            ky = kIndices[3]
            kyNegative = False
            if ky.startswith('-'):
                ky = kIndices[3] = ky[1:]
                kyNegative = True
        else:
            ky = ""

        if (kIndicesLen > 2):
            kx = kIndices[2]
            kxNegative = False
            if kx.startswith('-'):
                kx = kIndices[2] = kx[1:]
                kxNegative = True
        else:
            kx = ""

        if (kIndicesLen > 1):
            kz = kIndices[1]
            kzNegative = False
            if kz.startswith('-'):
                kz = kIndices[1] = kz[1:]
                kzNegative = True
        else:
            kz = ""

        while(len(kIndices) < 4):
            kIndices.append("")

        axisType = ZThenX
        if (not kz):
            axisType = NoAxisType
        if (kz and not kx):
            axisType = Zonly
        if (kz and kzNegative or kx and kxNegative):
            axisType = Bisector
        if (kx and kxNegative and ky and kyNegative):
            axisType = ZBisect
        if (kz and kzNegative and kx and kxNegative and ky and kyNegative):
            axisType = ThreeFold

        atom['axisType'] = axisType

    return atoms

def read_xml(fileobj):
    
    fileobj = minidom.parse(fileobj)

    multipoles = fileobj.getElementsByTagName("Multipole")

    residueTemplates = []
    atomTemplates = []
    
    for r in fileobj.getElementsByTagName('Residue'):
        
        resName = r.getAttribute("name")
        residueTemplate = {'resName': resName, 'atoms': [], }

        
        for a in r.getElementsByTagName('Atom'):
            atomName = a.getAttribute('name')
            atomType = a.getAttribute('type')
            atomTemplate = {'name': atomName, 'type': atomType}

            residueTemplate['atoms'].append(atomTemplate)
            atomTemplates.append(atomTemplate)
            
        topo = defaultdict(list)
        for b in r.getElementsByTagName('Bond'):
            
            from_ = b.getAttribute('from')
            to_ = b.getAttribute('to')
            topo[from_].append(to_)
            # topo[to_].append(from_)

        residueTemplate['topo'] = dict(topo)
        residueTemplates.append(residueTemplate)

    for i, multipole in enumerate(multipoles):
        
        multiDict = {
            "c0": float(multipole.getAttribute("c0")),
            "dX": float(multipole.getAttribute("dX")),
            "dY": float(multipole.getAttribute("dY")),
            "dZ": float(multipole.getAttribute("dZ")),
            "qXX": float(multipole.getAttribute("qXX")),
            "qXY": float(multipole.getAttribute("qXY")),
            "qYY": float(multipole.getAttribute("qYY")),
            "qXZ": float(multipole.getAttribute("qXZ")),
            "qYZ": float(multipole.getAttribute("qYZ")),
            "qZZ": float(multipole.getAttribute("qZZ")),
            "oXXX": float(multipole.getAttribute("oXXX")),
            "oXXY": float(multipole.getAttribute("oXXY")),
            "oXYY": float(multipole.getAttribute("oXYY")),
            "oYYY": float(multipole.getAttribute("oYYY")),
            "oXXZ": float(multipole.getAttribute("oXXZ")),
            "oXYZ": float(multipole.getAttribute("oXYZ")),
            "oYYZ": float(multipole.getAttribute("oYYZ")),
            "oXZZ": float(multipole.getAttribute("oXZZ")),
            "oYZZ": float(multipole.getAttribute("oYZZ")),
            "oZZZ": float(multipole.getAttribute("oZZZ")),
            "kx": multipole.getAttribute("kx"),
            "kz": multipole.getAttribute("kz"),
            "ky": multipole.getAttribute("ky")
        }
        
        for template in atomTemplates:
            if template['type'] == multipole.getAttribute("type"):
                template.update(multiDict)
        
    
    for p in fileobj.getElementsByTagName('Polarize'):
        
        pxx = p.getAttribute('polarizabilityXX')
        pyy = p.getAttribute('polarizabilityYY')
        pzz = p.getAttribute('polarizabilityZZ')
        thole = p.getAttribute('thole')
        polarDict = {'polarizabilityXX': pxx, 'polarizabilityYY': pyy, 'polarizabilityZZ':pzz, 'thole': thole}

        for template in atomTemplates:
            if template['type'] == p.getAttribute('type'):
                template.update(polarDict)
    
    set_axis_type(atomTemplates)

    return atomTemplates, residueTemplates
        
class Atom:
    
    def __init__(self, serial, name, resName, resSeq, position, charge, ) -> None:
        self.serial = serial
        self.name = name
        self.position = position
        self.charge = charge
        self.resName = resName
        self.charge = charge
        self.linkAtom = []
        self.resSeq = resSeq
        
    def __eq__(self, o):
        return o.serial == self.serial
        
    def link(self, atom):
        if atom not in self.linkAtom:
            self.linkAtom.append(atom)
        if self not in atom.linkAtom:
            atom.linkAtom.append(self)
            
    def __repr__(self) -> str:
        return f'< Atom{self.serial}: {self.name} >'
            
class Residue:
    
    def __init__(self, resName, resSeq) -> None:
        self.resName = resName
        self.resSeq = resSeq
        self.atoms = {}
        self.topo = []
        self.covalent_map = {}
        
    def add(self, serial, atom):
        self.atoms[serial] = atom
        
    def __next__(self):
        return next(self.atoms)
    
    def __getitem__(self, name):
        for atom in self.atoms.values():
            if atom.name == name:
                return atom
            
    def __repr__(self) -> str:
        return f'< Residue{self.resSeq}: {self.resName} >'

def init_residues(serials, names, resNames, resSeqs, positions, charges, atomTemplates, residueTemplates):
    
    residueDicts = {}
    atomDicts = {}
    
    for name, seq in zip(resNames, resSeqs):
        if seq not in residueDicts:
            residueDicts[seq] = Residue(name, seq)
        
    
    # build up residue
    for serial, name, resName, resSeq, position, charge in zip(serials, names, resNames, resSeqs, positions, charges):
        
        atom = Atom(serial, name, resName, resSeq, position, charge)

        for a in atomTemplates:
            if name == a['name']:
                for k, v in a.items():
                    setattr(atom, k, v)
        atomDicts[serial] = atom

        residueDicts[resSeq].add(atom.serial, atom)
        
    
    # build up topo
    for residue in residueDicts.values():
        
        for residueTemplate in residueTemplates:
            if residueTemplate['resName'] == residue.resName:
                template = residueTemplate

        for c, p in template['topo'].items():
            ctemp = template['atoms'][int(c)]
            catom = residue[ctemp['name']]

            for pp in p:
                ptemp = template['atoms'][int(pp)]
                patom = residue[ptemp['name']]
                catom.link(patom)
         
    # build up axis indices
    for residue in residueDicts.values():
        
        for atom in residue.atoms.values():
            indices = [index if index != '' else -1 for index in atom.axis_indices[1: ]]

            for patom in residue.atoms.values():
                if patom.serial == atom.serial:
                    continue
                for i in range(len(indices)):
                    if indices[i] == patom.type:
                        indices[i] = patom.serial
                        break
                            
            atom.axis_indices = indices
                    
    
    # build up covalent map in residue
        for i in residue.atoms.values():
            visited = [i.serial]
            residue.covalent_map[i.serial] = {}
            for j in i.linkAtom:
                residue.covalent_map[i.serial][j.serial] = 1
                visited.append(j.serial)
                for k in j.linkAtom:
                    if k.serial not in visited:
                        residue.covalent_map[i.serial][k.serial] = 2
                        visited.append(k.serial)
                    else:
                        continue
                    for l in k.linkAtom:
                        if l.serial not in visited:
                            residue.covalent_map[i.serial][l.serial] = 3
                            visited.append(l.serial)
                        else:
                            continue
                        for m in l.linkAtom:
                            if m.serial not in visited:
                                residue.covalent_map[i.serial][m.serial] = 4
                                visited.append(m.serial)
                            else:
                                continue
        
    return atomDicts, residueDicts

def assemble_covalent(residueDicts, natoms):
    
    covalents = [c.covalent_map for c in residueDicts.values()]
    
    covalent_map = np.zeros((natoms, natoms), dtype=int)
    
    for covalent in covalents:
        
        for c, p in covalent.items():
            
            for pp, dr in p.items():
                
                covalent_map[c][pp] = dr
                
    return covalent_map
    