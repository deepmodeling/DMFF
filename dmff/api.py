#!/usr/bin/env python
import openmm.app as app
import openmm.unit as unit
import numpy as np
import jax.numpy as jnp
from collections import defaultdict
from .admp.disp_pme import ADMPDispPmeForce
from .admp.multipole import convert_cart2harm, rot_local2global
from .admp.pairwise import TT_damping_qq_c6_kernel, generate_pairwise_interaction
from .admp.pme import ADMPPmeForce
from .admp.spatial import generate_construct_local_frames
from .admp.recip import Ck_1, generate_pme_recip
from jax_md import space, partition
from jax import grad
import linecache
import sys


def get_line_context(file_path, line_number):
    return linecache.getline(file_path, line_number).strip()


def build_covalent_map(data, max_neighbor):
    n_atoms = len(data.atoms)
    covalent_map = np.zeros((n_atoms, n_atoms), dtype=int)
    for bond in data.bonds:
        covalent_map[bond.atom1, bond.atom2] = 1
        covalent_map[bond.atom2, bond.atom1] = 1
    for n_curr in range(1, max_neighbor):
        for i in range(n_atoms):
            # current neighbors
            j_list = np.where(
                np.logical_and(covalent_map[i] <= n_curr,
                               covalent_map[i] > 0))[0]
            for j in j_list:
                k_list = np.where(covalent_map[j] == 1)[0]
                for k in k_list:
                    if k != i and k not in j_list:
                        covalent_map[i, k] = n_curr + 1
                        covalent_map[k, i] = n_curr + 1
    return covalent_map


class ADMPDispGenerator:
    def __init__(self, hamiltonian):
        self.ff = hamiltonian
        self.params = {
            "A": [],
            "B": [],
            "Q": [],
            "C6": [],
            "C8": [],
            "C10": []
        }
        self._jaxPotential = None
        self.types = []
        self.ethresh = 5e-4
        self.pmax = 10

    def registerAtomType(self, atom):
        self.types.append(atom["type"])
        self.params["A"].append(float(atom["A"]))
        self.params["B"].append(float(atom["B"]))
        self.params["Q"].append(float(atom["Q"]))
        self.params["C6"].append(float(atom["C6"]))
        self.params["C8"].append(float(atom["C8"]))
        self.params["C10"].append(float(atom["C10"]))

    @staticmethod
    def parseElement(element, hamiltonian):
        generator = ADMPDispGenerator(hamiltonian)
        hamiltonian.registerGenerator(generator)
        # covalent scales
        mScales = []
        for i in range(2, 7):
            mScales.append(float(element.attrib["mScale1%d" % i]))
        mScales.append(1.0)
        generator.params["mScales"] = mScales
        for atomtype in element.findall("Atom"):
            generator.registerAtomType(atomtype.attrib)
        # jax it!
        for k in generator.params.keys():
            generator.params[k] = jnp.array(generator.params[k])
        generator.types = np.array(generator.types)

    def createForce(self, system, data, nonbondedMethod, nonbondedCutoff,
                    args):

        n_atoms = len(data.atoms)
        # build index map
        map_atomtype = np.zeros(n_atoms, dtype=int)
        for i in range(n_atoms):
            atype = data.atomType[data.atoms[i]]
            map_atomtype[i] = np.where(self.types == atype)[0][0]
        # build covalent map
        covalent_map = build_covalent_map(data, 6)

        # here box is only used to setup ewald parameters, no need to be differentiable
        a, b, c = system.getDefaultPeriodicBoxVectors()
        box = jnp.array([a._value, b._value, c._value]) * 10
        # get the admp calculator
        rc = nonbondedCutoff.value_in_unit(unit.angstrom)

        # get calculator
        if 'ethresh' in args:
            self.ethresh = args['ethresh']

        Force_DispPME = ADMPDispPmeForce(box, covalent_map, rc, self.ethresh,
                                         self.pmax)
        self.disp_pme_force = Force_DispPME
        # debugging
        # Force_DispPME.update_env('kappa', 0.657065221219616)
        # Force_DispPME.update_env('K1', 96)
        # Force_DispPME.update_env('K2', 96)
        # Force_DispPME.update_env('K3', 96)
        pot_fn_lr = Force_DispPME.get_energy
        pot_fn_sr = generate_pairwise_interaction(TT_damping_qq_c6_kernel,
                                                  covalent_map,
                                                  static_args={})

        def potential_fn(positions, box, pairs, params):
            mScales = params["mScales"]
            a_list = (params["A"][map_atomtype] / 2625.5
                      )  # kj/mol to au, as expected by TT_damping kernel
            b_list = params["B"][map_atomtype] * 0.0529177249  # nm^-1 to au
            q_list = params["Q"][map_atomtype]
            c6_list = jnp.sqrt(params["C6"][map_atomtype] * 1e6)
            c8_list = jnp.sqrt(params["C8"][map_atomtype] * 1e8)
            c10_list = jnp.sqrt(params["C10"][map_atomtype] * 1e10)
            c_list = jnp.vstack((c6_list, c8_list, c10_list))

            E_sr = pot_fn_sr(positions, box, pairs, mScales, a_list, b_list,
                             q_list, c_list[0])
            E_lr = pot_fn_lr(positions, box, pairs, c_list.T, mScales)
            return E_sr - E_lr

        self._jaxPotential = potential_fn
        # self._top_data = data

    def getJaxPotential(self):
        return self._jaxPotential

    def renderXML(self):
        # generate xml force field file
        pass


# register all parsers
app.forcefield.parsers["ADMPDispForce"] = ADMPDispGenerator.parseElement


class ADMPPmeGenerator:
    def __init__(self, hamiltonian):
        self.ff = hamiltonian
        self.kStrings = {
            "kz": [],
            "kx": [],
            "ky": [],
        }
        self._input_params = {
            "c0": [],
            "dX": [],
            "dY": [],
            "dZ": [],
            "qXX": [],
            "qXY": [],
            "qYY": [],
            "qXZ": [],
            "qYZ": [],
            "qZZ": [],
            "oXXX": [],
            "oXXY": [],
            "oXYY": [],
            "oYYY": [],
            "oXXZ": [],
            "oXYZ": [],
            "oYYZ": [],
            "oXZZ": [],
            "oYZZ": [],
            "oZZZ": [],
            "thole": [],
            "polarizabilityXX": [],
            "polarizabilityYY": [],
            "polarizabilityZZ": [],

        }
        self.params = {
            "mScales": [],
            "pScales": [],
            "dScales": [],
        }
        # if more or optional input params
        # self._input_params = defaultDict(list)
        self._jaxPotential = None
        self.types = []
        self.ethresh = 5e-4
        self.lpol = False
        self.ref_dip = ''

    def registerAtomType(self, atom: dict):

        self.types.append(atom.pop("type"))

        kStrings = ["kz", "kx", "ky"]
        for kString in kStrings:
            if kString in atom:
                self.kStrings[kString].append(atom.pop(kString))
            else:
                self.kStrings[kString].append("0")

        for k, v in atom.items():
            self._input_params[k].append(float(v))

    @staticmethod
    def parseElement(element, hamiltonian):
        generator = ADMPPmeGenerator(hamiltonian)
        generator.lmax = int(element.attrib.get('lmax'))
        generator.defaultTholeWidth = 5

        hamiltonian.registerGenerator(generator)

        for i in range(2, 7):
            generator.params["mScales"].append(
                float(element.attrib["mScale1%d" % i]))
            generator.params["pScales"].append(
                float(element.attrib["pScale1%d" % i]))
            generator.params["dScales"].append(
                float(element.attrib["dScale1%d" % i]))

        # make sure the last digit is 1.0
        generator.params['mScales'].append(1.0)
        generator.params['pScales'].append(1.0)
        generator.params['dScales'].append(1.0)

        if element.findall('Polarize'):
            generator.lpol = True
        else:
            generator.lpol = False

        for atomType in element.findall("Atom"):
            atomAttrib = atomType.attrib
            # if not set
            atomAttrib.update({'polarizabilityXX': 0, 'polarizabilityYY': 0, 'polarizabilityZZ': 0})
            for polarInfo in element.findall("Polarize"):
                polarAttrib = polarInfo.attrib
                if polarInfo.attrib['type'] == atomAttrib['type']:
                    # cover default
                    atomAttrib.update(polarAttrib)
                    break
            generator.registerAtomType(atomAttrib)

        for k in generator._input_params.keys():
            generator._input_params[k] = jnp.array(generator._input_params[k])
        generator.types = np.array(generator.types)
        
        n_atoms = len(element.findall('Atom'))
        
        # map atom multipole moments
        if generator.lmax == 0:
            n_mtps = 1
        elif generator.lmax == 1:
            n_mtps = 4
        elif generator.lmax == 2:
            n_mtps = 10
        Q = np.zeros((n_atoms, n_mtps))

        Q[:, 0] = generator._input_params["c0"]
        if generator.lmax >= 1:
            Q[:, 1] = generator._input_params["dX"] * 10
            Q[:, 2] = generator._input_params["dY"] * 10
            Q[:, 3] = generator._input_params["dZ"] * 10
        if generator.lmax >= 2:
            Q[:, 4] = generator._input_params["qXX"] * 300
            Q[:, 5] = generator._input_params["qYY"] * 300
            Q[:, 6] = generator._input_params["qZZ"] * 300
            Q[:, 7] = generator._input_params["qXY"] * 300
            Q[:, 8] = generator._input_params["qXZ"] * 300
            Q[:, 9] = generator._input_params["qYZ"] * 300

        # add all differentiable params to self.params
        Q_local = convert_cart2harm(Q, generator.lmax)
        generator.params['Q_local'] = Q_local
       
        if generator.lpol:
            pol = jnp.vstack((generator._input_params['polarizabilityXX'], generator._input_params['polarizabilityYY'], generator._input_params['polarizabilityZZ'])).T
            pol = 1000*jnp.mean(pol,axis=1)
            tholes = jnp.array(generator._input_params['thole'])
            generator.params['pol'] = pol
            generator.params['tholes'] = tholes
        else:
            pol = None
            tholes = None

        # generator.params['']
        for k in generator.params.keys():
            generator.params[k] = jnp.array(generator.params[k])
        

    def createForce(self, system, data, nonbondedMethod, nonbondedCutoff,
                    args):

        n_atoms = len(data.atoms)
        map_atomtype = np.zeros(n_atoms, dtype=int)

        for i in range(n_atoms):
            atype = data.atomType[data.atoms[i]]
            map_atomtype[i] = np.where(self.types == atype)[0][0]

        # here box is only used to setup ewald parameters, no need to be differentiable
        a, b, c = system.getDefaultPeriodicBoxVectors()
        box = jnp.array([a._value, b._value, c._value]) * 10

        # get the admp calculator
        rc = nonbondedCutoff.value_in_unit(unit.angstrom)

        # build covalent map
        covalent_map = build_covalent_map(data, 6)

        # build intra-molecule axis
        # the following code is the direct transplant of forcefield.py in openmm 7.4.0

        if self.lmax > 0:

            # setting up axis_indices and axis_type
            ZThenX            = 0
            Bisector          = 1
            ZBisect           = 2
            ThreeFold         = 3
            Zonly             = 4
            NoAxisType        = 5
            LastAxisTypeIndex = 6

            self.axis_types = []
            self.axis_indices = []
            for i_atom in range(n_atoms):
                atom = data.atoms[i_atom]
                t = data.atomType[atom]
                # if t is in type list?
                if t in self.types:
                    itypes = np.where(self.types == t)[0]
                    hit = 0
                    # try to assign multipole parameters via only 1-2 connected atoms
                    for itype in itypes:
                        if hit != 0:
                            break
                        kz = int(self.kStrings['kz'][itype])
                        kx = int(self.kStrings['kx'][itype])
                        ky = int(self.kStrings['ky'][itype])
                        neighbors = np.where(covalent_map[i_atom] == 1)[0]
                        zaxis = -1
                        xaxis = -1
                        yaxis = -1
                        for z_index in neighbors:
                            if hit != 0:
                                break
                            z_type = int(data.atomType[data.atoms[z_index]])
                            if z_type == abs(kz):  # find the z atom, start searching for x
                                for x_index in neighbors:
                                    if x_index == z_index or hit != 0:
                                        continue
                                    x_type = int(data.atomType[data.atoms[x_index]])
                                    if x_type == abs(kx): # find the x atom, start searching for y
                                        if ky == 0:
                                            zaxis = z_index
                                            xaxis = x_index
                                            # cannot ditinguish x and z? use the smaller index for z, and the larger index for x
                                            if (x_type == z_type and xaxis < zaxis):
                                                swap = z_axis
                                                z_axis = x_axis
                                                x_axis = swap
                                            # otherwise, try to see if we can find an even smaller index for x?
                                            else:
                                                for x_index in neighbors:
                                                    x_type1 = int(data.atomType[data.atoms[x_index]])
                                                    if x_type1 == abs(kx) and x_index != z_index and x_index < xaxis:
                                                        xaxis = x_index
                                            hit = 1 # hit, finish matching
                                            matched_itype = itype
                                        else:
                                            for y_index in neighbors:
                                                if (y_index == z_index or y_index == x_index or hit != 0):
                                                    continue
                                                y_type = int(data.atomType[data.atoms[y_index]])
                                                if y_type == abs(ky):
                                                    zaxis = z_index
                                                    xaxis = x_index
                                                    yaxis = y_index
                                                    hit = 2
                                                    matched_itype = itype
                    # assign multipole parameters via 1-2 and 1-3 connected atoms
                    for itype in itypes:
                        if hit != 0:
                            break
                        kz = int(self.kStrings['kz'][itype])
                        kx = int(self.kStrings['kx'][itype])
                        ky = int(self.kStrings['ky'][itype])
                        neighbors_1st = np.where(covalent_map[i_atom] == 1)[0]
                        neighbors_2nd = np.where(covalent_map[i_atom] == 2)[0]
                        zaxis = -1
                        xaxis = -1
                        yaxis = -1
                        for z_index in neighbors_1st:
                            if hit != 0:
                                break
                            z_type = int(data.atomType[data.atoms[z_index]])
                            if z_type == abs(kz):
                                for x_index in neighbors_2nd:
                                    if x_index == z_index or hit != 0:
                                        continue
                                    x_type = int(data.atomType[data.atoms[x_index]])
                                    # we ask x to be in 2'nd neighbor, and x is z's neighbor
                                    if x_type == abs(kx) and covalent_map[z_index, x_index] == 1:
                                        if ky == 0:
                                            zaxis = z_index
                                            xaxis = x_index
                                            # select smallest x index
                                            for x_index in neighbors_2nd:
                                                x_type1 = int(data.atomType[data.atoms[x_index]])
                                                if x_type1 == abs(kx) and x_index != z_index and covalent_map[x_index, z_index] == 1 and x_index < xaxis:
                                                    xaxis = x_index
                                            hit = 3
                                            matched_itype = itype
                                        else:
                                            for y_index in neighbors_2nd:
                                                if y_index == z_index or y_index == x_index or hit != 0:
                                                    continue
                                                y_type = int(data.atomType[data.atoms[y_index]])
                                                if y_type == abs(ky) and covalent_map[y_index, z_index] == 1:
                                                    zaxis = z_index
                                                    xaxis = x_index
                                                    yaxis = y_index
                                                    hit = 4
                                                    matched_itype = itype
                    # assign multipole parameters via only a z-defining atom
                    for itype in itypes:
                        if hit != 0:
                            break
                        kz = int(self.kStrings['kz'][itype])
                        kx = int(self.kStrings['kx'][itype])
                        zaxis = -1
                        xaxis = -1
                        yaxis = -1
                        neighbors = np.where(covalent_map[i_atom] == 1)[0]
                        for z_index in neighbors:
                            if hit != 0:
                                break
                            z_type = int(data.atomType[data.atoms[z_index]])
                            if kx == 0 and z_type == abs(kz):
                                zaxis = z_index
                                hit = 5
                                matched_itype = itype
                    # assign multipole parameters via no connected atoms
                    for itype in itypes:
                        if hit != 0:
                            break
                        kz = int(self.kStrings['kz'][itype])
                        zaxis = -1
                        xaxis = -1
                        yaxis = -1
                        if kz == 0:
                            hit = 6
                            matched_itype = itype 
                    # add particle if there was a hit
                    if hit != 0:
                        map_atomtype[i_atom] = matched_itype
                        self.axis_indices.append([zaxis, xaxis, yaxis])

                        kz = int(self.kStrings['kz'][matched_itype])
                        kx = int(self.kStrings['kx'][matched_itype])
                        ky = int(self.kStrings['ky'][matched_itype])
                        axisType = ZThenX        
                        if (kz == 0):                                    
                            axisType = NoAxisType
                        if (kz != 0 and kx == 0):                        
                            axisType = ZOnly     
                        if (kz < 0 or kx < 0):                           
                            axisType = Bisector  
                        if (kx < 0 and ky < 0):                          
                            axisType = ZBisect   
                        if (kz < 0 and kx < 0 and ky  < 0):              
                            axisType = ThreeFold 
                        self.axis_types.append(axisType)
                        
                    else:
                        sys.exit('Atom %d not matched in forcefield!'%i_atom)
                        
                else:
                    sys.exit('Atom %d not matched in forcefield!'%i_atom)
            self.axis_indices = np.array(self.axis_indices)
            self.axis_types = np.array(self.axis_types)
        else:
            self.axis_types = None
            self.axis_indices = None


        # get calculator
        if 'ethresh' in args:
            self.ethresh = args['ethresh']

        pme_force = ADMPPmeForce(box, self.axis_types, self.axis_indices,
                                 covalent_map, rc, self.ethresh, self.lmax,
                                 self.lpol)
        self.pme_force = pme_force

        def potential_fn(positions, box, pairs, params):

            mScales = params["mScales"]
            Q_local = params["Q_local"][map_atomtype]
            if self.lpol:
                pScales = params["pScales"]
                dScales = params["dScales"]
                pol = params["pol"][map_atomtype]
                tholes = params["tholes"][map_atomtype]

                return pme_force.get_energy(positions, box, pairs, Q_local, pol, tholes, mScales, pScales, dScales, pme_force.U_ind)
            else: 
                return pme_force.get_energy(positions, box, pairs, Q_local, mScales)

        self._jaxPotential = potential_fn

    def getJaxPotential(self):
        return self._jaxPotential

    def renderXML(self):
        pass


app.forcefield.parsers["ADMPPmeForce"] = ADMPPmeGenerator.parseElement


class HarmonicBondGenerator:
    def __init__(self, hamiltonian):
        self.ff = hamiltonian
        self.params = {'k': [], 'length': []}
        self._jaxPotential = None
        self.types = []

    def registerBondType(self, bond):
        types = self.ff._findAtomTypes(bond, 2)
        self.types.append(types)
        self.params['k'].append(float(bond['k']))
        self.params['length'].append(float(bond['length']))

    @staticmethod
    def parseElement(element, hamiltonian):
        generator = HarmonicBondGenerator(hamiltonian)
        hamiltonian.registerGenerator(generator)
        for bondtype in element.findall("Bond"):
            generator.registerBondType(bondtype.attrib)
        # jax it!
        for k in generator.params.keys():
            generator.params[k] = jnp.array(generator.params[k])
        generator.types = np.array(generator.types)

    def createForce(self, system, data, nonbondedMethod, nonbondedCutoff,
                    args):

        n_bonds = len(data.bonds)
        # build map
        map_atom1 = np.zeros(n_bonds, dtype=int)
        map_atom2 = np.zeros(n_bonds, dtype=int)
        map_param = np.zeros(n_bonds, dtype=int)
        for i in range(n_bonds):
            idx1 = data.bonds[i].atom1
            idx2 = data.bonds[i].atom2
            type1 = data.atomType[data.atoms[idx1]]
            type2 = data.atomType[data.atoms[idx2]]
            ifFound = False
            for ii in range(len(self.types)):
                if (type1 in self.types[ii][0] and type2 in self.types[ii][1]
                    ) or (type1 in self.types[ii][1]
                          and type2 in self.types[ii][0]):
                    map_atom1[i] = idx1
                    map_atom2[i] = idx2
                    map_param[i] = ii
                    ifFound = True
                    break
            if not ifFound:
                raise BaseException("No parameter for bond %i - %i" %
                                    (idx1, idx2))

        bforce = HarmonicBondJaxForce(map_atom1, map_atom2, map_param)

        def potential_fn(positions, box, pairs, params):
            return bforce.get_energy(positions, box, pairs, params["k"],
                                     params["length"])

        self._jaxPotential = potential_fn
        # self._top_data = data

    def getJaxPotential(self):
        return self._jaxPotential

    def renderXML(self):
        # generate xml force field file
        pass


# register all parsers
app.forcefield.parsers[
    "HarmonicBondForce"] = HarmonicBondGenerator.parseElement


class HarmonicAngleGenerator:
    def __init__(self, hamiltonian):
        self.ff = hamiltonian
        self.params = {'k': [], 'theta0': []}
        self._jaxPotential = None
        self.types = []

    def registerAngleType(self, angle):
        types = self.ff._findAtomTypes(angle, 3)
        self.types.append(types)
        self.params['k'].append(float(angle['k']))
        self.params['theta0'].append(float(angle['theta0']))

    @staticmethod
    def parseElement(element, hamiltonian):
        generator = HarmonicAngleGenerator(hamiltonian)
        hamiltonian.registerGenerator(generator)
        for bondtype in element.findall("Angle"):
            generator.registerAngleType(bondtype.attrib)
        # jax it!
        for k in generator.params.keys():
            generator.params[k] = jnp.array(generator.params[k])
        generator.types = np.array(generator.types)

    def createForce(self, system, data, nonbondedMethod, nonbondedCutoff,
                    args):

        n_angles = len(data.angles)
        # build map
        map_atom1 = np.zeros(n_angles, dtype=int)
        map_atom2 = np.zeros(n_angles, dtype=int)
        map_atom3 = np.zeros(n_angles, dtype=int)
        map_param = np.zeros(n_angles, dtype=int)
        for i in range(n_angles):
            idx1 = data.angles[i].atom1
            idx2 = data.angles[i].atom2
            idx3 = data.angles[i].atom3
            type1 = data.atomType[data.atoms[idx1]]
            type2 = data.atomType[data.atoms[idx2]]
            type3 = data.atomType[data.atoms[idx3]]
            ifFound = False
            for ii in range(len(self.types)):
                if type2 in self.types[ii][1]:
                    if (type1 in self.types[ii][0]
                            and type3 in self.types[ii][2]) or (
                                type1 in self.types[ii][2]
                                and type3 in self.types[ii][0]):
                        map_atom1[i] = idx1
                        map_atom2[i] = idx2
                        map_atom3[i] = idx3
                        map_param[i] = ii
                        ifFound = True
                        break
            if not ifFound:
                raise BaseException("No parameter for angle %i - %i - %i" %
                                    (idx1, idx2, idx3))

        aforce = HarmonicAngleJaxForce(map_atom1, map_atom2, map_atom3,
                                       map_param)

        def potential_fn(positions, box, pairs, params):
            return aforce.get_energy(positions, box, pairs, params["k"],
                                     params["theta0"])

        self._jaxPotential = potential_fn
        # self._top_data = data

    def getJaxPotential(self):
        return self._jaxPotential

    def renderXML(self):
        # generate xml force field file
        pass


# register all parsers
app.forcefield.parsers[
    "HarmonicAngleForce"] = HarmonicAngleGenerator.parseElement


class PeriodicTorsion(object):
    """A PeriodicTorsion records the information for a periodic torsion definition."""
    def __init__(self, types):
        self.types1 = types[0]
        self.types2 = types[1]
        self.types3 = types[2]
        self.types4 = types[3]
        self.periodicity = []
        self.phase = []
        self.k = []
        self.ordering = 'default'


## @private
class PeriodicTorsionGenerator(object):
    """A PeriodicTorsionGenerator constructs a PeriodicTorsionForce."""
    def __init__(self, hamiltonian):
        self.ff = hamiltonian
        self.proper = []
        self.improper = []
        self.params = {'k': [], 'theta0': []}
        self.propersForAtomType = defaultdict(set)

    def registerProperTorsion(self, parameters):
        torsion = self.ff._parseTorsion(parameters)
        if torsion is not None:
            index = len(self.proper)
            self.proper.append(torsion)
            for t in torsion.types2:
                self.propersForAtomType[t].add(index)
            for t in torsion.types3:
                self.propersForAtomType[t].add(index)

    def registerImproperTorsion(self, parameters, ordering='default'):
        torsion = self.ff._parseTorsion(parameters)
        if torsion is not None:
            if ordering in ['default', 'charmm', 'amber', 'smirnoff']:
                torsion.ordering = ordering
            else:
                raise ValueError(
                    'Illegal ordering type %s for improper torsion %s' %
                    (ordering, torsion))
            self.improper.append(torsion)

    @staticmethod
    def parseElement(element, ff):
        existing = [
            f for f in ff._forces if isinstance(f, PeriodicTorsionGenerator)
        ]
        if len(existing) == 0:
            generator = PeriodicTorsionGenerator(ff)
            ff.registerGenerator(generator)
        else:
            generator = existing[0]
        for torsion in element.findall('Proper'):
            generator.registerProperTorsion(torsion.attrib)
        for torsion in element.findall('Improper'):
            if 'ordering' in element.attrib:
                generator.registerImproperTorsion(torsion.attrib,
                                                  element.attrib['ordering'])
            else:
                generator.registerImproperTorsion(torsion.attrib)
        # jax it!
        for k in generator.params.keys():
            generator.params[k] = jnp.array(generator.params[k])
        generator.types = np.array(generator.types)

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        wildcard = self.ff._atomClasses['']
        proper_cache = {}
        for torsion in data.propers:
            type1, type2, type3, type4 = [
                data.atomType[data.atoms[torsion[i]]] for i in range(4)
            ]
            sig = (type1, type2, type3, type4)
            sig = frozenset((sig, sig[::-1]))
            match = proper_cache.get(sig, None)
            if match == -1:
                continue
            if match is None:
                for index in self.propersForAtomType[type2]:
                    tordef = self.proper[index]
                    types1 = tordef.types1
                    types2 = tordef.types2
                    types3 = tordef.types3
                    types4 = tordef.types4
                    if (type2 in types2 and type3 in types3 and type4 in types4
                            and type1 in types1) or (type2 in types3
                                                     and type3 in types2
                                                     and type4 in types1
                                                     and type1 in types4):
                        hasWildcard = (wildcard
                                       in (types1, types2, types3, types4))
                        if match is None or not hasWildcard:  # Prefer specific definitions over ones with wildcards
                            match = tordef
                        if not hasWildcard:
                            break
                if match is None:
                    proper_cache[sig] = -1
                else:
                    proper_cache[sig] = match
            if match is not None:
                for i in range(len(match.phase)):
                    if match.k[i] != 0:
                        force.addTorsion(torsion[0], torsion[1], torsion[2],
                                         torsion[3], match.periodicity[i],
                                         match.phase[i], match.k[i])
        impr_cache = {}
        for torsion in data.impropers:
            t1, t2, t3, t4 = [
                data.atomType[data.atoms[torsion[i]]] for i in range(4)
            ]
            sig = (t1, t2, t3, t4)
            match = impr_cache.get(sig, None)
            if match == -1:
                # Previously checked, and doesn't appear in the database
                continue
            elif match:
                i1, i2, i3, i4, tordef = match
                a1, a2, a3, a4 = (torsion[i] for i in (i1, i2, i3, i4))
                match = (a1, a2, a3, a4, tordef)
            if match is None:
                match = _matchImproper(data, torsion, self)
                if match is not None:
                    order = match[:4]
                    i1, i2, i3, i4 = tuple(torsion.index(a) for a in order)
                    impr_cache[sig] = (i1, i2, i3, i4, match[-1])
                else:
                    impr_cache[sig] = -1
            if match is not None:
                (a1, a2, a3, a4, tordef) = match
                for i in range(len(tordef.phase)):
                    if tordef.k[i] != 0:
                        if tordef.ordering == 'smirnoff':
                            # Add all torsions in trefoil
                            force.addTorsion(a1, a2, a3, a4,
                                             tordef.periodicity[i],
                                             tordef.phase[i], tordef.k[i])
                            force.addTorsion(a1, a3, a4, a2,
                                             tordef.periodicity[i],
                                             tordef.phase[i], tordef.k[i])
                            force.addTorsion(a1, a4, a2, a3,
                                             tordef.periodicity[i],
                                             tordef.phase[i], tordef.k[i])
                        else:
                            force.addTorsion(a1, a2, a3, a4,
                                             tordef.periodicity[i],
                                             tordef.phase[i], tordef.k[i])


app.forcefield.parsers[
    "PeriodicTorsionForce"] = PeriodicTorsionGenerator.parseElement


class Hamiltonian(app.forcefield.ForceField):
    def __init__(self, xmlname):
        super().__init__(xmlname)
        self._potentials = []

    def createPotential(
        self,
        topology,
        nonbondedMethod=app.NoCutoff,
        nonbondedCutoff=1.0 * unit.nanometer,
        **args
    ):
        system = self.createSystem(topology,
                                   nonbondedMethod=nonbondedMethod,
                                   nonbondedCutoff=nonbondedCutoff,
                                   **args)
        # load_constraints_from_system_if_needed
        # create potentials
        for generator in self._forces:
            potentialImpl = generator.getJaxPotential()
            self._potentials.append(potentialImpl)
        return [p for p in self._potentials]


if __name__ == "__main__":
    H = Hamiltonian("forcefield.xml")
    generator = H.getGenerators()[0]
    app.Topology.loadBondDefinitions("residues.xml")
    pdb = app.PDBFile("../water1024.pdb")
    rc = 4.0
    potentials = H.createPotential(pdb.topology,
                                   nonbondedCutoff=rc * unit.angstrom)
    pot_disp = potentials[0]

    positions = jnp.array(pdb.positions._value) * 10
    a, b, c = pdb.topology.getPeriodicBoxVectors()
    box = jnp.array([a._value, b._value, c._value]) * 10

    # neighbor list
    displacement_fn, shift_fn = space.periodic_general(
        box, fractional_coordinates=False)
    neighbor_list_fn = partition.neighbor_list(displacement_fn,
                                               box,
                                               rc,
                                               0,
                                               format=partition.OrderedSparse)
    nbr = neighbor_list_fn.allocate(positions)
    pairs = nbr.idx.T

    param_grad = grad(pot_disp, argnums=3)(positions, box, pairs,
                                           generator.params)
    print(param_grad)
