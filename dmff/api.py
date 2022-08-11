import sys
import linecache
import itertools
from collections import defaultdict
from typing import Dict
import xml.etree.ElementTree as ET
from copy import deepcopy
import warnings

import numpy as np
import jax.numpy as jnp

import openmm as mm
import openmm.app as app
import openmm.app.element as elem
import openmm.unit as unit

from dmff.admp.disp_pme import ADMPDispPmeForce
from dmff.admp.multipole import convert_cart2harm, convert_harm2cart
from dmff.admp.pairwise import (TT_damping_qq_c6_kernel,
                                generate_pairwise_interaction,
                                slater_disp_damping_kernel, slater_sr_kernel,
                                TT_damping_qq_kernel)
from dmff.admp.pme import ADMPPmeForce, setup_ewald_parameters
from dmff.classical.intra import (
    HarmonicBondJaxForce,
    HarmonicAngleJaxForce,
    PeriodicTorsionJaxForce,
)
from dmff.classical.inter import (
    LennardJonesForce,
    LennardJonesLongRangeForce,
    CoulombPMEForce,
    CoulNoCutoffForce,
    CoulombPMEForce,
    CoulReactionFieldForce,
    LennardJonesForce,
)
from .classical.intra import (
    HarmonicAngleJaxForce,
    HarmonicBondJaxForce,
    PeriodicTorsionJaxForce,
)
from dmff.classical.fep import (LennardJonesFreeEnergyForce,
                                LennardJonesLongRangeFreeEnergyForce,
                                CoulombPMEFreeEnergyForce)
from dmff.utils import jit_condition, isinstance_jnp, DMFFException, findItemInList
from dmff.fftree import ForcefieldTree, XMLParser, TypeMatcher
from collections import defaultdict

jaxGenerators = {}


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
    return jnp.array(covalent_map)


def findAtomTypeTexts(attribs, num):
    typetxt = []
    for n in range(1, num + 1):
        for key in ["type%i" % n, "class%i" % n]:
            if key in attribs:
                typetxt.append((key, attribs[key]))
                break
    return typetxt


class ADMPDispGenerator:
    def __init__(self, ff):

        self.name = "ADMPDispForce"
        self.ff = ff
        self.fftree = ff.fftree
        self.paramtree = ff.paramtree

        # default params
        self._jaxPotential = None
        self.types = []
        self.ethresh = 5e-4
        self.pmax = 10

    def extract(self):

        mScales = [self.fftree.get_attribs(f'{self.name}', f'mScale1{i}')[0] for i in range(2, 7)]
        mScales.append(1.0)
        self.paramtree[self.name] = {}
        self.paramtree[self.name]['mScales'] = jnp.array(mScales)

        ABQC = self.fftree.get_attribs(f'{self.name}/Atom', ['A', 'B', 'Q', 'C6', 'C8', 'C10'])

        ABQC = np.array(ABQC)
        A = ABQC[:, 0]
        B = ABQC[:, 1]
        Q = ABQC[:, 2]
        C6 = ABQC[:, 3]
        C8 = ABQC[:, 4]
        C10 = ABQC[:, 5]

        self.paramtree[self.name]['A'] = jnp.array(A)
        self.paramtree[self.name]['B'] = jnp.array(B)
        self.paramtree[self.name]['Q'] = jnp.array(Q)
        self.paramtree[self.name]['C6'] = jnp.array(C6)
        self.paramtree[self.name]['C8'] = jnp.array(C8)
        self.paramtree[self.name]['C10'] = jnp.array(C10)

        atomTypes = self.fftree.get_attribs(f'{self.name}/Atom', f'type')
        if type(atomTypes[0]) != str:
            self.atomTypes = np.array(atomTypes, dtype=int).astype(str)
        else:
            self.atomTypes = np.array(atomTypes)


    def createForce(self, system, data, nonbondedMethod, nonbondedCutoff,
                    args):

        methodMap = {
            app.CutoffPeriodic: "CutoffPeriodic",
            app.NoCutoff: "NoCutoff",
            app.PME: "PME",
        }
        if nonbondedMethod not in methodMap:
            raise ValueError("Illegal nonbonded method for ADMPDispForce")
        if nonbondedMethod is app.CutoffPeriodic:
            self.lpme = False
        else:
            self.lpme = True

        n_atoms = len(data.atoms)
        # build index map
        map_atomtype = np.zeros(n_atoms, dtype=int)
        for i in range(n_atoms):
            atype = data.atomType[data.atoms[i]]
            map_atomtype[i] = np.where(self.atomTypes == atype)[0][0]
        self.map_atomtype = map_atomtype
        # build covalent map
        covalent_map = build_covalent_map(data, 6)
        # here box is only used to setup ewald parameters, no need to be differentiable
        a, b, c = system.getDefaultPeriodicBoxVectors()
        box = jnp.array([a._value, b._value, c._value]) * 10
        # get the admp calculator
        rc = nonbondedCutoff.value_in_unit(unit.angstrom)

        # get calculator
        if "ethresh" in args:
            self.ethresh = args["ethresh"]

        Force_DispPME = ADMPDispPmeForce(box,
                                         covalent_map,
                                         rc,
                                         self.ethresh,
                                         self.pmax,
                                         lpme=self.lpme)
        self.disp_pme_force = Force_DispPME
        pot_fn_lr = Force_DispPME.get_energy
        pot_fn_sr = generate_pairwise_interaction(TT_damping_qq_c6_kernel,
                                                  covalent_map,
                                                  static_args={})

        def potential_fn(positions, box, pairs, params):
            params = params[self.name]
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

    def overwrite(self):

        self.fftree.set_attrib(f'{self.name}', 'mScale12', [self.paramtree[self.name]['mScales'][0]])
        self.fftree.set_attrib(f'{self.name}', 'mScale13', [self.paramtree[self.name]['mScales'][1]])
        self.fftree.set_attrib(f'{self.name}', 'mScale14', [self.paramtree[self.name]['mScales'][2]])
        self.fftree.set_attrib(f'{self.name}', 'mScale15', [self.paramtree[self.name]['mScales'][3]])
        self.fftree.set_attrib(f'{self.name}', 'mScale16', [self.paramtree[self.name]['mScales'][4]])

        self.fftree.set_attrib(f'{self.name}/Atom', 'A', [self.paramtree[self.name]['A']])
        self.fftree.set_attrib(f'{self.name}/Atom', 'B', [self.paramtree[self.name]['B']])
        self.fftree.set_attrib(f'{self.name}/Atom', 'Q', [self.paramtree[self.name]['Q']])
        self.fftree.set_attrib(f'{self.name}/Atom', 'C6', [self.paramtree[self.name]['C6']])
        self.fftree.set_attrib(f'{self.name}/Atom', 'C8', [self.paramtree[self.name]['C8']])
        self.fftree.set_attrib(f'{self.name}/Atom', 'C10', [self.paramtree[self.name]['C10']])


    def getJaxPotential(self):
        return self._jaxPotential

jaxGenerators['ADMPDispForce'] = ADMPDispGenerator


class ADMPDispPmeGenerator:
    r"""
    This one computes the undamped C6/C8/C10 interactions
    u = \sum_{ij} c6/r^6 + c8/r^8 + c10/r^10
    """
    def __init__(self, ff):
        self.ff = ff
        self.fftree = ff.fftree
        self.paramtree = ff.paramtree

        self.params = {"C6": [], "C8": [], "C10": []}
        self._jaxPotential = None
        self.atomTypes = None
        self.ethresh = 5e-4
        self.pmax = 10
        self.name = "ADMPDispPmeForce"

    def extract(self):

        mScales = [self.fftree.get_attribs(f'{self.name}', f'mScale1{i}')[0] for i in range(2, 7)]
        mScales.append(1.0)

        self.paramtree[self.name] = {}
        self.paramtree[self.name]['mScales'] = jnp.array(mScales)

        C6 = self.fftree.get_attribs(f'{self.name}/Atom', f'C6')
        C8 = self.fftree.get_attribs(f'{self.name}/Atom', f'C8')
        C10 = self.fftree.get_attribs(f'{self.name}/Atom', f'C10')

        self.paramtree[self.name]['C6'] = jnp.array(C6)
        self.paramtree[self.name]['C8'] = jnp.array(C8)
        self.paramtree[self.name]['C10'] = jnp.array(C10)

        atomTypes = self.fftree.get_attribs(f'{self.name}/Atom', f'type')
        if type(atomTypes[0]) != str:
            self.atomTypes = np.array(atomTypes, dtype=int).astype(str)
        else:
            self.atomTypes = np.array(atomTypes)

    def overwrite(self):

        self.fftree.set_attrib(f'{self.name}', 'mScale12', [self.paramtree[self.name]['mScales'][0]])
        self.fftree.set_attrib(f'{self.name}', 'mScale13', [self.paramtree[self.name]['mScales'][1]])
        self.fftree.set_attrib(f'{self.name}', 'mScale14', [self.paramtree[self.name]['mScales'][2]])
        self.fftree.set_attrib(f'{self.name}', 'mScale15', [self.paramtree[self.name]['mScales'][3]])
        self.fftree.set_attrib(f'{self.name}', 'mScale16', [self.paramtree[self.name]['mScales'][4]])

        self.fftree.set_attrib(f'{self.name}/Atom', 'C6', self.paramtree[self.name]['C6'])
        self.fftree.set_attrib(f'{self.name}/Atom', 'C8', self.paramtree[self.name]['C8'])
        self.fftree.set_attrib(f'{self.name}/Atom', 'C10', self.paramtree[self.name]['C10'])


    def createForce(self, system, data, nonbondedMethod, nonbondedCutoff,
                    args):
        methodMap = {
            app.CutoffPeriodic: "CutoffPeriodic",
            app.NoCutoff: "NoCutoff",
            app.PME: "PME",
        }
        if nonbondedMethod not in methodMap:
            raise ValueError("Illegal nonbonded method for ADMPDispPmeForce")
        if nonbondedMethod is app.CutoffPeriodic:
            self.lpme = False
        else:
            self.lpme = True

        n_atoms = len(data.atoms)
        # build index map
        map_atomtype = np.zeros(n_atoms, dtype=int)
        for i in range(n_atoms):
            atype = data.atomType[data.atoms[i]]
            map_atomtype[i] = np.where(self.atomTypes == atype)[0][0]
        self.map_atomtype = map_atomtype

        # build covalent map
        covalent_map = build_covalent_map(data, 6)

        # here box is only used to setup ewald parameters, no need to be differentiable
        a, b, c = system.getDefaultPeriodicBoxVectors()
        box = jnp.array([a._value, b._value, c._value]) * 10
        # get the admp calculator
        rc = nonbondedCutoff.value_in_unit(unit.angstrom)

        # get calculator
        if "ethresh" in args:
            self.ethresh = args["ethresh"]

        disp_force = ADMPDispPmeForce(box, covalent_map, rc, self.ethresh,
                                      self.pmax, self.lpme)
        self.disp_force = disp_force
        pot_fn_lr = disp_force.get_energy

        def potential_fn(positions, box, pairs, params):
            params = params[self.name]
            mScales = params["mScales"]
            C6_list = params["C6"][map_atomtype] * 1e6  # to kj/mol * A**6
            C8_list = params["C8"][map_atomtype] * 1e8
            C10_list = params["C10"][map_atomtype] * 1e10
            c6_list = jnp.sqrt(C6_list)
            c8_list = jnp.sqrt(C8_list)
            c10_list = jnp.sqrt(C10_list)
            c_list = jnp.vstack((c6_list, c8_list, c10_list))
            E_lr = pot_fn_lr(positions, box, pairs, c_list.T, mScales)
            return -E_lr

        self._jaxPotential = potential_fn
        # self._top_data = data

    def getJaxPotential(self):
        return self._jaxPotential

jaxGenerators['ADMPDispPmeForce'] = ADMPDispPmeGenerator


class QqTtDampingGenerator:
    r"""
    This one calculates the tang-tonnies damping of charge-charge interaction
    E = \sum_ij exp(-B*r)*(1+B*r)*q_i*q_j/r
    """
    def __init__(self, ff):
        self.ff = ff
        self.fftree = ff.fftree
        self._jaxPotential = None
        self.paramtree = ff.paramtree
        self._jaxPotnetial = None
        self.name = "QqTtDampingForce"

    def extract(self):
        # get mscales
        mScales = [self.fftree.get_attribs(f'{self.name}', f'mScale1{i}')[0] for i in range(2, 7)]
        mScales.append(1.0)
        self.paramtree[self.name] = {}
        self.paramtree[self.name]['mScales'] = jnp.array(mScales)
        # get atomtypes
        atomTypes = self.fftree.get_attribs(f'{self.name}/Atom', f'type')
        if type(atomTypes[0]) != str:
            self.atomTypes = np.array(atomTypes, dtype=int).astype(str)
        else:
            self.atomTypes = np.array(atomTypes)
        # get atomic parameters
        B = self.fftree.get_attribs(f'{self.name}/Atom', f'B')
        Q = self.fftree.get_attribs(f'{self.name}/Atom', f'Q')
        self.paramtree[self.name]['B'] = jnp.array(B)
        self.paramtree[self.name]['Q'] = jnp.array(Q)


    def overwrite(self):

        self.fftree.set_attrib(f'{self.name}', 'mScale12', [self.paramtree[self.name]['mScales'][0]])
        self.fftree.set_attrib(f'{self.name}', 'mScale13', [self.paramtree[self.name]['mScales'][1]])
        self.fftree.set_attrib(f'{self.name}', 'mScale14', [self.paramtree[self.name]['mScales'][2]])
        self.fftree.set_attrib(f'{self.name}', 'mScale15', [self.paramtree[self.name]['mScales'][3]])
        self.fftree.set_attrib(f'{self.name}', 'mScale16', [self.paramtree[self.name]['mScales'][4]])

        self.fftree.set_attrib(f'{self.name}/Atom', 'B', self.paramtree[self.name]['B'])
        self.fftree.set_attrib(f'{self.name}/Atom', 'Q', self.paramtree[self.name]['Q'])

    # on working
    def createForce(self, system, data, nonbondedMethod, nonbondedCutoff,
                    args):

        n_atoms = len(data.atoms)
        # build index map
        map_atomtype = np.zeros(n_atoms, dtype=int)
        for i in range(n_atoms):
            atype = data.atomType[data.atoms[i]]
            map_atomtype[i] = np.where(self.atomTypes == atype)[0][0]
        self.map_atomtype = map_atomtype

        # build covalent map
        covalent_map = build_covalent_map(data, 6)

        pot_fn_sr = generate_pairwise_interaction(TT_damping_qq_kernel,
                                                  covalent_map,
                                                  static_args={})

        def potential_fn(positions, box, pairs, params):
            params = params[self.name]
            mScales = params["mScales"]
            b_list = params["B"][map_atomtype] / 10  # convert to A^-1
            q_list = params["Q"][map_atomtype]

            E_sr = pot_fn_sr(positions, box, pairs, mScales, b_list, q_list)
            return E_sr

        self._jaxPotential = potential_fn

    def getJaxPotential(self):
        return self._jaxPotential

# register all parsers
jaxGenerators['QqTtDampingForce'] = QqTtDampingGenerator


class SlaterDampingGenerator:
    r"""
    This one computes the slater-type damping function for c6/c8/c10 dispersion
    E = \sum_ij (f6-1)*c6/r6 + (f8-1)*c8/r8 + (f10-1)*c10/r10
    fn = f_tt(x, n)
    x = br - (2*br2 + 3*br) / (br2 + 3*br + 3)
    """
    def __init__(self, ff):
        self.name = "SlaterDampingForce"
        self.ff = ff
        self.fftree = ff.fftree
        self.paramtree = ff.paramtree
        self._jaxPotential = None

    def extract(self):
        # get mscales
        mScales = [self.fftree.get_attribs(f'{self.name}', f'mScale1{i}')[0] for i in range(2, 7)]
        mScales.append(1.0)
        self.paramtree[self.name] = {}
        self.paramtree[self.name]['mScales'] = jnp.array(mScales)
        # get atomtypes
        atomTypes = self.fftree.get_attribs(f'{self.name}/Atom', f'type')
        if type(atomTypes[0]) != str:
            self.atomTypes = np.array(atomTypes, dtype=int).astype(str)
        else:
            self.atomTypes = np.array(atomTypes)
        # get atomic parameters
        B = self.fftree.get_attribs(f'{self.name}/Atom', f'B')
        C6 = self.fftree.get_attribs(f'{self.name}/Atom', f'C6')
        C8 = self.fftree.get_attribs(f'{self.name}/Atom', f'C8')
        C10 = self.fftree.get_attribs(f'{self.name}/Atom', f'C10')
        self.paramtree[self.name]['B'] = jnp.array(B)
        self.paramtree[self.name]['C6'] = jnp.array(C6)
        self.paramtree[self.name]['C8'] = jnp.array(C8)
        self.paramtree[self.name]['C10'] = jnp.array(C10)


    def overwrite(self):

        self.fftree.set_attrib(f'{self.name}', 'mScale12', [self.paramtree[self.name]['mScales'][0]])
        self.fftree.set_attrib(f'{self.name}', 'mScale13', [self.paramtree[self.name]['mScales'][1]])
        self.fftree.set_attrib(f'{self.name}', 'mScale14', [self.paramtree[self.name]['mScales'][2]])
        self.fftree.set_attrib(f'{self.name}', 'mScale15', [self.paramtree[self.name]['mScales'][3]])
        self.fftree.set_attrib(f'{self.name}', 'mScale16', [self.paramtree[self.name]['mScales'][4]])

        self.fftree.set_attrib(f'{self.name}/Atom', 'B', [self.paramtree[self.name]['B']])
        self.fftree.set_attrib(f'{self.name}/Atom', 'C6', [self.paramtree[self.name]['C6']])
        self.fftree.set_attrib(f'{self.name}/Atom', 'C8', [self.paramtree[self.name]['C8']])
        self.fftree.set_attrib(f'{self.name}/Atom', 'C10', [self.paramtree[self.name]['C10']])


    def createForce(self, system, data, nonbondedMethod, nonbondedCutoff,
                    args):

        n_atoms = len(data.atoms)
        # build index map
        map_atomtype = np.zeros(n_atoms, dtype=int)
        for i in range(n_atoms):
            atype = data.atomType[data.atoms[i]]
            map_atomtype[i] = np.where(self.atomTypes == atype)[0][0]
        self.map_atomtype = map_atomtype
        # build covalent map
        covalent_map = build_covalent_map(data, 6)

        # WORKING
        pot_fn_sr = generate_pairwise_interaction(slater_disp_damping_kernel,
                                                  covalent_map,
                                                  static_args={})

        def potential_fn(positions, box, pairs, params):
            params = params[self.name]
            mScales = params["mScales"]
            b_list = params["B"][map_atomtype] / 10  # convert to A^-1
            c6_list = jnp.sqrt(params["C6"][map_atomtype] *
                               1e6)  # to kj/mol * A**6
            c8_list = jnp.sqrt(params["C8"][map_atomtype] * 1e8)
            c10_list = jnp.sqrt(params["C10"][map_atomtype] * 1e10)
            E_sr = pot_fn_sr(positions, box, pairs, mScales, b_list, c6_list,
                             c8_list, c10_list)
            return E_sr

        self._jaxPotential = potential_fn
        # self._top_data = data

    def getJaxPotential(self):
        return self._jaxPotential

jaxGenerators['SlaterDampingForce'] = SlaterDampingGenerator



class SlaterExGenerator:
    r"""
    This one computes the Slater-ISA type exchange interaction
    u = \sum_ij A * (1/3*(Br)^2 + Br + 1)
    """
    def __init__(self, ff):
        self.name = "SlaterExForce"
        self.ff = ff
        self.fftree = ff.fftree
        self.paramtree = ff.paramtree
        self._jaxPotential = None


    def extract(self):
        # get mscales
        mScales = [self.fftree.get_attribs(f'{self.name}', f'mScale1{i}')[0] for i in range(2, 7)]
        mScales.append(1.0)
        self.paramtree[self.name] = {}
        self.paramtree[self.name]['mScales'] = jnp.array(mScales)
        # get atomtypes
        atomTypes = self.fftree.get_attribs(f'{self.name}/Atom', f'type')
        if type(atomTypes[0]) != str:
            self.atomTypes = np.array(atomTypes, dtype=int).astype(str)
        else:
            self.atomTypes = np.array(atomTypes)
        # get atomic parameters
        A = self.fftree.get_attribs(f'{self.name}/Atom', f'A')
        B = self.fftree.get_attribs(f'{self.name}/Atom', f'B')
        self.paramtree[self.name]['A'] = jnp.array(A)
        self.paramtree[self.name]['B'] = jnp.array(B)


    def overwrite(self):

        self.fftree.set_attrib(f'{self.name}', 'mScale12', [self.paramtree[self.name]['mScales'][0]])
        self.fftree.set_attrib(f'{self.name}', 'mScale13', [self.paramtree[self.name]['mScales'][1]])
        self.fftree.set_attrib(f'{self.name}', 'mScale14', [self.paramtree[self.name]['mScales'][2]])
        self.fftree.set_attrib(f'{self.name}', 'mScale15', [self.paramtree[self.name]['mScales'][3]])
        self.fftree.set_attrib(f'{self.name}', 'mScale16', [self.paramtree[self.name]['mScales'][4]])

        self.fftree.set_attrib(f'{self.name}/Atom', 'A', [self.paramtree[self.name]['A']])
        self.fftree.set_attrib(f'{self.name}/Atom', 'B', [self.paramtree[self.name]['B']])


    def createForce(self, system, data, nonbondedMethod, nonbondedCutoff,
                    args):

        n_atoms = len(data.atoms)
        # build index map
        map_atomtype = np.zeros(n_atoms, dtype=int)
        for i in range(n_atoms):
            atype = data.atomType[data.atoms[i]]
            map_atomtype[i] = np.where(self.atomTypes == atype)[0][0]
        self.map_atomtype = map_atomtype
        # build covalent map
        covalent_map = build_covalent_map(data, 6)

        pot_fn_sr = generate_pairwise_interaction(slater_sr_kernel,
                                                  covalent_map,
                                                  static_args={})

        def potential_fn(positions, box, pairs, params):
            params = params[self.name]
            mScales = params["mScales"]
            a_list = params["A"][map_atomtype]
            b_list = params["B"][map_atomtype] / 10  # nm^-1 to A^-1

            return pot_fn_sr(positions, box, pairs, mScales, a_list, b_list)

        self._jaxPotential = potential_fn
        # self._top_data = data

    def getJaxPotential(self):
        return self._jaxPotential


jaxGenerators["SlaterExForce"] = SlaterExGenerator


# Here are all the short range "charge penetration" terms
# They all have the exchange form
class SlaterSrEsGenerator(SlaterExGenerator):
    def __init__(self, ff):
        super().__init__(ff)
        self.name = "SlaterSrEsForce"


class SlaterSrPolGenerator(SlaterExGenerator):
    def __init__(self, ff):
        super().__init__(ff)
        self.name = "SlaterSrPolForce"


class SlaterSrDispGenerator(SlaterExGenerator):
    def __init__(self, ff):
        super().__init__(ff)
        self.name = "SlaterSrDispForce"


class SlaterDhfGenerator(SlaterExGenerator):
    def __init__(self, ff):
        super().__init__(ff)
        self.name = "SlaterDhfForce"


# register all parsers
jaxGenerators["SlaterSrEsForce"] = SlaterSrEsGenerator
jaxGenerators["SlaterSrPolForce"] = SlaterSrPolGenerator
jaxGenerators["SlaterSrDispForce"] = SlaterSrDispGenerator
jaxGenerators["SlaterDhfForce"] = SlaterDhfGenerator


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
        self.step_pol = None
        self.lpol = False
        self.ref_dip = ""
        self.name = "ADMPPme"

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
        r"""parse admp related parameters in XML file

        example:

        <ADMPDispForce mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="1.00" mScale16="1.00">
          <Atom type="380" A="1203470.743" B="37.81265679" Q="-0.741706" C6="0.001383816" C8="7.27065e-05" C10="1.8076465e-6"/>
          <Atom type="381" A="83.2283563" B="37.78544799"  Q="0.370853" C6="5.7929e-05" C8="1.416624e-06" C10="2.26525e-08"/>
        </ADMPDispForce>

        <ADMPPmeForce lmax="2" mScale12="0.00" mScale13="0.00" mScale14="0.00" mScale15="1.00" mScale16="1.00" pScale12="0.00" pScale13="0.00" pScale14="0.00" pScale15="1.00" pScale16="1.00" dScale12="0.00" dScale13="0.00" dScale14="0.00" dScale15="1.00" dScale16="1.00">

          <Atom type="380" kz="-381" kx="-381"
                        c0="-1.0614"
                        dX="0.0" dY="0.0"  dZ="-0.023671684"
                        qXX="0.000150963" qXY="0.0" qYY="0.00008707" qXZ="0.0" qYZ="0.0" qZZ="-0.000238034"
                        oXXX="0.0" oXXY="0.0" oXYY="0.0" oYYY="0.0" oXXZ="0.0000" oXYZ="0.0" oYYZ="0.00000" oXZZ="0.0" oYZZ="0.0" oZZZ="-0.0000"
                        />
          <Atom type="381" kz="380" kx="381"
                        c0="0.5307"
                        dX="0.0" dY="0.0"  dZ="0.0"
                        qXX="0.0" qXY="0.0" qYY="0.0" qXZ="0.0" qYZ="0.0" qZZ="0.0"
                        oXXX="0.0" oXXY="0.0" oXYY="0.0" oYYY="0.0" oXXZ="0.0" oXYZ="0.0" oYYZ="0.0" oXZZ="0.0" oYZZ="0.0" oZZZ="0.0"
                        />
          <Polarize type="380" polarizabilityXX="0.00088" polarizabilityYY="0.00088" polarizabilityZZ="0.00088" thole="8.0"/>
          <Polarize type="381" polarizabilityXX="0.000" polarizabilityYY="0.000" polarizabilityZZ="0.000" thole="0.0"/>
        </ADMPPmeForce>

        """

        generator = ADMPPmeGenerator(hamiltonian)
        generator.lmax = int(element.attrib.get("lmax"))
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
        generator.params["mScales"].append(1.0)
        generator.params["pScales"].append(1.0)
        generator.params["dScales"].append(1.0)

        if element.findall("Polarize"):
            generator.lpol = True
        else:
            generator.lpol = False

        for atomType in element.findall("Atom"):
            atomAttrib = atomType.attrib
            # if not set
            atomAttrib.update({
                "polarizabilityXX": 0,
                "polarizabilityYY": 0,
                "polarizabilityZZ": 0
            })
            for polarInfo in element.findall("Polarize"):
                polarAttrib = polarInfo.attrib
                if polarInfo.attrib["type"] == atomAttrib["type"]:
                    # cover default
                    atomAttrib.update(polarAttrib)
                    break
            generator.registerAtomType(atomAttrib)

        for k in generator._input_params.keys():
            generator._input_params[k] = jnp.array(generator._input_params[k])
        generator.types = np.array(generator.types)

        n_atoms = len(element.findall("Atom"))
        generator.n_atoms = n_atoms

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
        generator.params["Q_local"] = Q_local

        if generator.lpol:
            pol = jnp.vstack((
                generator._input_params["polarizabilityXX"],
                generator._input_params["polarizabilityYY"],
                generator._input_params["polarizabilityZZ"],
            )).T
            pol = 1000 * jnp.mean(pol, axis=1)
            tholes = jnp.array(generator._input_params["thole"])
            generator.params["pol"] = pol
            generator.params["tholes"] = tholes
        else:
            pol = None
            tholes = None

        # generator.params['']
        for k in generator.params.keys():
            generator.params[k] = jnp.array(generator.params[k])

    def createForce(self, system, data, nonbondedMethod, nonbondedCutoff,
                    args):

        methodMap = {
            app.CutoffPeriodic: "CutoffPeriodic",
            app.NoCutoff: "NoCutoff",
            app.PME: "PME",
        }
        if nonbondedMethod not in methodMap:
            raise ValueError("Illegal nonbonded method for ADMPPmeForce")
        if nonbondedMethod is app.CutoffPeriodic:
            self.lpme = False
        else:
            self.lpme = True

        n_atoms = len(data.atoms)
        map_atomtype = np.zeros(n_atoms, dtype=int)

        for i in range(n_atoms):
            atype = data.atomType[data.atoms[i]]
            map_atomtype[i] = np.where(self.types == atype)[0][0]
        self.map_atomtype = map_atomtype

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
            ZThenX = 0
            Bisector = 1
            ZBisect = 2
            ThreeFold = 3
            ZOnly = 4  # typo fix
            NoAxisType = 5
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
                        kz = int(self.kStrings["kz"][itype])
                        kx = int(self.kStrings["kx"][itype])
                        ky = int(self.kStrings["ky"][itype])
                        neighbors = np.where(covalent_map[i_atom] == 1)[0]
                        zaxis = -1
                        xaxis = -1
                        yaxis = -1
                        for z_index in neighbors:
                            if hit != 0:
                                break
                            z_type = int(data.atomType[data.atoms[z_index]])
                            if z_type == abs(
                                    kz
                            ):  # find the z atom, start searching for x
                                for x_index in neighbors:
                                    if x_index == z_index or hit != 0:
                                        continue
                                    x_type = int(
                                        data.atomType[data.atoms[x_index]])
                                    if x_type == abs(
                                            kx
                                    ):  # find the x atom, start searching for y
                                        if ky == 0:
                                            zaxis = z_index
                                            xaxis = x_index
                                            # cannot ditinguish x and z? use the smaller index for z, and the larger index for x
                                            if x_type == z_type and xaxis < zaxis:
                                                swap = z_axis
                                                z_axis = x_axis
                                                x_axis = swap
                                            # otherwise, try to see if we can find an even smaller index for x?
                                            else:
                                                for x_index in neighbors:
                                                    x_type1 = int(
                                                        data.atomType[
                                                            data.
                                                            atoms[x_index]])
                                                    if (x_type1 == abs(kx) and
                                                            x_index != z_index
                                                            and
                                                            x_index < xaxis):
                                                        xaxis = x_index
                                            hit = 1  # hit, finish matching
                                            matched_itype = itype
                                        else:
                                            for y_index in neighbors:
                                                if (y_index == z_index
                                                        or y_index == x_index
                                                        or hit != 0):
                                                    continue
                                                y_type = int(data.atomType[
                                                    data.atoms[y_index]])
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
                        kz = int(self.kStrings["kz"][itype])
                        kx = int(self.kStrings["kx"][itype])
                        ky = int(self.kStrings["ky"][itype])
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
                                    x_type = int(
                                        data.atomType[data.atoms[x_index]])
                                    # we ask x to be in 2'nd neighbor, and x is z's neighbor
                                    if (x_type == abs(kx)
                                            and covalent_map[z_index,
                                                             x_index] == 1):
                                        if ky == 0:
                                            zaxis = z_index
                                            xaxis = x_index
                                            # select smallest x index
                                            for x_index in neighbors_2nd:
                                                x_type1 = int(data.atomType[
                                                    data.atoms[x_index]])
                                                if (x_type1 == abs(kx)
                                                        and x_index != z_index
                                                        and
                                                        covalent_map[x_index,
                                                                     z_index]
                                                        == 1
                                                        and x_index < xaxis):
                                                    xaxis = x_index
                                            hit = 3
                                            matched_itype = itype
                                        else:
                                            for y_index in neighbors_2nd:
                                                if (y_index == z_index
                                                        or y_index == x_index
                                                        or hit != 0):
                                                    continue
                                                y_type = int(data.atomType[
                                                    data.atoms[y_index]])
                                                if (y_type == abs(ky) and
                                                        covalent_map[y_index,
                                                                     z_index]
                                                        == 1):
                                                    zaxis = z_index
                                                    xaxis = x_index
                                                    yaxis = y_index
                                                    hit = 4
                                                    matched_itype = itype
                    # assign multipole parameters via only a z-defining atom
                    for itype in itypes:
                        if hit != 0:
                            break
                        kz = int(self.kStrings["kz"][itype])
                        kx = int(self.kStrings["kx"][itype])
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
                        kz = int(self.kStrings["kz"][itype])
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

                        kz = int(self.kStrings["kz"][matched_itype])
                        kx = int(self.kStrings["kx"][matched_itype])
                        ky = int(self.kStrings["ky"][matched_itype])
                        axisType = ZThenX
                        if kz == 0:
                            axisType = NoAxisType
                        if kz != 0 and kx == 0:
                            axisType = ZOnly
                        if kz < 0 or kx < 0:
                            axisType = Bisector
                        if kx < 0 and ky < 0:
                            axisType = ZBisect
                        if kz < 0 and kx < 0 and ky < 0:
                            axisType = ThreeFold
                        self.axis_types.append(axisType)

                    else:
                        sys.exit("Atom %d not matched in forcefield!" % i_atom)

                else:
                    sys.exit("Atom %d not matched in forcefield!" % i_atom)
            self.axis_indices = np.array(self.axis_indices)
            self.axis_types = np.array(self.axis_types)
        else:
            self.axis_types = None
            self.axis_indices = None

        if "ethresh" in args:
            self.ethresh = args["ethresh"]
        if "step_pol" in args:
            self.step_pol = args["step_pol"]

        pme_force = ADMPPmeForce(box, self.axis_types, self.axis_indices,
                                 covalent_map, rc, self.ethresh, self.lmax,
                                 self.lpol, self.lpme, self.step_pol)
        self.pme_force = pme_force

        def potential_fn(positions, box, pairs, params):

            mScales = params["mScales"]
            Q_local = params["Q_local"][map_atomtype]
            if self.lpol:
                pScales = params["pScales"]
                dScales = params["dScales"]
                pol = params["pol"][map_atomtype]
                tholes = params["tholes"][map_atomtype]

                return pme_force.get_energy(
                    positions,
                    box,
                    pairs,
                    Q_local,
                    pol,
                    tholes,
                    mScales,
                    pScales,
                    dScales,
                    pme_force.U_ind,
                )
            else:
                return pme_force.get_energy(positions, box, pairs, Q_local,
                                            mScales)

        self._jaxPotential = potential_fn

    def getJaxPotential(self):
        return self._jaxPotential

class ADMPPmeGenerator:

    def __init__(self, ff):

        self.name = 'ADMPPmeForce'
        self.ff = ff
        self.fftree = ff.fftree
        self.paramtree = ff.paramtree

        # default params
        self._jaxPotential = None
        self.types = []
        self.ethresh = 5e-4
        self.step_pol = None
        self.lpol = False
        self.ref_dip = ""

    def extract(self):

        self.lmax = self.fftree.get_attribs(f'{self.name}', 'lmax')[0]  # return [lmax]

        mScales = [self.fftree.get_attribs(f'{self.name}', f'mScale1{i}')[0] for i in range(2, 7)]
        pScales = [self.fftree.get_attribs(f'{self.name}', f'pScale1{i}')[0] for i in range(2, 7)]
        dScales = [self.fftree.get_attribs(f'{self.name}', f'dScale1{i}')[0] for i in range(2, 7)]

        # make sure the last digit is 1.0
        mScales.append(1.0)
        pScales.append(1.0)
        dScales.append(1.0)

        self.paramtree[self.name] = {}
        self.paramtree[self.name]['mScales'] = jnp.array(mScales)
        self.paramtree[self.name]['pScales'] = jnp.array(pScales)
        self.paramtree[self.name]['dScales'] = jnp.array(dScales)

        # check if polarize
        polarize = self.fftree.get_nodes(f'{self.name}/Polarize')
        if polarize:
            self.lpol = True
        else:
            self.lpol = False

        atomTypes = self.fftree.get_attribs(f'{self.name}/Atom', 'type')
        if type(atomTypes[0]) != str:
            self.atomTypes = np.array(atomTypes, dtype=int).astype(str)
        else:
            self.atomTypes = np.array(atomTypes)
        kx = self.fftree.get_attribs(f'{self.name}/Atom', 'kx')
        ky = self.fftree.get_attribs(f'{self.name}/Atom', 'ky')
        kz = self.fftree.get_attribs(f'{self.name}/Atom', 'kz')

        kx = [ 0 if kx_ is None else int(kx_) for kx_ in kx  ]
        ky = [ 0 if ky_ is None else int(ky_) for ky_ in ky  ]
        kz = [ 0 if kz_ is None else int(kz_) for kz_ in kz  ]

        # invoke by `self.kStrings["kz"][itype]`
        self.kStrings = {}
        self.kStrings['kx'] = kx
        self.kStrings['ky'] = ky
        self.kStrings['kz'] = kz

        c0 = self.fftree.get_attribs(f'{self.name}/Atom', 'c0')
        dX = self.fftree.get_attribs(f'{self.name}/Atom', 'dX')
        dY = self.fftree.get_attribs(f'{self.name}/Atom', 'dY')
        dZ = self.fftree.get_attribs(f'{self.name}/Atom', 'dZ')
        qXX = self.fftree.get_attribs(f'{self.name}/Atom', 'qXX')
        qYY = self.fftree.get_attribs(f'{self.name}/Atom', 'qYY')
        qZZ = self.fftree.get_attribs(f'{self.name}/Atom', 'qZZ')
        qXY = self.fftree.get_attribs(f'{self.name}/Atom', 'qXY')
        qYZ = self.fftree.get_attribs(f'{self.name}/Atom', 'qYZ')

        # assume that polarize tag match the per atom type
        # pol_XX = self.fftree.get_attribs(f'{self.name}/Polarize', 'polarizabilityXX')
        # pol_YY = self.fftree.get_attribs(f'{self.name}/Polarize', 'polarizabilityYY')
        # pol_ZZ = self.fftree.get_attribs(f'{self.name}/Polarize', 'polarizabilityZZ')
        # thole_0 = self.fftree.get_attribs(f'{self.name}/Polarize', 'thole')
        polarizabilityXX = self.fftree.get_attribs(f'{self.name}/Polarize', 'polarizabilityXX')
        polarizabilityYY = self.fftree.get_attribs(f'{self.name}/Polarize', 'polarizabilityYY')
        polarizabilityZZ = self.fftree.get_attribs(f'{self.name}/Polarize', 'polarizabilityZZ')
        thole = self.fftree.get_attribs(f'{self.name}/Polarize', 'thole')
        polarize_types = self.fftree.get_attribs(f'{self.name}/Polarize', 'type')
        if type(polarize_types[0]) != str:
            polarize_types = np.array(polarize_types, dtype=int).astype(str)
        else:
            polarize_types = np.array(polarize_types)
        self.polarize_types = polarize_types

        n_atoms = len(atomTypes)


        # assert n_atoms == len(polarizabilityXX), "Number of polarizabilityXX does not match number of atoms!"

        # map atom multipole moments
        if self.lmax == 0:
            n_mtps = 1
        elif self.lmax == 1:
            n_mtps = 4
        elif self.lmax == 2:
            n_mtps = 10
        Q = np.zeros((n_atoms, n_mtps))

        # TDDO: unit conversion
        Q[:, 0] = c0
        if self.lmax >= 1:
            Q[:, 1] = dX
            Q[:, 2] = dY
            Q[:, 3] = dZ
            Q[:, 1:4] *= 10
        if self.lmax >= 2:
            Q[:, 4] = qXX
            Q[:, 5] = qYY
            Q[:, 6] = qZZ
            Q[:, 7] = qXY
            Q[:, 8] = qYZ
            Q[:, 4:9] *= 300

        # add all differentiable params to self.params
        Q_local = convert_cart2harm(Q, self.lmax)
        self.paramtree[self.name]["Q_local"] = Q_local

        if self.lpol:
            pol = jnp.vstack((
                polarizabilityXX,
                polarizabilityYY,
                polarizabilityZZ,
            )).T
            pol = 1000 * jnp.mean(pol, axis=1)
            tholes = jnp.array(thole)
            self.paramtree[self.name]["pol"] = pol
            self.paramtree[self.name]["tholes"] = tholes
        else:
            pol = None
            tholes = None


    def overwrite(self):

        self.fftree.set_attrib(f'{self.name}', 'mScale12', [self.paramtree[self.name]['mScales'][0]])
        self.fftree.set_attrib(f'{self.name}', 'mScale13', [self.paramtree[self.name]['mScales'][1]])
        self.fftree.set_attrib(f'{self.name}', 'mScale14', [self.paramtree[self.name]['mScales'][2]])
        self.fftree.set_attrib(f'{self.name}', 'mScale15', [self.paramtree[self.name]['mScales'][3]])
        self.fftree.set_attrib(f'{self.name}', 'mScale16', [self.paramtree[self.name]['mScales'][4]])

        self.fftree.set_attrib(f'{self.name}', 'pScale12', [self.paramtree[self.name]['pScales'][0]])
        self.fftree.set_attrib(f'{self.name}', 'pScale13', [self.paramtree[self.name]['pScales'][1]])
        self.fftree.set_attrib(f'{self.name}', 'pScale14', [self.paramtree[self.name]['pScales'][2]])
        self.fftree.set_attrib(f'{self.name}', 'pScale15', [self.paramtree[self.name]['pScales'][3]])
        self.fftree.set_attrib(f'{self.name}', 'pScale16', [self.paramtree[self.name]['pScales'][4]])

        self.fftree.set_attrib(f'{self.name}', 'dScale12', [self.paramtree[self.name]['dScales'][0]])
        self.fftree.set_attrib(f'{self.name}', 'dScale13', [self.paramtree[self.name]['dScales'][1]])
        self.fftree.set_attrib(f'{self.name}', 'dScale14', [self.paramtree[self.name]['dScales'][2]])
        self.fftree.set_attrib(f'{self.name}', 'dScale15', [self.paramtree[self.name]['dScales'][3]])
        self.fftree.set_attrib(f'{self.name}', 'dScale16', [self.paramtree[self.name]['dScales'][4]])

        Q_global = convert_harm2cart(self.paramtree[self.name]['Q_local'], self.lmax)


        self.fftree.set_attrib(f'{self.name}/Atom', 'c0', Q_global[:, 0])
        self.fftree.set_attrib(f'{self.name}/Atom', 'dX', Q_global[:, 1])
        self.fftree.set_attrib(f'{self.name}/Atom', 'dY', Q_global[:, 2])
        self.fftree.set_attrib(f'{self.name}/Atom', 'dZ', Q_global[:, 3])
        self.fftree.set_attrib(f'{self.name}/Atom', 'qXX', Q_global[:, 4])
        self.fftree.set_attrib(f'{self.name}/Atom', 'qYY', Q_global[:, 5])
        self.fftree.set_attrib(f'{self.name}/Atom', 'qZZ', Q_global[:, 6])
        self.fftree.set_attrib(f'{self.name}/Atom', 'qXY', Q_global[:, 7])
        self.fftree.set_attrib(f'{self.name}/Atom', 'qYZ', Q_global[:, 8])
        self.fftree.set_attrib(f'{self.name}/Atom', 'qYZ', Q_global[:, 9])

        if self.lpol:
            # self.paramtree[self.name]['pol']: every element is the mean value of XX YY ZZ
            # get the number of polarize element
            n_pol = len(self.paramtree[self.name]['pol'])
            self.fftree.set_attrib(f'{self.name}/Polarize', 'polarizabilityXX', [self.paramtree[self.name]['pol'][0]] * n_pol)
            self.fftree.set_attrib(f'{self.name}/Polarize', 'polarizabilityYY', [self.paramtree[self.name]['pol'][1]] * n_pol)
            self.fftree.set_attrib(f'{self.name}/Polarize', 'polarizabilityZZ', [self.paramtree[self.name]['pol'][2]] * n_pol)
            self.fftree.set_attrib(f'{self.name}/Polarize', 'thole', self.paramtree[self.name]['tholes'])


    def createForce(self, system, data, nonbondedMethod, nonbondedCutoff,
                    args):

        methodMap = {
            app.CutoffPeriodic: "CutoffPeriodic",
            app.NoCutoff: "NoCutoff",
            app.PME: "PME",
        }
        if nonbondedMethod not in methodMap:
            raise ValueError("Illegal nonbonded method for ADMPPmeForce")
        if nonbondedMethod is app.CutoffPeriodic:
            self.lpme = False
        else:
            self.lpme = True

        n_atoms = len(data.atoms)
        map_atomtype = np.zeros(n_atoms, dtype=int)
        map_poltype = np.zeros(n_atoms, dtype=int)

        for i in range(n_atoms):
            atype = data.atomType[data.atoms[i]]  # convert str to int to match atomTypes
            map_atomtype[i] = np.where(self.atomTypes == atype)[0][0]
            map_poltype[i] = np.where(self.polarize_types == atype)[0][0]
        self.map_atomtype = map_atomtype
        self.map_poltype = map_poltype

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
            ZThenX = 0
            Bisector = 1
            ZBisect = 2
            ThreeFold = 3
            ZOnly = 4  # typo fix
            NoAxisType = 5
            LastAxisTypeIndex = 6

            self.axis_types = []
            self.axis_indices = []
            for i_atom in range(n_atoms):
                atom = data.atoms[i_atom]
                t = data.atomType[atom]
                # if t is in type list?
                if t in self.atomTypes:
                    itypes = np.where(self.atomTypes == t)[0]
                    hit = 0
                    # try to assign multipole parameters via only 1-2 connected atoms
                    for itype in itypes:
                        if hit != 0:
                            break
                        kz = int(self.kStrings["kz"][itype])
                        kx = int(self.kStrings["kx"][itype])
                        ky = int(self.kStrings["ky"][itype])
                        neighbors = np.where(covalent_map[i_atom] == 1)[0]
                        zaxis = -1
                        xaxis = -1
                        yaxis = -1
                        for z_index in neighbors:
                            if hit != 0:
                                break
                            z_type = int(data.atomType[data.atoms[z_index]])
                            if z_type == abs(
                                    kz
                            ):  # find the z atom, start searching for x
                                for x_index in neighbors:
                                    if x_index == z_index or hit != 0:
                                        continue
                                    x_type = int(
                                        data.atomType[data.atoms[x_index]])
                                    if x_type == abs(
                                            kx
                                    ):  # find the x atom, start searching for y
                                        if ky == 0:
                                            zaxis = z_index
                                            xaxis = x_index
                                            # cannot ditinguish x and z? use the smaller index for z, and the larger index for x
                                            if x_type == z_type and xaxis < zaxis:
                                                swap = z_axis
                                                z_axis = x_axis
                                                x_axis = swap
                                            # otherwise, try to see if we can find an even smaller index for x?
                                            else:
                                                for x_index in neighbors:
                                                    x_type1 = int(
                                                        data.atomType[
                                                            data.
                                                            atoms[x_index]])
                                                    if (x_type1 == abs(kx) and
                                                            x_index != z_index
                                                            and
                                                            x_index < xaxis):
                                                        xaxis = x_index
                                            hit = 1  # hit, finish matching
                                            matched_itype = itype
                                        else:
                                            for y_index in neighbors:
                                                if (y_index == z_index
                                                        or y_index == x_index
                                                        or hit != 0):
                                                    continue
                                                y_type = int(data.atomType[
                                                    data.atoms[y_index]])
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
                        kz = int(self.kStrings["kz"][itype])
                        kx = int(self.kStrings["kx"][itype])
                        ky = int(self.kStrings["ky"][itype])
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
                                    x_type = int(
                                        data.atomType[data.atoms[x_index]])
                                    # we ask x to be in 2'nd neighbor, and x is z's neighbor
                                    if (x_type == abs(kx)
                                            and covalent_map[z_index,
                                                             x_index] == 1):
                                        if ky == 0:
                                            zaxis = z_index
                                            xaxis = x_index
                                            # select smallest x index
                                            for x_index in neighbors_2nd:
                                                x_type1 = int(data.atomType[
                                                    data.atoms[x_index]])
                                                if (x_type1 == abs(kx)
                                                        and x_index != z_index
                                                        and
                                                        covalent_map[x_index,
                                                                     z_index]
                                                        == 1
                                                        and x_index < xaxis):
                                                    xaxis = x_index
                                            hit = 3
                                            matched_itype = itype
                                        else:
                                            for y_index in neighbors_2nd:
                                                if (y_index == z_index
                                                        or y_index == x_index
                                                        or hit != 0):
                                                    continue
                                                y_type = int(data.atomType[
                                                    data.atoms[y_index]])
                                                if (y_type == abs(ky) and
                                                        covalent_map[y_index,
                                                                     z_index]
                                                        == 1):
                                                    zaxis = z_index
                                                    xaxis = x_index
                                                    yaxis = y_index
                                                    hit = 4
                                                    matched_itype = itype
                    # assign multipole parameters via only a z-defining atom
                    for itype in itypes:
                        if hit != 0:
                            break
                        kz = int(self.kStrings["kz"][itype])
                        kx = int(self.kStrings["kx"][itype])
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
                        kz = int(self.kStrings["kz"][itype])
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

                        kz = int(self.kStrings["kz"][matched_itype])
                        kx = int(self.kStrings["kx"][matched_itype])
                        ky = int(self.kStrings["ky"][matched_itype])
                        axisType = ZThenX
                        if kz == 0:
                            axisType = NoAxisType
                        if kz != 0 and kx == 0:
                            axisType = ZOnly
                        if kz < 0 or kx < 0:
                            axisType = Bisector
                        if kx < 0 and ky < 0:
                            axisType = ZBisect
                        if kz < 0 and kx < 0 and ky < 0:
                            axisType = ThreeFold
                        self.axis_types.append(axisType)

                    else:
                        sys.exit("Atom %d not matched in forcefield!" % i_atom)

                else:
                    sys.exit("Atom %d not matched in forcefield!" % i_atom)
            self.axis_indices = np.array(self.axis_indices)
            self.axis_types = np.array(self.axis_types)
        else:
            self.axis_types = None
            self.axis_indices = None

        if "ethresh" in args:
            self.ethresh = args["ethresh"]
        if "step_pol" in args:
            self.step_pol = args["step_pol"]

        pme_force = ADMPPmeForce(box, self.axis_types, self.axis_indices,
                                 covalent_map, rc, self.ethresh, self.lmax,
                                 self.lpol, self.lpme, self.step_pol)
        self.pme_force = pme_force

        def potential_fn(positions, box, pairs, params):
            params = params['ADMPPmeForce']
            mScales = params["mScales"]
            Q_local = params["Q_local"][map_atomtype]
            if self.lpol:
                pScales = params["pScales"]
                dScales = params["dScales"]
                pol = params["pol"][map_poltype]
                tholes = params["tholes"][map_poltype]

                return pme_force.get_energy(
                    positions,
                    box,
                    pairs,
                    Q_local,
                    pol,
                    tholes,
                    mScales,
                    pScales,
                    dScales,
                    pme_force.U_ind,
                )
            else:
                return pme_force.get_energy(positions, box, pairs, Q_local,
                                            mScales)

        self._jaxPotential = potential_fn

    def getJaxPotential(self):
        return self._jaxPotential

# app.forcefield.parsers["ADMPPmeForce"] = ADMPPmeGenerator.parseElement
jaxGenerators["ADMPPmeForce"] = ADMPPmeGenerator

class Potential:
    def __init__(self):
        self.dmff_potentials = {}
        self.omm_system = None

    def addDmffPotential(self, name, potential):
        self.dmff_potentials[name] = potential

    def addOmmSystem(self, system):
        self.omm_system = system

    def buildOmmContext(self, integrator=mm.VerletIntegrator(0.1)):
        if self.omm_system is None:
            raise DMFFException(
                "OpenMM system is not initialized in this object.")
        self.omm_context = mm.Context(self.omm_system, integrator)

    def getPotentialFunc(self, names=[]):
        if len(self.dmff_potentials) == 0:
            raise DMFFException("No DMFF function in this potential object.")

        def totalPE(positions, box, pairs, params):
            totale_list = [
                self.dmff_potentials[k](positions, box, pairs, params)
                for k in self.dmff_potentials.keys() if (len(names) == 0 or k in names)
            ]
            totale = jnp.sum(jnp.array(totale_list))
            return totale

        return totalPE


class Hamiltonian(app.forcefield.ForceField):
    def __init__(self, *xmlnames):
        super().__init__(*xmlnames)
        # parse XML forcefields
        self.fftree = ForcefieldTree('ForcefieldTree')
        self.xmlparser = XMLParser(self.fftree)
        self.xmlparser.parse(*xmlnames)

        self._jaxGenerators = []
        self._potentials = []
        self.paramtree = {}

        self.ommsys = None

        for child in self.fftree.children:
            if child.tag in jaxGenerators:
                self._jaxGenerators.append(jaxGenerators[child.tag](self))

        # initialize paramtree
        self.extractParameterTree()

        # hook generators to self._forces
        for jaxGen in self._jaxGenerators:
            self._forces.append(jaxGen)

    def getGenerators(self):
        return self._jaxGenerators

    def extractParameterTree(self):
        # load Force info
        for jaxgen in self._jaxGenerators:
            jaxgen.extract()

    def overwriteParameterTree(self):
        # write Force info
        for jaxgen in self._jaxGenerators:
            jaxgen.overwrite()
        pass

    def createPotential(self,
                        topology,
                        nonbondedMethod=app.NoCutoff,
                        nonbondedCutoff=1.0 * unit.nanometer,
                        jaxForces=[],
                        **args):
        # load_constraints_from_system_if_needed
        # create potentials

        system = self.createSystem(
            topology,
            nonbondedMethod=nonbondedMethod,
            nonbondedCutoff=nonbondedCutoff,
            **args,
        )
        removeIdx = []
        jaxGens = [i.name for i in self._jaxGenerators]
        for nf, force in enumerate(system.getForces()):
            if (len(jaxForces) > 0 and force.getName() in jaxForces) or (force.getName() in jaxGens):
                removeIdx.append(nf)
        for nf in removeIdx[::-1]:
            system.removeForce(nf)

        potObj = Potential()
        potObj.addOmmSystem(system)
        for generator in self._jaxGenerators:
            if len(jaxForces) > 0 and generator.name not in jaxForces:
                continue
            try:
                potentialImpl = generator.getJaxPotential()
                potObj.addDmffPotential(generator.name, potentialImpl)
            except Exception as e:
                print(e)
                pass

        return potObj

    def render(self, filename):
        self.overwriteParameterTree()
        self.xmlparser.write(filename)

    def getParameters(self):
        return self.paramtree

    def updateParameters(self, paramtree):
        def update_iter(node, ref):
            for key in ref:
                if isinstance(ref[key], dict):
                    update_iter(node[key], ref[key])
                else:
                    node[key] = ref[key]

        update_iter(self.paramtree, paramtree)


class HarmonicBondJaxGenerator:
    def __init__(self, ff:Hamiltonian):
        self.name = "HarmonicBondForce"
        self.ff:Hamiltonian = ff
        self.fftree:ForcefieldTree = ff.fftree
        self.paramtree:Dict = ff.paramtree

    def extract(self):
        """
        extract forcefield paramters from ForcefieldTree. 
        """
        lengths = self.fftree.get_attribs(f"{self.name}/Bond", "length")
        # get_attribs will return a list of list. 
        ks = self.fftree.get_attribs(f"{self.name}/Bond", "k")
        self.paramtree[self.name] = {}
        self.paramtree[self.name]["length"] = jnp.array(lengths)
        self.paramtree[self.name]["k"] = jnp.array(ks)

    def overwrite(self):
        """
        update parameters in the fftree by using paramtree of this generator.
        """
        self.fftree.set_attrib(f"{self.name}/Bond", "length",
                               self.paramtree[self.name]["length"])
        self.fftree.set_attrib(f"{self.name}/Bond", "k",
                               self.paramtree[self.name]["k"])

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        """
        This method will create a potential calculation kernel. It usually should do the following:
        
        1. Match the corresponding bond parameters according to the atomic types at both ends of each bond.

        2. Create a potential calculation kernel, and pass those mapped parameters to the kernel.

        3. assign the jax potential to the _jaxPotential.

        Args:
            Those args are the same as those in createSystem.
        """

        # initialize typemap
        matcher = TypeMatcher(self.fftree, "HarmonicBondForce/Bond")

        map_atom1, map_atom2, map_param = [], [], []
        n_bonds = len(data.bonds)
        # build map
        for i in range(n_bonds):
            idx1 = data.bonds[i].atom1
            idx2 = data.bonds[i].atom2
            type1 = data.atomType[data.atoms[idx1]]
            type2 = data.atomType[data.atoms[idx2]]
            ifFound, ifForward, nfunc = matcher.matchGeneral([type1, type2])
            if not ifFound:
                raise BaseException(
                    f"No parameter for bond ({idx1},{type1}) - ({idx2},{type2})"
                )
            map_atom1.append(idx1)
            map_atom2.append(idx2)
            map_param.append(nfunc)
        map_atom1 = np.array(map_atom1, dtype=int)
        map_atom2 = np.array(map_atom2, dtype=int)
        map_param = np.array(map_param, dtype=int)

        bforce = HarmonicBondJaxForce(map_atom1, map_atom2, map_param)

        def potential_fn(positions, box, pairs, params):
            return bforce.get_energy(positions, box, pairs,
                                     params[self.name]["k"],
                                     params[self.name]["length"])

        self._jaxPotential = potential_fn
        # self._top_data = data

    def getJaxPotential(self):
        return self._jaxPotential


jaxGenerators["HarmonicBondForce"] = HarmonicBondJaxGenerator


class HarmonicAngleJaxGenerator:
    def __init__(self, ff):
        self.name = "HarmonicAngleForce"
        self.ff = ff
        self.fftree = ff.fftree
        self.paramtree = ff.paramtree

    def extract(self):
        angles = self.fftree.get_attribs(f"{self.name}/Angle", "angle")
        ks = self.fftree.get_attribs(f"{self.name}/Angle", "k")
        self.paramtree[self.name] = {}
        self.paramtree[self.name]["angle"] = jnp.array(angles)
        self.paramtree[self.name]["k"] = jnp.array(ks)

    def overwrite(self):
        self.fftree.set_attrib(f"{self.name}/Angle", "angle",
                               self.paramtree[self.name]["angle"])
        self.fftree.set_attrib(f"{self.name}/Angle", "k",
                               self.paramtree[self.name]["k"])

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        matcher = TypeMatcher(self.fftree, "HarmonicAngleForce/Angle")

        map_atom1, map_atom2, map_atom3, map_param = [], [], [], []
        n_angles = len(data.angles)
        for nangle in range(n_angles):
            idx1 = data.angles[nangle][0]
            idx2 = data.angles[nangle][1]
            idx3 = data.angles[nangle][2]
            type1 = data.atomType[data.atoms[idx1]]
            type2 = data.atomType[data.atoms[idx2]]
            type3 = data.atomType[data.atoms[idx3]]
            ifFound, ifForward, nfunc = matcher.matchGeneral(
                [type1, type2, type3])
            if not ifFound:
                print(
                    f"No parameter for angle ({idx1},{type1}) - ({idx2},{type2}) - ({idx3},{type3})"
                )
            else:
                map_atom1.append(idx1)
                map_atom2.append(idx2)
                map_atom3.append(idx3)
                map_param.append(nfunc)
        map_atom1 = np.array(map_atom1, dtype=int)
        map_atom2 = np.array(map_atom2, dtype=int)
        map_atom3 = np.array(map_atom3, dtype=int)
        map_param = np.array(map_param, dtype=int)

        aforce = HarmonicAngleJaxForce(map_atom1, map_atom2, map_atom3,
                                       map_param)

        def potential_fn(positions, box, pairs, params):
            return aforce.get_energy(positions, box, pairs,
                                     params[self.name]["k"],
                                     params[self.name]["angle"])

        self._jaxPotential = potential_fn
        # self._top_data = data

    def getJaxPotential(self):
        return self._jaxPotential


jaxGenerators["HarmonicAngleForce"] = HarmonicAngleJaxGenerator


class PeriodicTorsionJaxGenerator:
    def __init__(self, ff):
        self.name = "PeriodicTorsionForce"
        self.ff = ff
        self.fftree = ff.fftree
        self.paramtree = ff.paramtree
        self.meta = {}

        self.meta["prop_order"] = defaultdict(list)
        self.meta["prop_nodeidx"] = defaultdict(list)

        self.meta["impr_order"] = defaultdict(list)
        self.meta["impr_nodeidx"] = defaultdict(list)

        self.max_pred_prop = 0
        self.max_pred_impr = 0

    def extract(self):
        propers = self.fftree.get_nodes("PeriodicTorsionForce/Proper")
        impropers = self.fftree.get_nodes("PeriodicTorsionForce/Improper")
        self.paramtree[self.name] = {}
        # propers
        prop_phase = defaultdict(list)
        prop_k = defaultdict(list)
        for nnode, node in enumerate(propers):
            for key in node.attrs:
                if "periodicity" in key:
                    order = int(key[-1])
                    phase = float(node.attrs[f"phase{order}"])
                    k = float(node.attrs[f"k{order}"])
                    periodicity = int(node.attrs[f"periodicity{order}"])
                    if self.max_pred_prop < periodicity:
                        self.max_pred_prop = periodicity
                    prop_phase[f"{periodicity}"].append(phase)
                    prop_k[f"{periodicity}"].append(k)
                    self.meta[f"prop_order"][f"{periodicity}"].append(order)
                    self.meta[f"prop_nodeidx"][f"{periodicity}"].append(nnode)

        self.paramtree[self.name]["prop_phase"] = {}
        self.paramtree[self.name]["prop_k"] = {}
        for npred in range(1, self.max_pred_prop + 1):
            self.paramtree[self.name]["prop_phase"][f"{npred}"] = jnp.array(
                prop_phase[f"{npred}"])
            self.paramtree[self.name]["prop_k"][f"{npred}"] = jnp.array(
                prop_k[f"{npred}"])
        if self.max_pred_prop == 0:
            del self.paramtree[self.name]["prop_phase"]
            del self.paramtree[self.name]["prop_k"]

        # impropers
        impr_phase = defaultdict(list)
        impr_k = defaultdict(list)
        for nnode, node in enumerate(impropers):
            for key in node.attrs:
                if "periodicity" in key:
                    order = int(key[-1])
                    phase = float(node.attrs[f"phase{order}"])
                    k = float(node.attrs[f"k{order}"])
                    periodicity = int(node.attrs[f"periodicity{order}"])
                    if self.max_pred_impr < periodicity:
                        self.max_pred_impr = periodicity
                    impr_phase[f"{periodicity}"].append(phase)
                    impr_k[f"{periodicity}"].append(k)
                    self.meta[f"impr_order"][f"{periodicity}"].append(order)
                    self.meta[f"impr_nodeidx"][f"{periodicity}"].append(nnode)

        self.paramtree[self.name]["impr_phase"] = {}
        self.paramtree[self.name]["impr_k"] = {}
        for npred in range(1, self.max_pred_impr + 1):
            self.paramtree[self.name]["impr_phase"][f"{npred}"] = jnp.array(
                impr_phase[f"{npred}"])
            self.paramtree[self.name]["impr_k"][f"{npred}"] = jnp.array(
                impr_k[f"{npred}"])
        if self.max_pred_impr == 0:
            del self.paramtree[self.name]["impr_phase"]
            del self.paramtree[self.name]["impr_k"]

    def overwrite(self):
        propers = self.fftree.get_nodes("PeriodicTorsionForce/Proper")
        impropers = self.fftree.get_nodes("PeriodicTorsionForce/Improper")
        prop_data = [{} for _ in propers]
        impr_data = [{} for _ in impropers]
        # make propers
        for periodicity in range(1, self.max_pred_prop+1):
            nterms = len(
                self.paramtree[self.name][f"prop_phase"][f"{periodicity}"])
            for nitem in range(nterms):
                phase = self.paramtree[
                    self.name][f"prop_phase"][f"{periodicity}"][nitem]
                k = self.paramtree[
                    self.name][f"prop_k"][f"{periodicity}"][nitem]
                nodeidx = self.meta[f"prop_nodeidx"][f"{periodicity}"][nitem]
                order = self.meta[f"prop_order"][f"{periodicity}"][nitem]
                prop_data[nodeidx][f"phase{order}"] = phase
                prop_data[nodeidx][f"k{order}"] = k
        if "prop_phase" in self.paramtree[self.name]:
            self.fftree.set_node("PeriodicTorsionForce/Proper", prop_data)

        # make impropers
        for periodicity in range(1, self.max_pred_impr+1):
            nterms = len(
                self.paramtree[self.name][f"impr_phase"][f"{periodicity}"])
            for nitem in range(nterms):
                phase = self.paramtree[
                    self.name][f"impr_phase"][f"{periodicity}"][nitem]
                k = self.paramtree[
                    self.name][f"impr_k"][f"{periodicity}"][nitem]
                nodeidx = self.meta[f"impr_nodeidx"][f"{periodicity}"][nitem]
                order = self.meta[f"impr_order"][f"{periodicity}"][nitem]
                impr_data[nodeidx][f"phase{order}"] = phase
                impr_data[nodeidx][f"k{order}"] = k
        if "impr_phase" in self.paramtree[self.name]:
            self.fftree.set_node("PeriodicTorsionForce/Improper", impr_data)

    def createForce(self, sys, data, nonbondedMethod, nonbondedCutoff, args):
        proper_matcher = TypeMatcher(self.fftree,
                                     "PeriodicTorsionForce/Proper")
        map_prop_atom1 = {i: [] for i in range(1, self.max_pred_prop + 1)}
        map_prop_atom2 = {i: [] for i in range(1, self.max_pred_prop + 1)}
        map_prop_atom3 = {i: [] for i in range(1, self.max_pred_prop + 1)}
        map_prop_atom4 = {i: [] for i in range(1, self.max_pred_prop + 1)}
        map_prop_param = {i: [] for i in range(1, self.max_pred_prop + 1)}
        n_matched_props = 0
        for torsion in data.propers:
            types = [data.atomType[data.atoms[torsion[i]]] for i in range(4)]
            ifFound, ifForward, nnode = proper_matcher.matchGeneral(types)
            if not ifFound:
                continue
            # find terms for node
            for periodicity in range(1, self.max_pred_prop + 1):
                idx = findItemInList(
                    nnode, self.meta[f"prop_nodeidx"][f"{periodicity}"])
                if idx < 0:
                    continue
                n_matched_props += 1
                map_prop_atom1[periodicity].append(torsion[0])
                map_prop_atom2[periodicity].append(torsion[1])
                map_prop_atom3[periodicity].append(torsion[2])
                map_prop_atom4[periodicity].append(torsion[3])
                map_prop_param[periodicity].append(idx)

        impr_matcher = TypeMatcher(self.fftree,
                                   "PeriodicTorsionForce/Improper")
        try:
            ordering = self.fftree.get_attribs("PeriodicTorsionForce",
                                              "ordering")[0]
        except KeyError as e:
            ordering = "default"
        map_impr_atom1 = {i: [] for i in range(1, self.max_pred_impr + 1)}
        map_impr_atom2 = {i: [] for i in range(1, self.max_pred_impr + 1)}
        map_impr_atom3 = {i: [] for i in range(1, self.max_pred_impr + 1)}
        map_impr_atom4 = {i: [] for i in range(1, self.max_pred_impr + 1)}
        map_impr_param = {i: [] for i in range(1, self.max_pred_impr + 1)}
        n_matched_imprs = 0
        for impr in data.impropers:
            match = impr_matcher.matchImproper(impr, data, ordering=ordering)
            if match is not None:
                (a1, a2, a3, a4, nnode) = match
                n_matched_imprs += 1
                # find terms for node
                for periodicity in range(1, self.max_pred_impr + 1):
                    idx = findItemInList(
                        nnode, self.meta[f"impr_nodeidx"][f"{periodicity}"])
                    if idx < 0:
                        continue
                    if ordering == 'smirnoff':
                        # Add all torsions in trefoil
                        map_impr_atom1[periodicity].append(a1)
                        map_impr_atom2[periodicity].append(a2)
                        map_impr_atom3[periodicity].append(a3)
                        map_impr_atom4[periodicity].append(a4)
                        map_impr_param[periodicity].append(idx)
                        map_impr_atom1[periodicity].append(a1)
                        map_impr_atom2[periodicity].append(a3)
                        map_impr_atom3[periodicity].append(a4)
                        map_impr_atom4[periodicity].append(a2)
                        map_impr_param[periodicity].append(idx)
                        map_impr_atom1[periodicity].append(a1)
                        map_impr_atom2[periodicity].append(a4)
                        map_impr_atom3[periodicity].append(a2)
                        map_impr_atom4[periodicity].append(a3)
                        map_impr_param[periodicity].append(idx)
                    else:
                        map_impr_atom1[periodicity].append(a1)
                        map_impr_atom2[periodicity].append(a2)
                        map_impr_atom3[periodicity].append(a3)
                        map_impr_atom4[periodicity].append(a4)
                        map_impr_param[periodicity].append(idx)

        props = [
            PeriodicTorsionJaxForce(jnp.array(map_prop_atom1[p], dtype=int),
                                    jnp.array(map_prop_atom2[p], dtype=int),
                                    jnp.array(map_prop_atom3[p], dtype=int),
                                    jnp.array(map_prop_atom4[p], dtype=int),
                                    jnp.array(map_prop_param[p], dtype=int), p)
            for p in range(1, self.max_pred_prop + 1)
        ]
        imprs = [
            PeriodicTorsionJaxForce(jnp.array(map_impr_atom1[p], dtype=int),
                                    jnp.array(map_impr_atom2[p], dtype=int),
                                    jnp.array(map_impr_atom3[p], dtype=int),
                                    jnp.array(map_impr_atom4[p], dtype=int),
                                    jnp.array(map_impr_param[p], dtype=int), p)
            for p in range(1, self.max_pred_impr + 1)
        ]

        def potential_fn(positions, box, pairs, params):
            prop_sum = sum([
                props[i].get_energy(
                    positions, box, pairs,
                    params["PeriodicTorsionForce"]["prop_k"][f"{i+1}"],
                    params["PeriodicTorsionForce"]["prop_phase"][f"{i+1}"])
                for i in range(self.max_pred_prop)
            ])
            impr_sum = sum([
                imprs[i].get_energy(
                    positions, box, pairs,
                    params["PeriodicTorsionForce"]["impr_k"][f"{i+1}"],
                    params["PeriodicTorsionForce"]["impr_phase"][f"{i+1}"])
                for i in range(self.max_pred_impr)
            ])

            return prop_sum + impr_sum

        self._jaxPotential = potential_fn

    def getJaxPotential(self):
        return self._jaxPotential


jaxGenerators["PeriodicTorsionForce"] = PeriodicTorsionJaxGenerator


class NonbondedJaxGenerator:
    def __init__(self, ff):
        self.name = "NonbondedForce"
        self.ff = ff
        self.fftree = ff.fftree
        self.paramtree = ff.paramtree
        self.paramtree[self.name] = {}
        self.paramtree[self.name]["sigfix"] = jnp.array([])
        self.paramtree[self.name]["epsfix"] = jnp.array([])

        self.from_force = []
        self.from_residue = []
        self.ra2idx = {}
        self.idx2rai = {}

    def extract(self):
        self.from_residue = self.fftree.get_attribs(
            "NonbondedForce/UseAttributeFromResidue", "name")
        self.from_force = [
            i for i in ["charge", "sigma", "epsilon"]
            if i not in self.from_residue
        ]
        # Build per-atom array for from_force
        for prm in self.from_force:
            vals = self.fftree.get_attribs("NonbondedForce/Atom", prm)
            self.paramtree[self.name][prm] = jnp.array(vals)
        # Build per-atom array for from_residue
        residues = self.fftree.get_nodes("Residues/Residue")
        resvals = {k: [] for k in self.from_residue}
        for resnode in residues:
            resname = resnode.attrs["name"]
            resvals[resname] = []
            atomname = resnode.get_attribs("Atom", "name")
            shift = len(self.ra2idx)
            for natom, aname in enumerate(atomname):
                self.ra2idx[(resname, aname)] = shift + natom
                self.idx2rai[shift + natom] = (resname, atomname, natom)
            for prm in self.from_residue:
                atomval = resnode.get_attribs("Atom", prm)
                resvals[prm].extend(atomval)
        for prm in self.from_residue:
            self.paramtree[self.name][prm] = jnp.array(resvals[prm])
        # Build coulomb14scale and lj14scale
        coulomb14scale, lj14scale = self.fftree.get_attribs("NonbondedForce",
                                                ["coulomb14scale", "lj14scale"])[0]
        self.paramtree[self.name]["coulomb14scale"] = jnp.array([coulomb14scale])
        self.paramtree[self.name]["lj14scale"] = jnp.array([lj14scale])

    def overwrite(self):
        # write coulomb14scale
        self.fftree.set_attrib("NonbondedForce", "coulomb14scale",
                               self.paramtree[self.name]["coulomb14scale"])
        # write lj14scale
        self.fftree.set_attrib("NonbondedForce", "lj14scale",
                               self.paramtree[self.name]["lj14scale"])
        # write prm from force
        for prm in self.from_force:
            self.fftree.set_attrib("NonbondedForce/Atom", prm,
                                   self.paramtree[self.name][prm])
        # write prm from residue
        residues = self.fftree.get_nodes("Residues/Residue")
        for prm in self.from_residue:
            vals = self.paramtree[self.name][prm]
            data = []
            for idx in range(vals.shape[0]):
                rname, atomname, aidx = self.idx2rai[idx]
                data.append((rname, aidx, vals[idx]))

            for resnode in residues:
                tmp = sorted(
                    [d for d in data if d[0] == resnode.attrs["name"]],
                    key=lambda x: x[1])
                resnode.set_attrib("Atom", prm, [t[2] for t in tmp])

    def createForce(self, system, data, nonbondedMethod, nonbondedCutoff,
                    args):
        methodMap = {
            app.NoCutoff: "NoCutoff",
            app.CutoffPeriodic: "CutoffPeriodic",
            app.CutoffNonPeriodic: "CutoffNonPeriodic",
            app.PME: "PME",
        }
        if nonbondedMethod not in methodMap:
            raise DMFFException("Illegal nonbonded method for NonbondedForce")
        isNoCut = False
        if nonbondedMethod is app.NoCutoff:
            isNoCut = True

        mscales_coul = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0,
                                  1.0])  # mscale for PME
        mscales_coul = mscales_coul.at[2].set(
            self.paramtree[self.name]["coulomb14scale"][0])
        mscales_lj = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])  # mscale for LJ
        mscales_lj = mscales_lj.at[2].set(
            self.paramtree[self.name]["lj14scale"][0])

        # Coulomb: only support PME for now
        # set PBC
        if nonbondedMethod not in [app.NoCutoff, app.CutoffNonPeriodic]:
            ifPBC = True
        else:
            ifPBC = False

        nbmatcher = TypeMatcher(self.fftree, "NonbondedForce/Atom")
        # load LJ from types
        maps = {}
        for prm in self.from_force:
            maps[prm] = []
            for atom in data.atoms:
                atype = data.atomType[atom]
                ifFound, _, nnode = nbmatcher.matchGeneral([atype])
                if not ifFound:
                    raise DMFFException(
                        "AtomType of %s mismatched in NonbondedForce" %
                        (str(atom)))
                maps[prm].append(nnode)
            maps[prm] = jnp.array(maps[prm], dtype=int)

        for prm in self.from_residue:
            maps[prm] = []
            for atom in data.atoms:
                resname, aname = atom.residue.name, atom.name
                maps[prm].append(self.ra2idx[(resname, aname)])
        # TODO: implement NBFIX
        map_nbfix = []
        map_nbfix = np.array(map_nbfix, dtype=int).reshape((-1, 2))

        colv_map = build_covalent_map(data, 6)

        if unit.is_quantity(nonbondedCutoff):
            r_cut = nonbondedCutoff.value_in_unit(unit.nanometer)
        else:
            r_cut = nonbondedCutoff
        if "switchDistance" in args and args["switchDistance"] is not None:
            r_switch = args["switchDistance"]
            r_switch = (r_switch if not unit.is_quantity(r_switch) else
                        r_switch.value_in_unit(unit.nanometer))
            ifSwitch = True
        else:
            r_switch = r_cut
            ifSwitch = False

        # PME Settings
        if nonbondedMethod is app.PME:
            a, b, c = system.getDefaultPeriodicBoxVectors()
            box = jnp.array([a._value, b._value, c._value])
            self.ethresh = args.get("ethresh", 1e-6)
            self.coeff_method = args.get("PmeCoeffMethod", "openmm")
            self.fourier_spacing = args.get("PmeSpacing", 0.1)
            kappa, K1, K2, K3 = setup_ewald_parameters(r_cut, self.ethresh,
                                                       box,
                                                       self.fourier_spacing,
                                                       self.coeff_method)

        map_lj = jnp.array(maps["sigma"])
        map_nbfix = jnp.array(map_nbfix)
        map_charge = jnp.array(maps["charge"])

        # Free Energy Settings #
        isFreeEnergy = args.get("isFreeEnergy", False)
        if isFreeEnergy:
            vdwLambda = args.get("vdwLambda", 0.0)
            coulLambda = args.get("coulLambda", 0.0)
            ifStateA = args.get("ifStateA", True)

            # soft-cores
            vdwSoftCore = args.get("vdwSoftCore", False)
            coulSoftCore = args.get("coulSoftCore", False)
            scAlpha = args.get("scAlpha", 0.0)
            scSigma = args.get("scSigma", 0.0)

            # couple
            coupleIndex = args.get("coupleIndex", [])
            if len(coupleIndex) > 0:
                coupleMask = [False for _ in range(len(data.atoms))]
                for atomIndex in coupleIndex:
                    coupleMask[atomIndex] = True
                coupleMask = jnp.array(coupleMask, dtype=bool)
            else:
                coupleMask = None

        if not isFreeEnergy:
            ljforce = LennardJonesForce(r_switch,
                                        r_cut,
                                        map_lj,
                                        map_nbfix,
                                        colv_map,
                                        isSwitch=ifSwitch,
                                        isPBC=ifPBC,
                                        isNoCut=isNoCut)
        else:
            ljforce = LennardJonesFreeEnergyForce(r_switch,
                                                  r_cut,
                                                  map_lj,
                                                  map_nbfix,
                                                  colv_map,
                                                  isSwitch=ifSwitch,
                                                  isPBC=ifPBC,
                                                  isNoCut=isNoCut,
                                                  feLambda=vdwLambda,
                                                  coupleMask=coupleMask,
                                                  useSoftCore=vdwSoftCore,
                                                  ifStateA=ifStateA,
                                                  sc_alpha=scAlpha,
                                                  sc_sigma=scSigma)

        ljenergy = ljforce.generate_get_energy()

        # dispersion correction
        useDispersionCorrection = args.get("useDispersionCorrection", False)
        if useDispersionCorrection:
            numTypes = self.paramtree[self.name]["sigma"].shape[0]
            countVec = np.zeros(numTypes, dtype=int)
            countMat = np.zeros((numTypes, numTypes), dtype=int)
            types, count = np.unique(map_lj, return_counts=True)

            for typ, cnt in zip(types, count):
                countVec[typ] += cnt
            for i in range(numTypes):
                for j in range(i, numTypes):
                    if i != j:
                        countMat[i, j] = countVec[i] * countVec[j]
                    else:
                        countMat[i, i] = countVec[i] * (countVec[i] - 1) // 2
            assert np.sum(countMat) == len(map_lj) * (len(map_lj) - 1) // 2

            colv_pairs = np.argwhere(
                np.logical_and(colv_map > 0, colv_map <= 3))
            for pair in colv_pairs:
                if pair[0] <= pair[1]:
                    tmp = (map_lj[pair[0]], map_lj[pair[1]])
                    t1, t2 = min(tmp), max(tmp)
                    countMat[t1, t2] -= 1

            if not isFreeEnergy:
                ljDispCorrForce = LennardJonesLongRangeForce(
                    r_cut, map_lj, map_nbfix, countMat)
            else:
                ljDispCorrForce = LennardJonesLongRangeFreeEnergyForce(
                    r_cut, map_lj, map_nbfix, countMat, vdwLambda, ifStateA,
                    coupleMask)
            ljDispEnergyFn = ljDispCorrForce.generate_get_energy()

        if not isFreeEnergy:
            if nonbondedMethod is not app.PME:
                # do not use PME
                if nonbondedMethod in [
                        app.CutoffPeriodic, app.CutoffNonPeriodic
                ]:
                    # use Reaction Field
                    coulforce = CoulReactionFieldForce(r_cut,
                                                       map_charge,
                                                       colv_map,
                                                       isPBC=ifPBC)
                if nonbondedMethod is app.NoCutoff:
                    # use NoCutoff
                    coulforce = CoulNoCutoffForce(map_charge, colv_map)
            else:
                coulforce = CoulombPMEForce(r_cut, map_charge, colv_map, kappa,
                                            (K1, K2, K3))
        else:
            assert nonbondedMethod is app.PME, "Only PME is supported in free energy calculations"
            coulforce = CoulombPMEFreeEnergyForce(r_cut,
                                                  map_charge,
                                                  colv_map,
                                                  kappa, (K1, K2, K3),
                                                  coulLambda,
                                                  ifStateA=ifStateA,
                                                  coupleMask=coupleMask,
                                                  useSoftCore=coulSoftCore,
                                                  sc_alpha=scAlpha,
                                                  sc_sigma=scSigma)

        coulenergy = coulforce.generate_get_energy()

        if not isFreeEnergy:

            def potential_fn(positions, box, pairs, params):

                # check whether args passed into potential_fn are jnp.array and differentiable
                # note this check will be optimized away by jit
                # it is jit-compatiable
                isinstance_jnp(positions, box, params)

                ljE = ljenergy(positions, box, pairs,
                               params[self.name]["epsilon"],
                               params[self.name]["sigma"],
                               params[self.name]["epsfix"],
                               params[self.name]["sigfix"], mscales_lj)
                coulE = coulenergy(positions, box, pairs,
                                   params[self.name]["charge"], mscales_coul)

                if useDispersionCorrection:
                    ljDispEnergy = ljDispEnergyFn(box,
                                                  params[self.name]['epsilon'],
                                                  params[self.name]['sigma'],
                                                  params[self.name]['epsfix'],
                                                  params[self.name]['sigfix'])

                    return ljE + coulE + ljDispEnergy
                else:
                    return ljE + coulE

            self._jaxPotential = potential_fn
        else:
            # Free Energy
            @jit_condition()
            def potential_fn(positions, box, pairs, params, vdwLambda,
                             coulLambda):
                ljE = ljenergy(positions, box, pairs,
                               params[self.name]["epsilon"],
                               params[self.name]["sigma"],
                               params[self.name]["epsfix"],
                               params[self.name]["sigfix"], mscales_lj,
                               vdwLambda)
                coulE = coulenergy(positions, box, pairs,
                                   params[self.name]["charge"], mscales_coul,
                                   coulLambda)

                if useDispersionCorrection:
                    ljDispEnergy = ljDispEnergyFn(box,
                                                  params[self.name]['epsilon'],
                                                  params[self.name]['sigma'],
                                                  params[self.name]['epsfix'],
                                                  params[self.name]['sigfix'],
                                                  vdwLambda)
                    return ljE + coulE + ljDispEnergy
                else:
                    return ljE + coulE

            self._jaxPotential = potential_fn

    def getJaxPotential(self):
        return self._jaxPotential


jaxGenerators["NonbondedForce"] = NonbondedJaxGenerator
