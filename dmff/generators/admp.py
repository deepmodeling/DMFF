import sys

import numpy as np
import jax.numpy as jnp

import openmm.app as app
import openmm.unit as unit

import dmff
from dmff.api import build_covalent_map
from dmff.admp.disp_pme import ADMPDispPmeForce
from dmff.admp.multipole import convert_cart2harm, convert_harm2cart
from dmff.admp.pairwise import (
    TT_damping_qq_c6_kernel, 
    generate_pairwise_interaction,
    slater_disp_damping_kernel, 
    slater_sr_kernel,
    TT_damping_qq_kernel
)
from dmff.admp.pme import ADMPPmeForce


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

        mScales = [
            self.fftree.get_attribs(f'{self.name}', f'mScale1{i}')[0]
            for i in range(2, 7)
        ]
        mScales.append(1.0)
        self.paramtree[self.name] = {}
        self.paramtree[self.name]['mScales'] = jnp.array(mScales)

        ABQC = self.fftree.get_attribs(f'{self.name}/Atom',
                                       ['A', 'B', 'Q', 'C6', 'C8', 'C10'])

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
        self.covalent_map = build_covalent_map(data, 6)
        # here box is only used to setup ewald parameters, no need to be differentiable
        a, b, c = system.getDefaultPeriodicBoxVectors()
        box = jnp.array([a._value, b._value, c._value]) * 10
        # get the admp calculator
        rc = nonbondedCutoff.value_in_unit(unit.angstrom)

        # get calculator
        if "ethresh" in args:
            self.ethresh = args["ethresh"]

        Force_DispPME = ADMPDispPmeForce(box,
                                         rc,
                                         self.ethresh,
                                         self.pmax,
                                         lpme=self.lpme)
        self.disp_pme_force = Force_DispPME
        pot_fn_lr = Force_DispPME.get_energy
        pot_fn_sr = generate_pairwise_interaction(TT_damping_qq_c6_kernel,
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

        self.fftree.set_attrib(f'{self.name}', 'mScale12',
                               [self.paramtree[self.name]['mScales'][0]])
        self.fftree.set_attrib(f'{self.name}', 'mScale13',
                               [self.paramtree[self.name]['mScales'][1]])
        self.fftree.set_attrib(f'{self.name}', 'mScale14',
                               [self.paramtree[self.name]['mScales'][2]])
        self.fftree.set_attrib(f'{self.name}', 'mScale15',
                               [self.paramtree[self.name]['mScales'][3]])
        self.fftree.set_attrib(f'{self.name}', 'mScale16',
                               [self.paramtree[self.name]['mScales'][4]])

        self.fftree.set_attrib(f'{self.name}/Atom', 'A',
                               [self.paramtree[self.name]['A']])
        self.fftree.set_attrib(f'{self.name}/Atom', 'B',
                               [self.paramtree[self.name]['B']])
        self.fftree.set_attrib(f'{self.name}/Atom', 'Q',
                               [self.paramtree[self.name]['Q']])
        self.fftree.set_attrib(f'{self.name}/Atom', 'C6',
                               [self.paramtree[self.name]['C6']])
        self.fftree.set_attrib(f'{self.name}/Atom', 'C8',
                               [self.paramtree[self.name]['C8']])
        self.fftree.set_attrib(f'{self.name}/Atom', 'C10',
                               [self.paramtree[self.name]['C10']])

    def getJaxPotential(self):
        return self._jaxPotential


dmff.api.jaxGenerators['ADMPDispForce'] = ADMPDispGenerator


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

        mScales = [
            self.fftree.get_attribs(f'{self.name}', f'mScale1{i}')[0]
            for i in range(2, 7)
        ]
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

        self.fftree.set_attrib(f'{self.name}', 'mScale12',
                               [self.paramtree[self.name]['mScales'][0]])
        self.fftree.set_attrib(f'{self.name}', 'mScale13',
                               [self.paramtree[self.name]['mScales'][1]])
        self.fftree.set_attrib(f'{self.name}', 'mScale14',
                               [self.paramtree[self.name]['mScales'][2]])
        self.fftree.set_attrib(f'{self.name}', 'mScale15',
                               [self.paramtree[self.name]['mScales'][3]])
        self.fftree.set_attrib(f'{self.name}', 'mScale16',
                               [self.paramtree[self.name]['mScales'][4]])

        self.fftree.set_attrib(f'{self.name}/Atom', 'C6',
                               self.paramtree[self.name]['C6'])
        self.fftree.set_attrib(f'{self.name}/Atom', 'C8',
                               self.paramtree[self.name]['C8'])
        self.fftree.set_attrib(f'{self.name}/Atom', 'C10',
                               self.paramtree[self.name]['C10'])

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
        self.covalent_map = build_covalent_map(data, 6)

        # here box is only used to setup ewald parameters, no need to be differentiable
        a, b, c = system.getDefaultPeriodicBoxVectors()
        box = jnp.array([a._value, b._value, c._value]) * 10
        # get the admp calculator
        rc = nonbondedCutoff.value_in_unit(unit.angstrom)

        # get calculator
        if "ethresh" in args:
            self.ethresh = args["ethresh"]

        disp_force = ADMPDispPmeForce(box, rc, self.ethresh, self.pmax,
                                      self.lpme)
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


dmff.api.jaxGenerators['ADMPDispPmeForce'] = ADMPDispPmeGenerator


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
        mScales = [
            self.fftree.get_attribs(f'{self.name}', f'mScale1{i}')[0]
            for i in range(2, 7)
        ]
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

        self.fftree.set_attrib(f'{self.name}', 'mScale12',
                               [self.paramtree[self.name]['mScales'][0]])
        self.fftree.set_attrib(f'{self.name}', 'mScale13',
                               [self.paramtree[self.name]['mScales'][1]])
        self.fftree.set_attrib(f'{self.name}', 'mScale14',
                               [self.paramtree[self.name]['mScales'][2]])
        self.fftree.set_attrib(f'{self.name}', 'mScale15',
                               [self.paramtree[self.name]['mScales'][3]])
        self.fftree.set_attrib(f'{self.name}', 'mScale16',
                               [self.paramtree[self.name]['mScales'][4]])

        self.fftree.set_attrib(f'{self.name}/Atom', 'B',
                               self.paramtree[self.name]['B'])
        self.fftree.set_attrib(f'{self.name}/Atom', 'Q',
                               self.paramtree[self.name]['Q'])

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
        self.covalent_map = build_covalent_map(data, 6)

        pot_fn_sr = generate_pairwise_interaction(TT_damping_qq_kernel,
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
dmff.api.jaxGenerators['QqTtDampingForce'] = QqTtDampingGenerator


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
        mScales = [
            self.fftree.get_attribs(f'{self.name}', f'mScale1{i}')[0]
            for i in range(2, 7)
        ]
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

        self.fftree.set_attrib(f'{self.name}', 'mScale12',
                               [self.paramtree[self.name]['mScales'][0]])
        self.fftree.set_attrib(f'{self.name}', 'mScale13',
                               [self.paramtree[self.name]['mScales'][1]])
        self.fftree.set_attrib(f'{self.name}', 'mScale14',
                               [self.paramtree[self.name]['mScales'][2]])
        self.fftree.set_attrib(f'{self.name}', 'mScale15',
                               [self.paramtree[self.name]['mScales'][3]])
        self.fftree.set_attrib(f'{self.name}', 'mScale16',
                               [self.paramtree[self.name]['mScales'][4]])

        self.fftree.set_attrib(f'{self.name}/Atom', 'B',
                               [self.paramtree[self.name]['B']])
        self.fftree.set_attrib(f'{self.name}/Atom', 'C6',
                               [self.paramtree[self.name]['C6']])
        self.fftree.set_attrib(f'{self.name}/Atom', 'C8',
                               [self.paramtree[self.name]['C8']])
        self.fftree.set_attrib(f'{self.name}/Atom', 'C10',
                               [self.paramtree[self.name]['C10']])

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
        self.covalent_map = build_covalent_map(data, 6)

        # WORKING
        pot_fn_sr = generate_pairwise_interaction(slater_disp_damping_kernel,
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


dmff.api.jaxGenerators['SlaterDampingForce'] = SlaterDampingGenerator


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
        mScales = [
            self.fftree.get_attribs(f'{self.name}', f'mScale1{i}')[0]
            for i in range(2, 7)
        ]
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

        self.fftree.set_attrib(f'{self.name}', 'mScale12',
                               [self.paramtree[self.name]['mScales'][0]])
        self.fftree.set_attrib(f'{self.name}', 'mScale13',
                               [self.paramtree[self.name]['mScales'][1]])
        self.fftree.set_attrib(f'{self.name}', 'mScale14',
                               [self.paramtree[self.name]['mScales'][2]])
        self.fftree.set_attrib(f'{self.name}', 'mScale15',
                               [self.paramtree[self.name]['mScales'][3]])
        self.fftree.set_attrib(f'{self.name}', 'mScale16',
                               [self.paramtree[self.name]['mScales'][4]])

        self.fftree.set_attrib(f'{self.name}/Atom', 'A',
                               [self.paramtree[self.name]['A']])
        self.fftree.set_attrib(f'{self.name}/Atom', 'B',
                               [self.paramtree[self.name]['B']])

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
        self.covalent_map = build_covalent_map(data, 6)

        pot_fn_sr = generate_pairwise_interaction(slater_sr_kernel,
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


dmff.api.jaxGenerators["SlaterExForce"] = SlaterExGenerator


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
dmff.api.jaxGenerators["SlaterSrEsForce"] = SlaterSrEsGenerator
dmff.api.jaxGenerators["SlaterSrPolForce"] = SlaterSrPolGenerator
dmff.api.jaxGenerators["SlaterSrDispForce"] = SlaterSrDispGenerator
dmff.api.jaxGenerators["SlaterDhfForce"] = SlaterDhfGenerator


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

        self.lmax = self.fftree.get_attribs(f'{self.name}',
                                            'lmax')[0]  # return [lmax]

        mScales = [
            self.fftree.get_attribs(f'{self.name}', f'mScale1{i}')[0]
            for i in range(2, 7)
        ]
        pScales = [
            self.fftree.get_attribs(f'{self.name}', f'pScale1{i}')[0]
            for i in range(2, 7)
        ]
        dScales = [
            self.fftree.get_attribs(f'{self.name}', f'dScale1{i}')[0]
            for i in range(2, 7)
        ]

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

        kx = [0 if kx_ is None else int(kx_) for kx_ in kx]
        ky = [0 if ky_ is None else int(ky_) for ky_ in ky]
        kz = [0 if kz_ is None else int(kz_) for kz_ in kz]

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
        qXZ = self.fftree.get_attribs(f'{self.name}/Atom', 'qXZ')
        qYZ = self.fftree.get_attribs(f'{self.name}/Atom', 'qYZ')

        # assume that polarize tag match the per atom type
        # pol_XX = self.fftree.get_attribs(f'{self.name}/Polarize', 'polarizabilityXX')
        # pol_YY = self.fftree.get_attribs(f'{self.name}/Polarize', 'polarizabilityYY')
        # pol_ZZ = self.fftree.get_attribs(f'{self.name}/Polarize', 'polarizabilityZZ')
        # thole_0 = self.fftree.get_attribs(f'{self.name}/Polarize', 'thole')
        if self.lpol:
            polarizabilityXX = self.fftree.get_attribs(f'{self.name}/Polarize',
                                                       'polarizabilityXX')
            polarizabilityYY = self.fftree.get_attribs(f'{self.name}/Polarize',
                                                       'polarizabilityYY')
            polarizabilityZZ = self.fftree.get_attribs(f'{self.name}/Polarize',
                                                       'polarizabilityZZ')
            thole = self.fftree.get_attribs(f'{self.name}/Polarize', 'thole')
            polarize_types = self.fftree.get_attribs(f'{self.name}/Polarize',
                                                     'type')
            if type(polarize_types[0]) != str:
                polarize_types = np.array(polarize_types,
                                          dtype=int).astype(str)
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
            Q[:, 8] = qXZ
            Q[:, 9] = qYZ
            Q[:, 4:10] *= 300

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

        self.fftree.set_attrib(f'{self.name}', 'mScale12',
                               [self.paramtree[self.name]['mScales'][0]])
        self.fftree.set_attrib(f'{self.name}', 'mScale13',
                               [self.paramtree[self.name]['mScales'][1]])
        self.fftree.set_attrib(f'{self.name}', 'mScale14',
                               [self.paramtree[self.name]['mScales'][2]])
        self.fftree.set_attrib(f'{self.name}', 'mScale15',
                               [self.paramtree[self.name]['mScales'][3]])
        self.fftree.set_attrib(f'{self.name}', 'mScale16',
                               [self.paramtree[self.name]['mScales'][4]])

        self.fftree.set_attrib(f'{self.name}', 'pScale12',
                               [self.paramtree[self.name]['pScales'][0]])
        self.fftree.set_attrib(f'{self.name}', 'pScale13',
                               [self.paramtree[self.name]['pScales'][1]])
        self.fftree.set_attrib(f'{self.name}', 'pScale14',
                               [self.paramtree[self.name]['pScales'][2]])
        self.fftree.set_attrib(f'{self.name}', 'pScale15',
                               [self.paramtree[self.name]['pScales'][3]])
        self.fftree.set_attrib(f'{self.name}', 'pScale16',
                               [self.paramtree[self.name]['pScales'][4]])

        self.fftree.set_attrib(f'{self.name}', 'dScale12',
                               [self.paramtree[self.name]['dScales'][0]])
        self.fftree.set_attrib(f'{self.name}', 'dScale13',
                               [self.paramtree[self.name]['dScales'][1]])
        self.fftree.set_attrib(f'{self.name}', 'dScale14',
                               [self.paramtree[self.name]['dScales'][2]])
        self.fftree.set_attrib(f'{self.name}', 'dScale15',
                               [self.paramtree[self.name]['dScales'][3]])
        self.fftree.set_attrib(f'{self.name}', 'dScale16',
                               [self.paramtree[self.name]['dScales'][4]])

        Q_global = convert_harm2cart(self.paramtree[self.name]['Q_local'],
                                     self.lmax)

        self.fftree.set_attrib(f'{self.name}/Atom', 'c0', Q_global[:, 0])
        self.fftree.set_attrib(f'{self.name}/Atom', 'dX', Q_global[:, 1])
        self.fftree.set_attrib(f'{self.name}/Atom', 'dY', Q_global[:, 2])
        self.fftree.set_attrib(f'{self.name}/Atom', 'dZ', Q_global[:, 3])
        self.fftree.set_attrib(f'{self.name}/Atom', 'qXX', Q_global[:, 4])
        self.fftree.set_attrib(f'{self.name}/Atom', 'qYY', Q_global[:, 5])
        self.fftree.set_attrib(f'{self.name}/Atom', 'qZZ', Q_global[:, 6])
        self.fftree.set_attrib(f'{self.name}/Atom', 'qXY', Q_global[:, 7])
        self.fftree.set_attrib(f'{self.name}/Atom', 'qXZ', Q_global[:, 8])
        self.fftree.set_attrib(f'{self.name}/Atom', 'qYZ', Q_global[:, 9])

        if self.lpol:
            # self.paramtree[self.name]['pol']: every element is the mean value of XX YY ZZ
            # get the number of polarize element
            n_pol = len(self.paramtree[self.name]['pol'])
            self.fftree.set_attrib(f'{self.name}/Polarize', 'polarizabilityXX',
                                   [self.paramtree[self.name]['pol'][0]] *
                                   n_pol)
            self.fftree.set_attrib(f'{self.name}/Polarize', 'polarizabilityYY',
                                   [self.paramtree[self.name]['pol'][1]] *
                                   n_pol)
            self.fftree.set_attrib(f'{self.name}/Polarize', 'polarizabilityZZ',
                                   [self.paramtree[self.name]['pol'][2]] *
                                   n_pol)
            self.fftree.set_attrib(f'{self.name}/Polarize', 'thole',
                                   self.paramtree[self.name]['tholes'])

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
            atype = data.atomType[
                data.atoms[i]]  # convert str to int to match atomTypes
            map_atomtype[i] = np.where(self.atomTypes == atype)[0][0]
            if self.lpol:
                map_poltype[i] = np.where(self.polarize_types == atype)[0][0]
        self.map_atomtype = map_atomtype
        if self.lpol:
            self.map_poltype = map_poltype

        # here box is only used to setup ewald parameters, no need to be differentiable
        a, b, c = system.getDefaultPeriodicBoxVectors()
        box = jnp.array([a._value, b._value, c._value]) * 10

        # get the admp calculator
        rc = nonbondedCutoff.value_in_unit(unit.angstrom)

        # build covalent map
        self.covalent_map = covalent_map = build_covalent_map(data, 6)

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

        pme_force = ADMPPmeForce(box, self.axis_types, self.axis_indices, rc,
                                 self.ethresh, self.lmax, self.lpol, self.lpme,
                                 self.step_pol)
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


dmff.api.jaxGenerators["ADMPPmeForce"] = ADMPPmeGenerator