import openmm.app as app
import openmm.unit as unit
from .xmlio import XMLIO
from .paramset import ParamSet
from .topology import DMFFTopology
from ..operators.templatetype import TemplateATypeOperator
from ..operators.templatevsite import TemplateVSiteOperator
from ..utils import DMFFException
import jax
import jax.numpy as jnp
import numpy as np
from typing import Union, Callable

_DMFFGenerators = {}


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


class Potential:

    def __init__(self, topology: DMFFTopology, update_func: Callable):
        self.dmff_potentials = {}
        self.update_func = update_func
        self.meta = {}
        self.topology = topology
        self.meta["cov_map"] = topology.buildCovMat()
        for key in self.topology._meta:
            self.meta[key] = self.topology._meta[key]

    def add(self, name, func):
        self.dmff_potentials[name] = func

    def getPotentialFunc(self, names: list=[]):
        if isinstance(names, str):
            names = [names]
        if len(names) == 0:
            names = self.dmff_potentials.keys()
        def efunc(positions, box, pairs, prms):
            pos_update = self.update_func(positions)
            return sum([self.dmff_potentials[name](pos_update, box, pairs, prms) for name in names])
        return efunc


class Hamiltonian:
    # 存Residue templates
    # 存Generators

    def __init__(self, *args):
        self._xmlio = XMLIO()
        self.generators = {}
        self.templates = []
        self.paramset = ParamSet()

        xmlfiles = [i for i in args if isinstance(i, str)]
        if isinstance(xmlfiles, str):
            self._xmlio.loadXML(xmlfiles)
        else:
            for xml in xmlfiles:
                self._xmlio.loadXML(xml)
        ffinfo = self._xmlio.parseXML()
        self.ffinfo = ffinfo

        # 处理Forces
        for key in ffinfo["Forces"].keys():
            if key not in _DMFFGenerators:
                print(f"Generator for {key} is not implemented.")
            else:
                self.generators[key] = _DMFFGenerators[key](
                    ffinfo, self.paramset)
                
    def getGenerators(self):
        return [g for g in self.generators.values()]

    def createPotential(self, topdata: Union[DMFFTopology, app.Topology], nonbondedMethod=app.NoCutoff,
                           nonbondedCutoff=1.0 * unit.nanometer, **kwargs):
        if isinstance(topdata, app.Topology):
            topdata = DMFFTopology(from_top=topdata)
            # initialize template operator
            vsite = TemplateVSiteOperator(self.ffinfo)
            topdata = vsite(topdata)
            template = TemplateATypeOperator(self.ffinfo)
            topdata = template(topdata)

        efuncs = {}
        for key in self.generators:
            gen = self.generators[key]
            efuncs[gen.getName()] = gen.createPotential(topdata, nonbondedMethod,
                                                        nonbondedCutoff, **kwargs)

        update_func = topdata.buildVSiteUpdateFunction()
        potential = Potential(topdata, update_func)
        for key in efuncs:
            potential.add(key, efuncs[key])

        return potential

    def renderXML(self, out: str, residues=True, atomtypes=True, forces=True, operators=True):
        for key in self.generators.keys():
            self.generators[key].overwrite(self.paramset)
        self._xmlio.writeXML(out, self.ffinfo, write_residues=residues,
                             write_forces=forces, write_atomtypes=atomtypes, write_operators=operators)

    def getParameters(self):
        return self.paramset