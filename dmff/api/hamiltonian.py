import openmm.app as app
import openmm.unit as unit
from .xmlio import XMLIO
from .paramset import ParamSet
from .topology import DMFFTopology
from ..utils import DMFFException
import jax
import jax.numpy as jnp
import numpy as np

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


class Hamiltonian:
    # 存Residue templates
    # 存Generators

    def __init__(self, xmlfiles):
        self._xmlio = XMLIO()
        self.generators = {}
        self.templates = []
        self.paramset = ParamSet()

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
                raise BaseException(f"Generator for {key} is not implemented.")
            self.generators[key] = _DMFFGenerators[key](
                ffinfo, self.paramset)

    def createJaxPotential(self, topdata: DMFFTopology, nonbondedMethod=app.NoCutoff,
                           nonbondedCutoff=1.0 * unit.nanometer, args={}, forces=None):
        efuncs = {}
        for key in self.generators:
            gen = self.generators[key]
            if forces is not None and gen.getName() not in forces:
                continue
            efuncs[gen.getName()] = gen.createPotential(topdata, nonbondedMethod,
                                                        nonbondedCutoff, args)

        update_func = topdata.buildVSiteUpdateFunction()
        def efunc_total(pos, box, pairs, prms):
            pos_updated = update_func(pos)
            return sum([e[1](pos_updated, box, pairs, prms) for e in efuncs.items()])

        return efunc_total

    def renderXML(self, out: str, residues=True, atomtypes=True, forces=True, operators=True):
        for key in self.generators.keys():
            self.generators[key].overwrite(self.paramset)
        self._xmlio.writeXML(out, self.ffinfo, write_residues=residues,
                             write_forces=forces, write_atomtypes=atomtypes, write_operators=operators)
