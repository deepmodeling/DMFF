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
from dmff.utils import jit_condition, isinstance_jnp, DMFFException, findItemInList
from dmff.fftree import ForcefieldTree, XMLParser, TypeMatcher
from collections import defaultdict


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


jaxGenerators = {}


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
                for k in self.dmff_potentials.keys()
                if (len(names) == 0 or k in names)
            ]
            totale = jnp.sum(jnp.array(totale_list))
            return totale

        return totalPE


class Hamiltonian(app.forcefield.ForceField):
    def __init__(self, *xmlnames):
        super().__init__(*xmlnames)
        self._pseudo_ff = app.ForceField(*xmlnames)
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
        pseudo_data = app.ForceField._SystemData(topology)
        residueTemplates = {}
        templateForResidue = self._pseudo_ff._matchAllResiduesToTemplates(pseudo_data, topology, residueTemplates, False)
        self.templateNameForResidue = [i.name for i in templateForResidue]

        system = self.createSystem(
            topology,
            nonbondedMethod=nonbondedMethod,
            nonbondedCutoff=nonbondedCutoff,
            **args,
        )
        removeIdx = []
        jaxGens = [i.name for i in self._jaxGenerators]
        for nf, force in enumerate(system.getForces()):
            if (len(jaxForces) > 0
                    and force.getName() in jaxForces) or (force.getName()
                                                          in jaxGens):
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