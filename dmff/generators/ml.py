from ..api.topology import DMFFTopology
from ..api.paramset import ParamSet
from ..api.hamiltonian import _DMFFGenerators
from ..utils import DMFFException, isinstance_jnp
from ..utils import jit_condition
import numpy as np
import jax
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
import pickle

from ..sgnn.graph import MAX_VALENCE, TopGraph, from_pdb
from ..sgnn.gnn import MolGNNForce, prm_transform_f2i
from ..eann.eann import EANNForce, get_elem_indices

class SGNNGenerator:
    def __init__(self, ffinfo: dict, paramset: ParamSet):

        self.name = "SGNNForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self.key_type = None

        self.file = self.ffinfo["Forces"][self.name]["meta"]["file"]
        self.nn = int(self.ffinfo["Forces"][self.name]["meta"]["nn"])
        self.pdb = self.ffinfo["Forces"][self.name]["meta"]["pdb"]

        # load ML potential parameters
        with open(self.file, 'rb') as ifile:
            params = pickle.load(ifile)

        # convert to jnp array
        for k in params:
            params[k] = jnp.array(params[k])
            # set mask to all true
            paramset.addParameter(params[k], k, field=self.name, mask=jnp.ones(params[k].shape))

        # mask = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape), params)
        # paramset.addParameter(params, "params", field=self.name, mask=mask)
       

    def getName(self) -> str:
        return self.name

    def overwrite(self, paramset):
        # do not use xml to handle ML potentials
        # for ML potentials, xml only documents param file path
        # so for ML potentials, overwrite function overwrites the file directly
        with open(self.file, 'wb') as ofile:
            pickle.dump(paramset[self.name], ofile)
        return

    def createPotential(self, topdata: DMFFTopology, nonbondedMethod, nonbondedCutoff, **kwargs):
        self.G = from_pdb(self.pdb)
        n_atoms = topdata.getNumAtoms()
        self.model = MolGNNForce(self.G, nn=self.nn)
        n_layers = self.model.n_layers
        def potential_fn(positions, box, pairs, params):
            # convert unit to angstrom
            positions = positions * 10
            box = box * 10
            prms = prm_transform_f2i(params[self.name], n_layers)
            return self.model.get_energy(positions, box, prms)

        self._jaxPotential = potential_fn
        return potential_fn

    def getJaxPotential(self):
        return self._jaxPotential

_DMFFGenerators["SGNNForce"] = SGNNGenerator

class EANNGenerator:
    def __init__(self, ffinfo: dict, paramset: ParamSet):

        self.name = "EANNForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self.key_type = None

        self.file = self.ffinfo["Forces"][self.name]["meta"]["file"]
        self.ngto = int(self.ffinfo["Forces"][self.name]["meta"]["ngto"])
        self.nipsin = int(self.ffinfo["Forces"][self.name]["meta"]["nipsin"])
        self.rc = int(self.ffinfo["Forces"][self.name]["meta"]["rc"])

        self.pdb = self.ffinfo["Forces"][self.name]["meta"]["pdb"]
        self.ommtopology = app.PDBFile(self.pdb).topology
        # load ML potential parameters
        with open(self.file, 'rb') as ifile:
            params = pickle.load(ifile)
        self.params = params
        # convert to jnp array
        for k in params:
            params[k] = params[k]
            # set mask to all true
            paramset.addParameter(params[k], k, field=self.name, mask=jnp.ones(params[k].shape))

        # mask = jax.tree_util.tree_map(lambda x: jnp.ones(x.shape), params)
        # paramset.addParameter(params, "params", field=self.name, mask=mask)
       

    def getName(self) -> str:
        return self.name

    def overwrite(self, params):
        # do not use xml to handle ML potentials
        # for ML potentials, xml only documents param file path
        # so for ML potentials, overwrite function overwrites the file directly
        with open(self.file, 'wb') as ofile:
            pickle.dump(paramset[self.name], ofile)
        return

    def createPotential(self, topdata: DMFFTopology, nonbondedMethod, nonbondedCutoff, **kwargs):
        n_atoms = topdata.getNumAtoms()
        n_elem, elem_indices = get_elem_indices(self.ommtopology)
        self.model = EANNForce(n_elem, elem_indices, n_gto=self.ngto, nipsin=self.nipsin, rc=self.rc)
        n_layers = self.model.n_layers
        
        has_aux = False
        if "has_aux" in kwargs and kwargs["has_aux"]:
            has_aux = True
        
        def potential_fn(positions, box, pairs, params, aux=None):
            # convert unit to angstrom
            positions = positions * 10
            box = box * 10
            if has_aux:
                return self.model.get_energy(positions, box, pairs, params[self.name]), aux
            else:
                return self.model.get_energy(positions, box, pairs, params[self.name])

        self._jaxPotential = potential_fn
        return potential_fn

    def getJaxPotential(self):
        return self._jaxPotential

_DMFFGenerators["EANNForce"] = EANNGenerator


