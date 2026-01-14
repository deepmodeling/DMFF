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
import re
from functools import partial
from collections import OrderedDict
import copy

from ..sgnn.graph import MAX_VALENCE, TopGraph, from_pdb
from ..sgnn.gnn import MolGNNForce, prm_transform_f2i
from ..eann.eann import EANNForce, get_elem_indices
from ..api.topology import elem_to_index
from ..common.constants import EV2KJ

# load torch-related module
try:
    import torch
    import torch.nn as nn
    from ..torch_tools import t2j_pytree, j2t_pytree, wrap_torch_potential_kernel, t2j_extract_grad
    from torch2jax import t2j, j2t
except ImportError:
    pass

# load base-related module
try:
    from base.inference.calculator import get_parser
    import pymatgen.core.structure
except ImportError:
    pass


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
        self.rc = float(self.ffinfo["Forces"][self.name]["meta"]["rc"]) * 10

        self.pdb = self.ffinfo["Forces"][self.name]["meta"]["pdb"]
        self.ommtopology = app.PDBFile(self.pdb).topology
        # load ML potential parameters
        with open(self.file, 'rb') as ifile:
            params = pickle.load(ifile)
        self.params = params
        # convert to jnp array
        for k in params:
            params[k] = jnp.array(params[k])
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


class CustomTorchGenerator:

    def __init__(self, ffinfo: dict, paramset: ParamSet, dtype=None):
        """
        A custom torch model is specified by a full model checkpoint file
        The xml front end should be:

        ```xml
        <ForceField>
           <CustomTorchForce ckpt="model.pt" torch_script="True" dtype="float32"/>
        </ForceField>
        ```
        """

        self.name = "CustomTorchForce"
        self.ffinfo = ffinfo
        paramset.addField(self.name)
        self.key_type = None
        ffmeta = self.ffinfo["Forces"][self.name]["meta"]
        self.ckpt_file = None
        self.state_dict_file = None
        self.config_file = None
        self.torch_script = False

        if dtype is None:
            self.dtype = torch.float32
        else:
            self.dtype = dtype
        # precision
        if "dtype" in ffmeta["dtype"]:
            if '32' in ffmeta["dtype"]:
                self.dtype = torch.float32
            elif '64' in ffmeta["dtype"]:
                self.dtype = torch.float64
        self.ckpt_file = ffmeta["ckpt"]
        self.torch_script = (ffmeta["torch_script"] == 'True')

        if self.torch_script:
            self.load_method = torch.jit.load
            self.save_method = torch.jit.save
        else:
            self.load_method = partial(torch.load, weights_only=False)
            self.save_method = torch.save

        self.model = self._initialize_model()
        
        # now model is fully loaded, start to register parameters
        named_parameters = self.model.named_parameters()
        self.params_t = OrderedDict()
        for name, param in named_parameters:
            self.params_t[name] = param
        self.params = t2j_pytree(self.params_t)
        for k in self.params:
            # set mask to all true
            paramset.addParameter(self.params[k], k, field=self.name, mask=jnp.ones(self.params[k].shape))

        self.params_noopt = OrderedDict()
        state_dict = self.model.state_dict()
        for k in state_dict:
            if k not in self.params_t:
                self.params_noopt[k] = state_dict[k]
        return

    
    def _initialize_model(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = self.load_method(self.ckpt_file, map_location=self.device)
        # state dictionary file
        self.state_dict_file = re.sub('.([a-zA-Z0-9]+)$', '_sd.\g<1>', self.ckpt_file)
        return model

    def getName(self) -> str:
        return self.name

    def createPotential(self, topdata: DMFFTopology, nonbondedMethod, nonbondedCutoff, **kwargs):
        self.topdata = topdata
        self.n_atoms = topdata.getNumAtoms()

        # topo(primarily atom type) data
        self.atom_types = []
        for atom in topdata.atoms():
            element = atom.element.upper()
            self.atom_types.append(elem_to_index[element])
        self.atom_types = np.array(self.atom_types)

        # torch kernel
        def potential_torch_kernel(positions, box, pairs, params):

            # load parameter to model
            if self.name in params.keys():
                state_dict = copy.deepcopy(params[self.name])
            else:
                state_dict = copy.deepcopy(params)
            for k in self.params_noopt:
                state_dict[k] = self.params_noopt[k]

            # build a model object for every invokation to avoid gradient accumulation
            model = copy.deepcopy(self.model)
            model.load_state_dict(state_dict)
            results = model.forward(positions, box, self.atom_types)
            return results, model

        # jax wrapper
        @partial(jax.custom_vjp, nondiff_argnums=(2,))
        def potential_fn(position, box, pairs, params):
            position_t = j2t(position)
            box_t = j2t(box)
            params_t = j2t_pytree(params)
            results, model = potential_torch_kernel(position_t, box_t, None, params_t)
            return t2j(result['pred_energy'])

        def potential_fwd(positions, box, pairs, params):
            # gradient of positions and box will be computed internally and returned
            # by force an virial
            position_t = j2t(positions).detach()
            box_t = j2t(box).detach()
            position_t.requires_grad_(False)
            box_t.requires_grad_(False)
            params_t = j2t_pytree(params)
            result, model = potential_torch_kernel(position_t, box_t, None, params_t)
            model.zero_grad()
            result['pred_energy'].backward()

            inputs = {'pos': positions,
                      'box': box,
                      'params': params
                    }
            energy = t2j(result['pred_energy'])
            dE_dp = jax.tree.map(lambda x: jnp.zeros(x.shape), inputs['params'])
            # read parameter gradient from the model
            for name, param in model.named_parameters():
                dE_dp[self.name][name] = t2j_extract_grad(param)
            return energy, (t2j_pytree(result), inputs, dE_dp)

        def potential_bwd(pairs, res, g):
            preds = res[0]
            inputs = res[1]
            dE_dp = res[2]
            force = preds['pred_forces']
            # virial is in kJ/mol
            virial = preds['pred_virial']
            pos = inputs['pos'] # in nm
            box = inputs['box'] # in nm
            box_inv = jnp.linalg.inv(box) # in nm-1
            # force in kJ/mol/nm, positions in nm
            dE_dB = box_inv.T@(pos.T@force - virial)
            dE_dr = -force
            # unit conversion from eV/A to kJ/mol/nm
            return dE_dr*g, dE_dB, jax.tree.map(lambda x: x*g, dE_dp)

        potential_fn.defvjp(potential_fwd, potential_bwd)

        return potential_fn


    def write_to(self, params, ckpt_file, state_dict_file):
        if self.name in params:
            self.params = params[self.name]
        else:
            self.params = params
        self.params_t = j2t_pytree(self.params)
        state_dict = copy.deepcopy(self.params_t)
        for k in self.params_noopt:
            state_dict[k] = self.params_noopt[k]
        # save state dictiontary file
        torch.save(state_dict, state_dict_file)
        # save the full model checkpoint
        self.model.load_state_dict(state_dict)
        self.save_method(self.model, ckpt_file)
        return

    def overwrite(self, params):
        # do not use xml to handle ML potentials
        # for ML potentials, xml only documents param file path
        # so for ML potentials, overwrite function overwrites the file directly
        self.write_to(params, self.ckpt_file, self.state_dict_file)
        return


    def getJaxPotential(self):
        return self._jaxPotential

_DMFFGenerators["CustomTorchForce"] = CustomTorchGenerator
