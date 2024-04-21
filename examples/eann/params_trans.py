#!/usr/bin/env python
import torch
import jax.numpy as jnp
import pickle
from collections import OrderedDict

def transfer_params_to_jax(input_path, output_path):
    state_dict = torch.load(input_path, map_location='cpu')
    new_state_dict = OrderedDict()

    for k, v in state_dict['eannparam'].items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    param = new_state_dict

    for key in param:
        param[key] = jnp.array(param[key])

    params = {}
    element_names = ['H', 'C', 'O']
    for i in range(7):
        if i == 2 or i == 5:
            alpha_stack = jnp.stack([param[f'nnmod.elemental_nets.{elem}.{i}.alpha'] for elem in element_names])
            beta_stack = jnp.stack([param[f'nnmod.elemental_nets.{elem}.{i}.beta'] for elem in element_names])
            params[f'w.{i}'] = alpha_stack
            params[f'b.{i}'] = beta_stack
        else:
            w_stack = jnp.stack([param[f'nnmod.elemental_nets.{elem}.{i}.weight'] for elem in element_names])
            params[f'w.{i}'] = w_stack

            b_stack = jnp.stack([param[f'nnmod.elemental_nets.{elem}.{i}.bias'] for elem in element_names])
            params[f'b.{i}'] = b_stack

    params['density.params'] = param['density.params']
    params['density.rs'] = param['density.rs']
    params['density.inta'] = param['density.inta']
    params['nnmod.initpot'] = param['nnmod.initpot']

    with open(output_path, 'wb') as f:
        pickle.dump(params, f)

# Usage:
input_path = "EANN.pth"
output_path = "eann_model.pickle"
transfer_params_to_jax(input_path, output_path)
