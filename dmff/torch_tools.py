#!/usr/bin/env python
import numpy as np
import jax.numpy as jnp
import jax
import torch
from torch2jax import j2t, t2j
from functools import partial


def t2j_element(e):
    if torch.is_tensor(e):
        return t2j(e)
    else:
        return e

def j2t_element(e):
    if isinstance(e, jax.numpy.ndarray):
        e = j2t(e)
        e.requires_grad = True
        return e
    else:
        return e

def t2j_extract_grad(v):
    # extract the gradient of a pytree composed by torch tensors
    def t2j_extract_grad_element(x):
        if torch.is_tensor(x) and x.grad is not None:
            return t2j(x.grad)
        else:
            return t2j(torch.zeros(x.shape))
    return jax.tree.map(t2j_extract_grad_element, v)


def t2j_pytree(v):
    return jax.tree.map(t2j_element, v)


def j2t_pytree(v):
    return jax.tree.map(j2t_element, v)


def wrap_torch_potential_kernel(potential_t):

    @partial(jax.custom_jvp,  nondiff_argnums=(2,))
    def potential(positions, box, pairs, params):
        res = potential_t(j2t_pytree(positions), \
                          j2t_pytree(box), \
                          np.array(pairs), \
                          j2t_pytree(params))
        return res

    @potential.defjvp
    def potential_jvp(pairs, primals, tangents):
        positions, box, params = primals
        dpositions, dbox, dparams = tangents
        # convert inputs to torch
        positions_t = j2t_pytree(positions)
        box_t = j2t_pytree(box)
        params_t = j2t_pytree(params)
        # do fwd and bwd in torch
        primal_out_torch = potential_t(positions_t, box_t, np.array(pairs), params_t)
        primal_out_torch.backward()
        # read gradient in torch
        g_positions = t2j_extract_grad(positions_t)
        g_box = t2j_extract_grad(box_t)
        g_params = t2j_extract_grad(params_t)
        # prepare output
        primal_out = t2j(primal_out_torch)
        tangent_out = jnp.sum(g_positions * dpositions) + jnp.sum(g_box * box)
        tangents_leaves = jax.tree.leaves(dparams)
        grad_leaves = jax.tree.leaves(g_params)
        for x, y in zip(tangents_leaves, grad_leaves):
            tangent_out += jnp.sum(x * y)
        return primal_out, tangent_out

    return potential


