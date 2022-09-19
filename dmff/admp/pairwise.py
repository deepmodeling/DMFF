from functools import partial

import jax.numpy as jnp
from dmff.admp.spatial import v_pbc_shift
from dmff.utils import jit_condition, pair_buffer_scales, regularize_pairs
from jax import vmap

DIELECTRIC = 1389.35455846

# for debug
# from jax_md import partition, space
# from admp.parser import *
# from admp.multipole import *
# from jax import grad, value_and_grad
# from admp.pme import *

# jitted and vmapped parameter distributors
# all three look identical, but they assume different input shapes
# you should use different functions for different inputs, to avoid recompiling
@partial(vmap, in_axes=(None, 0), out_axes=(0))
@jit_condition(static_argnums=())
def distribute_scalar(params, index):
    return params[index]


@partial(vmap, in_axes=(None, 0), out_axes=(0))
@jit_condition(static_argnums=())
def distribute_v3(pos, index):
    return pos[index]


@partial(vmap, in_axes=(None, 0), out_axes=(0))
@jit_condition(static_argnums=())
def distribute_multipoles(multipoles, index):
    return multipoles[index]


@partial(vmap, in_axes=(None, 0), out_axes=(0))
@jit_condition(static_argnums=())
def distribute_dispcoeff(c_list, index):
    return c_list[index]

@jit_condition(static_argnums=())
def distribute_matrix(multipoles,index1,index2):
    return multipoles[index1,index2]

def generate_pairwise_interaction(pair_int_kernel, static_args):
    '''
    This is a calculator generator for pairwise interaction 

    Input:
        pair_int_kernel:
            function type (dr, m, p1i, p1j, p2i, p2j) -> energy : the vectorized kernel function, 
            dr is the distance, m is the topological scaling factor, p1i, p1j, p2i, p2j are pairwise parameters

        static_args:
            dict: a dictionary that stores all static global parameters (such as lmax, kappa, etc)

    Output:
        pair_int:
            function type (positions, box, pairs, mScales, p1, p2, ...) -> energy
            The pair interaction calculator. p1, p2 ... involved atomic parameters, the order should be consistent
            with the order in kernel
    '''

    def pair_int(positions, box, pairs, mScales, *atomic_params):
        # pairs = regularize_pairs(pairs)
        pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))

        ri = distribute_v3(positions, pairs[:, 0])
        rj = distribute_v3(positions, pairs[:, 1])
        # ri = positions[pairs[:, 0]]
        # rj = positions[pairs[:, 1]]
        nbonds = pairs[:, 3]
        mscales = distribute_scalar(mScales, nbonds-1)

        buffer_scales = pair_buffer_scales(pairs)
        mscales = mscales * buffer_scales
        # mscales = mScales[nbonds-1]
        box_inv = jnp.linalg.inv(box)
        dr = ri - rj
        dr = v_pbc_shift(dr, box, box_inv)
        dr = jnp.linalg.norm(dr, axis=1)

        pair_params = []
        for i, param in enumerate(atomic_params):
            pair_params.append(distribute_scalar(param, pairs[:, 0]))
            pair_params.append(distribute_scalar(param, pairs[:, 1]))
            # pair_params.append(param[pairs[:, 0]])
            # pair_params.append(param[pairs[:, 1]])

        energy = jnp.sum(pair_int_kernel(dr, mscales, *pair_params) * buffer_scales)
        return energy

    return pair_int


@vmap
@jit_condition(static_argnums={})
def TT_damping_qq_c6_kernel(dr, m, ai, aj, bi, bj, qi, qj, ci, cj):
    a = jnp.sqrt(ai * aj)
    b = jnp.sqrt(bi * bj)
    c = ci * cj
    q = qi * qj
    r = dr * 1.889726878 # convert to bohr
    br = b * r
    br2 = br * br
    br3 = br2 * br
    br4 = br3 * br
    br5 = br4 * br
    br6 = br5 * br
    exp_br = jnp.exp(-br)
    f = 2625.5 * a * exp_br \
        + (-2625.5) * exp_br * (1+br) * q / r \
        + exp_br*(1+br+br2/2+br3/6+br4/24+br5/120+br6/720) * c / dr**6

    return f * m


@vmap
@jit_condition(static_argnums={})
def TT_damping_qq_kernel(dr, m, bi, bj, qi, qj):
    b = jnp.sqrt(bi * bj)
    q = qi * qj
    br = b * dr
    exp_br = jnp.exp(-br)
    f = - DIELECTRIC * exp_br * (1+br) * q / dr 
    return f * m


@vmap
@jit_condition(static_argnums=())
def slater_disp_damping_kernel(dr, m, bi, bj, c6i, c6j, c8i, c8j, c10i, c10j):
    r'''
    Slater-ISA type damping for dispersion:
    f(x) = -e^{-x} * \sum_{k} x^k/k!
    x = Br - \frac{2*(Br)^2 + 3Br}{(Br)^2 + 3*Br + 3}
    see jctc 12 3851
    '''
    b = jnp.sqrt(bi * bj)
    c6 = c6i * c6j
    c8 = c8i * c8j
    c10 = c10i * c10j
    br = b * dr
    br2 = br * br
    x = br - (2*br2 + 3*br) / (br2 + 3*br + 3)
    s6 = 1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 + x**6/720
    s8 = s6 + x**7/5040 + x**8/40320
    s10 = s8 + x**9/362880 + x**10/3628800
    exp_x = jnp.exp(-x)
    f6 = exp_x * s6
    f8 = exp_x * s8
    f10 = exp_x * s10
    return (f6*c6/dr**6 + f8*c8/dr**8 + f10*c10/dr**10) * m


@vmap
@jit_condition(static_argnums=())
def slater_sr_kernel(dr, m, ai, aj, bi, bj):
    '''
    Slater-ISA type short range terms
    see jctc 12 3851
    '''
    b = jnp.sqrt(bi * bj)
    a = ai * aj
    br = b * dr
    br2 = br * br
    P = 1/3 * br2 + br + 1 
    return a * P * jnp.exp(-br) * m

