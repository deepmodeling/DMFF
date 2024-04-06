from functools import partial

import jax.numpy as jnp
from .pairwise import (distribute_dispcoeff, distribute_scalar,
                                distribute_v3)
from .pme import setup_ewald_parameters
from .recip import Ck_6, Ck_8, Ck_10, generate_pme_recip
from .spatial import pbc_shift
from ..utils import jit_condition, pair_buffer_scales, regularize_pairs
from jax import value_and_grad, vmap


class ADMPDispPmeForce:
    '''
    This is a convenient wrapper for dispersion PME calculations
    It wrapps all the environment parameters of multipolar PME calculation
    The so called "environment paramters" means parameters that do not need to be differentiable
    '''

    def __init__(self, box, rc, ethresh, pmax, lpme=True):

        self.rc = rc
        self.ethresh = ethresh
        self.pmax = pmax
        # Need a different function for dispersion ??? Need tests
        self.lpme = lpme
        if lpme:
            kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
            self.kappa = kappa
            self.K1 = K1
            self.K2 = K2
            self.K3 = K3
        else:
            self.kappa = 0.0
            self.K1 = 0
            self.K2 = 0
            self.K3 = 0
        self.pme_order = 6
        # setup calculators
        self.refresh_calculators()
        return


    def generate_get_energy(self):
        def get_energy(positions, box, pairs, c_list, mScales):
            return energy_disp_pme(positions, box, pairs, 
                                  c_list, mScales,
                                  self.kappa, self.K1, self.K2, self.K3, self.pmax,
                                  self.d6_recip, self.d8_recip, self.d10_recip, lpme=self.lpme)
        return get_energy

    
    def update_env(self, attr, val):
        '''
        Update the environment of the calculator
        '''
        setattr(self, attr, val)
        self.refresh_calculators()


    def refresh_calculators(self):
        '''
        refresh the energy and force calculator according to the current environment
        '''
        self.d6_recip = generate_pme_recip(Ck_6, self.kappa, True, self.pme_order, self.K1, self.K2, self.K3, 0)
        if self.pmax >= 8:
            self.d8_recip = generate_pme_recip(Ck_8, self.kappa, True, self.pme_order, self.K1, self.K2, self.K3, 0)
        else:
            self.d8_recip = None
        if self.pmax >= 10:
            self.d10_recip = generate_pme_recip(Ck_10, self.kappa, True, self.pme_order, self.K1, self.K2, self.K3, 0)
        else:
            self.d10_recip = None
        # create the energy calculator according to PME environment
        self.get_energy = self.generate_get_energy()
        self.get_forces = value_and_grad(self.get_energy)
        return


def energy_disp_pme(positions, box, pairs,
        c_list, mScales,
        kappa, K1, K2, K3, pmax, 
        recip_fn6, recip_fn8, recip_fn10, lpme=True):
    '''
    Top level wrapper for dispersion pme

    Input:
        positions:
            Na * 3: positions
        box:
            3 * 3: box, axes arranged in row
        pairs:
            Np * 3: interacting pair indices and topology distance
        c_list:
            Na * (pmax-4)/2: atomic dispersion coefficients
        mScales:
            (Nexcl,): permanent multipole-multipole interaction exclusion scalings: 1-2, 1-3 ...
        covalent_map:
            Na * Na: topological distances between atoms, if i, j are topologically distant, then covalent_map[i, j] == 0
        disp_pme_recip_fn:
            function: the reciprocal calculator, see recip.py
        kappa:
            float: kappa in A^-1
        K1, K2, K3:
            int: max K for reciprocal calculations
        pmax:
            int array: maximal exponents (p) to compute, e.g., (6, 8, 10)
        lpme:
            bool: whether do pme or not, useful when doing cluster calculations

    Output:
        energy: total dispersion pme energy
    '''

    if lpme is False:
        kappa = 0

    ene_real = disp_pme_real(positions, box, pairs, c_list, mScales, kappa, pmax)

    if lpme:
        ene_recip = recip_fn6(positions, box, c_list[:, 0, jnp.newaxis])
        if pmax >= 8:
            ene_recip += recip_fn8(positions, box, c_list[:, 1, jnp.newaxis])
        if pmax >= 10:
            ene_recip += recip_fn10(positions, box, c_list[:, 2, jnp.newaxis])
        ene_self = disp_pme_self(c_list, kappa, pmax)
        return ene_real + ene_recip + ene_self

    else:
        return ene_real


def disp_pme_real(positions, box, pairs, 
        c_list, 
        mScales, 
        kappa, pmax):
    '''
    This function calculates the dispersion real space energy
    It expands the atomic parameters to pairwise parameters

    Input:
        positions:
            Na * 3: positions
        box:
            3 * 3: box, axes arranged in row
        pairs:
            Np * 3: interacting pair indices and topology distance
        c_list:
            Na * (pmax-4)/2: atomic dispersion coefficients
        mScales:
            (Nexcl,): permanent multipole-multipole interaction exclusion scalings: 1-2, 1-3 ...
        covalent_map:
            Na * Na: topological distances between atoms, if i, j are topologically distant, then covalent_map[i, j] == 0
        kappa:
            float: kappa in A^-1
        pmax:
            int array: maximal exponents (p) to compute, e.g., (6, 8, 10)

    Output:
        ene: dispersion pme realspace energy
    '''

    # expand pairwise parameters
    # pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))

    box_inv = jnp.linalg.inv(box + jnp.eye(3) * 1e-36)

    ri = distribute_v3(positions, pairs[:, 0])
    rj = distribute_v3(positions, pairs[:, 1])
    nbonds = pairs[:, 2]
    mscales = distribute_scalar(mScales, nbonds-1)

    buffer_scales = pair_buffer_scales(pairs[:, :2])
    mscales = mscales * buffer_scales

    ci = distribute_dispcoeff(c_list, pairs[:, 0])
    cj = distribute_dispcoeff(c_list, pairs[:, 1])

    ene_real = jnp.sum(
            disp_pme_real_kernel(ri, rj, ci, cj, box, box_inv, mscales, kappa, pmax)
            * buffer_scales
            )

    return jnp.sum(ene_real)


@partial(vmap, in_axes=(0, 0, 0, 0, None, None, 0, None, None), out_axes=(0))
@jit_condition(static_argnums=(8))
def disp_pme_real_kernel(ri, rj, ci, cj, box, box_inv, mscales, kappa, pmax):
    '''
    The kernel to calculate the realspace dispersion energy
    
    Inputs:
        ri: 
            Np * 3: position i
        rj:
            Np * 3: position j
        ci: 
            Np * (pmax-4)/2: dispersion coeffs of i, c6, c8, c10 etc
        cj:
            Np * (pmax-4)/2: dispersion coeffs of j, c6, c8, c10 etc
        kappa:
            float: kappa
        pmax:
            int: largest p in 1/r^p, assume starting from 6 with increment of 2

    Output:
        energy: 
            float: the dispersion pme energy
    '''
    dr = ri - rj
    dr = pbc_shift(dr, box, box_inv)
    dr2 = jnp.dot(dr, dr)

    x2 = kappa * kappa * dr2
    g = g_p(x2, pmax)
    dr6 = dr2 * dr2 * dr2
    ene = (mscales + g[0] - 1) * ci[0] * cj[0] / dr6
    if pmax >= 8:
        dr8 = dr6 * dr2
        ene += (mscales + g[1] - 1) * ci[1] * cj[1] / dr8
    if pmax >= 10:
        dr10 = dr8 * dr2
        ene += (mscales + g[2] - 1) * ci[2] * cj[2] / dr10
    return ene


def g_p(x2, pmax):
    '''
    Compute the g(x, p) function

    Inputs:
        x:
            float: the input variable
        pmax:
            int: the maximal powers of dispersion, here we assume evenly spacing even powers starting from 6
            e.g., (6,), (6, 8) or (6, 8, 10)

    Outputs:
        g:
            (p-4)//2: g(x, p)
    '''

    x4 = x2 * x2
    x8 = x4 * x4
    exp_x2 = jnp.exp(-x2)
    g6 = 1 + x2 + 0.5*x4
    if pmax >= 8:
        g8 = g6 + x4*x2/6
    if pmax >= 10:
        g10 = g8 + x8/24

    if pmax == 6:
        g = jnp.array([g6])
    elif pmax == 8:
        g = jnp.array([g6, g8])
    elif pmax == 10:
        g = jnp.array([g6, g8, g10])

    return g * exp_x2


@jit_condition(static_argnums=(2))
def disp_pme_self(c_list, kappa, pmax):
    '''
    This function calculates the dispersion self energy

    Inputs:
        c_list:
            Na * 3: dispersion susceptibilities C_6, C_8, C_10
        kappa:
            float: kappa used in dispersion

    Output:
        ene_self:
            float: the self energy
    '''
    E_6 = -kappa**6/12 * jnp.sum(c_list[:, 0]**2)
    if pmax >= 8:
        E_8 = -kappa**8/48 * jnp.sum(c_list[:, 1]**2)
    if pmax >= 10:
        E_10 = -kappa**10/240 * jnp.sum(c_list[:, 2]**2)
    E = E_6
    if pmax >= 8:
        E += E_8
    if pmax >= 10:
        E += E_10
    return E


