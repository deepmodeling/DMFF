import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, vmap, jit
from jax.scipy.special import erf
from dmff.settings import DO_JIT
from dmff.admp.settings import POL_CONV, MAX_N_POL
from dmff.utils import jit_condition
from dmff.admp.multipole import C1_c2h, convert_cart2harm
from dmff.admp.multipole import rot_ind_global2local, rot_global2local, rot_local2global
from dmff.admp.spatial import v_pbc_shift, generate_construct_local_frames, build_quasi_internal
from dmff.admp.pairwise import distribute_scalar, distribute_v3, distribute_multipoles
from functools import partial

DIELECTRIC = 1389.35455846
DEFAULT_THOLE_WIDTH = 0.3

from dmff.admp.recip import generate_pme_recip, Ck_1

# for debugging use only
# from jax_md import partition, space
# from admp.parser import *

# from jax.config import config
# config.update("jax_enable_x64", True)

# Functions that are related to electrostatic pme

class ADMPPmeForce:
    '''
    This is a convenient wrapper for multipolar PME calculations
    It wrapps all the environment parameters of multipolar PME calculation
    The so called "environment paramters" means parameters that do not need to be differentiable
    '''

    def __init__(self, box, axis_type, axis_indices, covalent_map, rc, ethresh, lmax, lpol=False):
        self.axis_type = axis_type
        self.axis_indices = axis_indices
        self.rc = rc
        self.ethresh = ethresh
        self.lmax = int(lmax)  # jichen: type checking
        kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
        self.kappa = kappa
        self.K1 = K1
        self.K2 = K2
        self.K3 = K3
        self.pme_order = 6
        self.covalent_map = covalent_map
        self.lpol = lpol
        self.n_atoms = int(covalent_map.shape[0]) # len(axis_type)

        # setup calculators
        self.refresh_calculators()
        return


    def generate_get_energy(self):
        # if the force field is not polarizable
        if not self.lpol:
            def get_energy(positions, box, pairs, Q_local, mScales):
                return energy_pme(positions, box, pairs,
                                 Q_local, None, None, None,
                                 mScales, None, None, self.covalent_map,
                                 self.construct_local_frames, self.pme_recip,
                                 self.kappa, self.K1, self.K2, self.K3, self.lmax, False)
            return get_energy
        else:
            # this is the bare energy calculator, with Uind as explicit input
            def energy_fn(positions, box, pairs, Q_local, Uind_global, pol, tholes, mScales, pScales, dScales):
                return energy_pme(positions, box, pairs,
                                 Q_local, Uind_global, pol, tholes,
                                 mScales, pScales, dScales, self.covalent_map,
                                 self.construct_local_frames, self.pme_recip,
                                 self.kappa, self.K1, self.K2, self.K3, self.lmax, True)
            self.energy_fn = energy_fn
            self.grad_U_fn = grad(self.energy_fn, argnums=(4))
            self.grad_pos_fn = grad(self.energy_fn, argnums=(0))
            self.U_ind = jnp.zeros((self.n_atoms, 3))
            # this is the wrapper that include a Uind optimizer
            def get_energy(positions, box, pairs, Q_local, pol, tholes, mScales, pScales, dScales, U_init=self.U_ind):
                self.U_ind, self.lconverg, self.n_cycle = self.optimize_Uind(positions, box, pairs, Q_local, pol, tholes, mScales, pScales, dScales, U_init=U_init)
                # here we rely on Feynman-Hellman theorem, drop the term dV/dU*dU/dr !
                # self.U_ind = jax.lax.stop_gradient(U_ind)
                return self.energy_fn(positions, box, pairs, Q_local, self.U_ind, pol, tholes, mScales, pScales, dScales)
            return get_energy


    def update_env(self, attr, val):
        '''
        Update the environment of the calculator
        '''
        setattr(self, attr, val)
        self.refresh_calculators()


    def refresh_calculators(self):
        '''
        refresh the energy and force calculators according to the current environment
        '''
        if self.lmax > 0:
            self.construct_local_frames = generate_construct_local_frames(self.axis_type, self.axis_indices)
        else:
            self.construct_local_frames = None
        self.pme_recip = generate_pme_recip(Ck_1, self.kappa, False, self.pme_order, self.K1, self.K2, self.K3, self.lmax)
        # generate the force calculator
        self.get_energy = self.generate_get_energy()
        self.get_forces = value_and_grad(self.get_energy)
        return

    def optimize_Uind(self, positions, box, pairs, Q_local, pol, tholes, mScales, pScales, dScales, U_init=None, maxiter=MAX_N_POL, thresh=POL_CONV):
        '''
        This function converges the induced dipole
        Note that we cut all the gradient chain passing through this function as we assume Feynman-Hellman theorem
        Gradients related to Uind should be dropped
        '''
        # Do not track gradient in Uind optimization
        positions = jax.lax.stop_gradient(positions)
        box = jax.lax.stop_gradient(box)
        Q_local = jax.lax.stop_gradient(Q_local)
        pol = jax.lax.stop_gradient(pol)
        tholes = jax.lax.stop_gradient(tholes)
        mScales = jax.lax.stop_gradient(mScales)
        pScales = jax.lax.stop_gradient(pScales)
        dScales = jax.lax.stop_gradient(dScales)
        if U_init is None:
            U = jnp.zeros((self.n_atoms, 3))
        else:
            U = U_init
        site_filter = (pol>0.001) # focus on the actual polarizable sites

        for i in range(maxiter):
            field = self.grad_U_fn(positions, box, pairs, Q_local, U, pol, tholes, mScales, pScales, dScales)
            E = self.energy_fn(positions, box, pairs, Q_local, U, pol, tholes, mScales, pScales, dScales)
            # print(i, E, jnp.max(jnp.abs(field[site_filter])))
            if jnp.max(jnp.abs(field[site_filter])) < thresh:
                break
            U = U - field * pol[:, jnp.newaxis] / DIELECTRIC
        if i == maxiter-1:
            flag = False
        else: # converged
            flag = True
        return U, flag, i


def setup_ewald_parameters(rc, ethresh, box):
    '''
    Given the cutoff distance, and the required precision, determine the parameters used in
    Ewald sum, including: kappa, K1, K2, and K3.
    The algorithm is exactly the same as OpenMM, see: 
    http://docs.openmm.org/latest/userguide/theory.html

    Input:
        rc:
            float, the cutoff distance
        ethresh:
            float, required energy precision
        box:
            3*3 matrix, box size, a, b, c arranged in rows

    Output:
        kappa:
            float, the attenuation factor
        K1, K2, K3:
            integers, sizes of the k-points mesh
    '''
    kappa = jnp.sqrt(-jnp.log(2*ethresh))/rc
    K1 = jnp.ceil(2 * kappa * box[0, 0] / 3 / ethresh**0.2)
    K2 = jnp.ceil(2 * kappa * box[1, 1] / 3 / ethresh**0.2)
    K3 = jnp.ceil(2 * kappa * box[2, 2] / 3 / ethresh**0.2)

    return kappa, int(K1), int(K2), int(K3)


# @jit_condition(static_argnums=())
def energy_pme(positions, box, pairs,
        Q_local, Uind_global, pol, tholes,
        mScales, pScales, dScales, covalent_map, 
        construct_local_frame_fn, pme_recip_fn, kappa, K1, K2, K3, lmax, lpol):
    '''
    This is the top-level wrapper for multipole PME

    Input:
        positions:
            Na * 3: positions
        box: 
            3 * 3: box
        Q_local: 
            Na * (lmax+1)^2: harmonic multipoles of each site in local frame
        Uind_global:
            Na * 3: the induced dipole moment, in GLOBAL CARTESIAN!
        pol: 
            (Na,) float: the polarizability of each site, unit in A**3
        tholes: 
            (Na,) float: the thole damping widths for each atom, it's dimensionless, default is 8 according to MPID paper
        mScales, pScale, dScale:
            (Nexcl,): multipole-multipole interaction exclusion scalings: 1-2, 1-3 ...
            for permanent-permanent, permanent-induced, induced-induced interactions
        pairs:
            Np * 2: interacting pair indices
        covalent_map:
            Na * Na: topological distances between atoms, if i, j are topologically distant, then covalent_map[i, j] == 0
        construct_local_frame_fn:
            function: local frame constructors, from generate_local_frame_constructor
        pme_recip:
            function: see recip.py, a reciprocal space calculator
        kappa:
            float: kappa in A^-1
        K1, K2, K3:
            int: max K for reciprocal calculations
        lmax:
            int: maximum L
        bool:
            int: if polarizable or not? if yes, 1, otherwise 0

    Output:
        energy: total pme energy
    '''
    # if doing a multipolar calculation
    if lmax > 0:
        local_frames = construct_local_frame_fn(positions, box)
        Q_global = rot_local2global(Q_local, local_frames, lmax)
    else:
        if lpol:
            # if fixed multipole only contains charge, and it's polarizable, then expand Q matrix
            dips = jnp.zeros((Q_local.shape[0], 3))
            Q_global = jnp.hstack((Q_global, dips))
            lmax = 1
        else:
            Q_global = Q_local

    # note we assume when lpol is True, lmax should be >= 1
    if lpol:
        # convert Uind to global harmonics, in accord with Q_global
        U_ind = C1_c2h.dot(Uind_global.T).T
        Q_global_tot = Q_global.at[:, 1:4].add(U_ind)
    else:
        Q_global_tot = Q_global

    if lpol:
        ene_real = pme_real(positions, box, pairs, Q_global, U_ind, pol, tholes, 
                           mScales, pScales, dScales, covalent_map, kappa, lmax, True)
    else:
        ene_real = pme_real(positions, box, pairs, Q_global, None, None, None,
                           mScales, None, None, covalent_map, kappa, lmax, False)

    ene_recip = pme_recip_fn(positions, box, Q_global_tot)

    ene_self = pme_self(Q_global_tot, kappa, lmax)

    if lpol:
        ene_self += pol_penalty(U_ind, pol)

    return ene_real + ene_recip + ene_self


# @partial(vmap, in_axes=(0, 0, None, None), out_axes=0)
@jit_condition(static_argnums=(3))
def calc_e_perm(dr, mscales, kappa, lmax=2):

    '''
    This function calculates the ePermCoefs at once
    ePermCoefs is basically the interaction tensor between permanent multipole components
    Everything should be done in the so called quasi-internal (qi) frame
    Energy = \sum_ij qiQI * ePermCoeff_ij * qiQJ

    Inputs:
        dr: 
            float: distance between one pair of particles
        mscales:
            float: scaling factor between permanent - permanent multipole interactions, for each pair
        kappa:
            float: \kappa in PME, unit in A^-1
        lmax:
            int: max L

    Output:
        cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2:
            n * 1 array: ePermCoefs
    '''

    # be aware of unit and dimension !!
    rInv = 1 / dr
    rInvVec = jnp.array([DIELECTRIC*(rInv**i) for i in range(0, 9)])
    alphaRVec = jnp.array([(kappa*dr)**i for i in range(0, 10)])
    X = 2 * jnp.exp(-alphaRVec[2]) / jnp.sqrt(np.pi)
    tmp = jnp.array(alphaRVec[1])
    doubleFactorial = 1
    facCount = 1
    erfAlphaR = erf(alphaRVec[1])
        
    # bVec = jnp.empty((6, len(erfAlphaR)))
    bVec = jnp.empty(6)

    bVec = bVec.at[1].set(-erfAlphaR)
    for i in range(2, 6):
        bVec = bVec.at[i].set((bVec[i-1]+(tmp*X/doubleFactorial)))
        facCount += 2
        doubleFactorial *= facCount
        tmp *= 2 * alphaRVec[2]    
    
    # C-C: 1
    cc = rInvVec[1] * (mscales + bVec[2] - alphaRVec[1]*X)
    if lmax >= 1:
        # C-D
        cd = rInvVec[2] * (mscales + bVec[2])
        # D-D: 2
        dd_m0 = -2/3 * rInvVec[3] * (3*(mscales + bVec[3]) + alphaRVec[3]*X)
        dd_m1 = rInvVec[3] * (mscales + bVec[3] - (2/3)*alphaRVec[3]*X)
    else:
        cd = 0
        dd_m0 = 0
        dd_m1 = 0

    if lmax >= 2:
        ## C-Q: 1
        cq = (mscales + bVec[3]) * rInvVec[3]
        ## D-Q: 2
        dq_m0 = rInvVec[4] * (3* (mscales + bVec[3]) + (4/3) * alphaRVec[5]*X)
        dq_m1 = -jnp.sqrt(3) * rInvVec[4] * (mscales + bVec[3])
        ## Q-Q
        qq_m0 = rInvVec[5] * (6* (mscales + bVec[4]) + (4/45)* (-3 + 10*alphaRVec[2]) * alphaRVec[5]*X)
        qq_m1 = - (4/15) * rInvVec[5] * (15*(mscales+bVec[4]) + alphaRVec[5]*X)
        qq_m2 = rInvVec[5] * (mscales + bVec[4] - (4/15)*alphaRVec[5]*X)
    else:
        cq = 0
        dq_m0 = 0
        dq_m1 = 0
        qq_m0 = 0
        qq_m1 = 0
        qq_m1 = 0
        qq_m2 = 0

    return cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2


@jit_condition(static_argnums=())
def switch_val(x, x0, sigma, y0, y1):
    '''
    This is a Fermi function switches between y0 and y1, according to the value of x
    y = y0 when x << x0
    y = y1 when x >> x1
    sigma control sthe switch width
    '''
    u = (x-x0) / sigma
    w0 = 1 / (jnp.exp(u) + 1)
    w1 = 1 - w0
    return w0*y0 + w1*y1


def gen_trim_val_0(thresh):
    '''
    Trim the value at zero point to avoid singularity
    '''
    def trim_val_0(x):
        return jnp.piecewise(x, [x<thresh, x>=thresh], [lambda x: jnp.array(thresh), lambda x: x])
    if DO_JIT:
        return jit(trim_val_0)
    else:
        return trim_val_0

trim_val_0 = gen_trim_val_0(1e-8)


def gen_trim_val_infty(thresh):
    '''
    Trime the value at infinity to avoid divergence
    '''
    def trim_val_infty(x):
        return jnp.piecewise(x, [x<thresh, x>=thresh], [lambda x: x, lambda x: jnp.array(thresh)])
    if DO_JIT:
        return jit(trim_val_infty)
    else:
        return trim_val_infty

trim_val_infty = gen_trim_val_infty(1e8)


@jit_condition(static_argnums=(7))
def calc_e_ind(dr, thole1, thole2, dmp, pscales, dscales, kappa, lmax=2):

    '''
    This function calculates the eUindCoefs at once
       ## compute the Thole damping factors for energies
     eUindCoefs is basically the interaction tensor between permanent multipole components and induced dipoles
    Everything should be done in the so called quasi-internal (qi) frame
    

    Inputs:
        dr: 
            float: distance between one pair of particles
        dmp
            float: damping factors between one pair of particles
        mscales:
            float: scaling factor between permanent - permanent multipole interactions, for each pair
        pscales:
            float: scaling factor between permanent - induced multipole interactions, for each pair
        au:
            float: for damping factors
        kappa:
            float: \kappa in PME, unit in A^-1
        lmax:
            int: max L

    Output:
        Interaction tensors components
    '''
    ## switch function

    # a = 1/(jnp.exp((pscales-0.001)*10000)+1) * (thole1 + thole2) + 8/(jnp.exp((-pscales+0.01)*10000)+1)
    a = switch_val(pscales, 1e-3, 1e-5, DEFAULT_THOLE_WIDTH, thole1+thole2)

    dmp = trim_val_0(dmp)
    u = trim_val_infty(dr/dmp)

    ## au <= 50 aupi = au ;au> 50 aupi = 50
    au = a * u
    expau = jnp.piecewise(au, [au<50, au>=50], [lambda au: jnp.exp(-au), lambda au: jnp.array(0)])

    ## compute the Thole damping factors for energies
    au2 = trim_val_infty(au*au)
    au3 = trim_val_infty(au2*au)
    au4 = trim_val_infty(au3*au)
    au5 = trim_val_infty(au4*au)
    au6 = trim_val_infty(au5*au)

    ##  Thole damping factors for energies
    thole_c   = 1.0 - expau*(1.0 + au + 0.5*au2)
    thole_d0  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/4.0)
    thole_d1  = 1.0 - expau*(1.0 + au + 0.5*au2)
    thole_q0  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/18.0)
    thole_q1  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0)
    # copied from calc_e_perm
    # be aware of unit and dimension !!
    rInv = 1 / dr
    rInvVec = jnp.array([DIELECTRIC*(rInv**i) for i in range(0, 9)])
    alphaRVec = jnp.array([(kappa*dr)**i for i in range(0, 10)])
    X = 2 * jnp.exp(-alphaRVec[2]) / jnp.sqrt(np.pi)
    tmp = jnp.array(alphaRVec[1])
    doubleFactorial = 1
    facCount = 1
    erfAlphaR = erf(alphaRVec[1])

    #bVec = jnp.empty((6, len(erfAlphaR)))
    bVec = jnp.empty(6)

    bVec = bVec.at[1].set(-erfAlphaR)
    for i in range(2, 6):
        bVec = bVec.at[i].set((bVec[i-1]+(tmp*X/doubleFactorial)))
        facCount += 2
        doubleFactorial *= facCount
        tmp *= 2 * alphaRVec[2]

    ## C-Uind 
    cud = 2.0*rInvVec[2]*(pscales*thole_c + bVec[2])
    if lmax >= 1:
        ##  D-Uind terms 
        dud_m0 = -2.0*2.0/3.0*rInvVec[3]*(3.0*(pscales*thole_d0 + bVec[3]) + alphaRVec[3]*X)
        dud_m1 = 2.0*rInvVec[3]*(pscales*thole_d1 + bVec[3] - 2.0/3.0*alphaRVec[3]*X)
    else:
        dud_m0 = 0.0
        dud_m1 = 0.0

    if lmax >= 2:
        ## Uind-Q
        udq_m0 = 2.0*rInvVec[4]*(3.0*(pscales*thole_q0 + bVec[3]) + 4/3*alphaRVec[5]*X)
        udq_m1 =  -2.0*jnp.sqrt(3)*rInvVec[4]*(pscales*thole_q1 + bVec[3])
    else:
        udq_m0 = 0.0
        udq_m1 = 0.0
    ## Uind-Uind
    uscales = 1
    udud_m0 = -2.0/3.0*rInvVec[3]*(3.0*(uscales*thole_d0 + bVec[3]) + alphaRVec[3]*X)
    udud_m1 = rInvVec[3]*(uscales*thole_d1 + bVec[3] - 2.0/3.0*alphaRVec[3]*X)
    return cud, dud_m0, dud_m1, udq_m0, udq_m1, udud_m0, udud_m1



@partial(vmap, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None), out_axes=0)
@jit_condition(static_argnums=(12, 13))
def pme_real_kernel(dr, qiQI, qiQJ, qiUindI, qiUindJ, thole1, thole2, dmp, mscales, pscales, dscales, kappa, lmax=2, lpol=False):
    '''
    This is the heavy-lifting kernel function to compute the realspace multipolar PME 
    Vectorized over interacting pairs

    Input:
        dr: 
            float, the interatomic distances, (np) array if vectorized
        qiQI:
            [(lmax+1)^2] float array, the harmonic multipoles of site i in quasi-internal frame
        qiQJ:
            [(lmax+1)^2] float array, the harmonic multipoles of site j in quasi-internal frame
        qiUindI
            (3,) float array, the harmonic dipoles of site i in QI frame
        qiUindJ
            (3,) float array, the harmonic dipoles of site j in QI frame
        thole1
            float: thole damping coeff of site i
        thole2
            float: thole damping coeff of site j
        dmp:
            float: (pol1 * pol2)**1/6, distance rescaling params used in thole damping
        mscale:
            float, scaling factor between interacting sites (permanent-permanent)
        pscale:
            float, scaling factor between perm-ind interaction
        dscale:
            float, scaling factor between ind-ind interaction
        kappa:
            float, kappa in unit A^1
        lmax:
            int, maximum angular momentum
        lpol:
            bool, doing polarization?

    Output:
        energy: 
            float, realspace interaction energy between the sites
    '''

    cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2 = calc_e_perm(dr, mscales, kappa, lmax)
    if lpol:
        cud, dud_m0, dud_m1, udq_m0, udq_m1, udud_m0, udud_m1 = calc_e_ind(dr, thole1, thole2, dmp, pscales, dscales, kappa, lmax)

    Vij0 = cc*qiQI[0]
    Vji0 = cc*qiQJ[0]
    # C-Uind
    if lpol: 
        Vij0 -= cud * qiUindI[0]
        Vji0 += cud * qiUindJ[0]

    if lmax >= 1:
        # C-D 
        Vij0 = Vij0 - cd*qiQI[1]
        Vji1 = -cd*qiQJ[0]
        Vij1 = cd*qiQI[0]
        Vji0 = Vji0 + cd*qiQJ[1]
        # D-D m0 
        Vij1 += dd_m0 * qiQI[1]
        Vji1 += dd_m0 * qiQJ[1]    
        # D-D m1 
        Vij2 = dd_m1*qiQI[2]
        Vji2 = dd_m1*qiQJ[2]
        Vij3 = dd_m1*qiQI[3]
        Vji3 = dd_m1*qiQJ[3]
        # D-Uind
        if lpol:
            Vij1 += dud_m0 * qiUindI[0]
            Vji1 += dud_m0 * qiUindJ[0]
            Vij2 += dud_m1 * qiUindI[1]
            Vji2 += dud_m1 * qiUindJ[1]
            Vij3 += dud_m1 * qiUindI[2]
            Vji3 += dud_m1 * qiUindJ[2]

    if lmax >= 2:
        # C-Q
        Vij0 = Vij0 + cq*qiQI[4]
        Vji4 = cq*qiQJ[0]
        Vij4 = cq*qiQI[0]
        Vji0 = Vji0 + cq*qiQJ[4]
        # D-Q m0
        Vij1 += dq_m0*qiQI[4]
        Vji4 += dq_m0*qiQJ[1] 
        # Q-D m0
        Vij4 -= dq_m0*qiQI[1]
        Vji1 -= dq_m0*qiQJ[4]
        # D-Q m1
        Vij2 = Vij2 + dq_m1*qiQI[5]
        Vji5 = dq_m1*qiQJ[2]
        Vij3 += dq_m1*qiQI[6]
        Vji6 = dq_m1*qiQJ[3]
        Vij5 = -(dq_m1*qiQI[2])
        Vji2 += -(dq_m1*qiQJ[5])
        Vij6 = -(dq_m1*qiQI[3])
        Vji3 += -(dq_m1*qiQJ[6])
        # Q-Q m0
        Vij4 += qq_m0*qiQI[4]
        Vji4 += qq_m0*qiQJ[4] 
        # Q-Q m1
        Vij5 += qq_m1*qiQI[5]
        Vji5 += qq_m1*qiQJ[5]
        Vij6 += qq_m1*qiQI[6]
        Vji6 += qq_m1*qiQJ[6]
        # Q-Q m2
        Vij7  = qq_m2*qiQI[7]
        Vji7  = qq_m2*qiQJ[7]
        Vij8  = qq_m2*qiQI[8]
        Vji8  = qq_m2*qiQJ[8]
        # Q-Uind
        if lpol:
            Vji4 += udq_m0*qiUindJ[0]
            Vij4 -= udq_m0*qiUindI[0]
            Vji5 += udq_m1*qiUindJ[1]
            Vji6 += udq_m1*qiUindJ[2]
            Vij5 -= udq_m1*qiUindI[1]
            Vij6 -= udq_m1*qiUindI[2]

    # Uind - Uind
    if lpol:
        Vij1dd = udud_m0 * qiUindI[0]
        Vji1dd = udud_m0 * qiUindJ[0]
        Vij2dd = udud_m1 * qiUindI[1]
        Vji2dd = udud_m1 * qiUindJ[1]
        Vij3dd = udud_m1 * qiUindI[2]
        Vji3dd = udud_m1 * qiUindJ[2]
        Vijdd = jnp.stack(( Vij1dd, Vij2dd, Vij3dd))
        Vjidd = jnp.stack(( Vji1dd, Vji2dd, Vji3dd))

    if lmax == 0:
        Vij = Vij0
        Vji = Vji0
    elif lmax == 1:
        Vij = jnp.stack((Vij0, Vij1, Vij2, Vij3))
        Vji = jnp.stack((Vji0, Vji1, Vji2, Vji3))
    elif lmax == 2:
        Vij = jnp.stack((Vij0, Vij1, Vij2, Vij3, Vij4, Vij5, Vij6, Vij7, Vij8))
        Vji = jnp.stack((Vji0, Vji1, Vji2, Vji3, Vji4, Vji5, Vji6, Vji7, Vji8))
    else:
        print('Error: Lmax must <= 2')

    if lpol:
        return jnp.array(0.5) * (jnp.sum(qiQJ*Vij) + jnp.sum(qiQI*Vji)) + jnp.array(0.5) * (jnp.sum(qiUindJ*Vijdd) + jnp.sum(qiUindI*Vjidd))
    else:
        return jnp.array(0.5) * (jnp.sum(qiQJ*Vij) + jnp.sum(qiQI*Vji))


# @jit_condition(static_argnums=(7))
def pme_real(positions, box, pairs, 
        Q_global, Uind_global, pol, tholes,
        mScales, pScales, dScales, covalent_map, 
        kappa, lmax, lpol):
    '''
    This is the real space PME calculate function
    NOTE: only deals with permanent-permanent multipole interactions
    It expands the pairwise parameters, and then invoke pme_real_kernel
    It seems pointless to jit it:
    1. the heavy-lifting kernel function is jitted and vmapped
    2. len(pairs) keeps changing throughout the simulation, the function would just recompile everytime

    Input:
        positions:
            Na * 3: positions
        box:
            3 * 3: box, axes arranged in row
        pairs:
            Np * 2: interacting pair indices
        Q_global:
            Na * (l+1)**2: harmonics multipoles of each atom, in global frame
        Uind_global:
            Na * 3: harmonic induced dipoles, in global frame
        pol:
            (Na,): polarizabilities
        tholes:
            (Na,): thole damping parameters
        mScales:
            (Nexcl,): permanent multipole-multipole interaction exclusion scalings: 1-2, 1-3 ...
        covalent_map:
            Na * Na: topological distances between atoms, if i, j are topologically distant, then covalent_map[i, j] == 0
        kappa:
            float: kappa in A^-1
        lmax:
            int: maximum L
        lpol:
            Bool: whether do a polarizable calculation?

    Output:
        ene: pme realspace energy
    '''

    # expand pairwise parameters, from atomic parameters
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    box_inv = jnp.linalg.inv(box)
    r1 = distribute_v3(positions, pairs[:, 0])
    r2 = distribute_v3(positions, pairs[:, 1])
    # r1 = positions[pairs[:, 0]]
    # r2 = positions[pairs[:, 1]]
    Q_extendi = distribute_multipoles(Q_global, pairs[:, 0])
    Q_extendj = distribute_multipoles(Q_global, pairs[:, 1])
    # Q_extendi = Q_global[pairs[:, 0]]
    # Q_extendj = Q_global[pairs[:, 1]]
    nbonds = covalent_map[pairs[:, 0], pairs[:, 1]]
    indices = nbonds-1
    mscales = distribute_scalar(mScales, indices)
    # mscales = mScales[nbonds-1]
    if lpol:
        pol1 = distribute_scalar(pol, pairs[:, 0])
        pol2 = distribute_scalar(pol, pairs[:, 1])
        thole1 = distribute_scalar(tholes, pairs[:, 0])
        thole2 = distribute_scalar(tholes, pairs[:, 1])
        Uind_extendi = distribute_v3(Uind_global, pairs[:, 0])
        Uind_extendj = distribute_v3(Uind_global, pairs[:, 1])
        pscales = distribute_scalar(pScales, indices)
        dscales = distribute_scalar(dScales, indices)
        # pol1 = pol[pairs[:,0]]
        # pol2 = pol[pairs[:,1]]
        # thole1 = tholes[pairs[:,0]]
        # thole2 = tholes[pairs[:,1]]
        # Uind_extendi = Uind_global[pairs[:, 0]]
        # Uind_extendj = Uind_global[pairs[:, 1]]
        # pscales = pScales[nbonds-1]
        # dscales = dScales[nbonds-1]
        dmp = get_pair_dmp(pol1, pol2)
    else:
        Uind_extendi = None
        Uind_extendj = None
        pscales = None
        dscales = None
        thole1 = None
        thole2 = None
        dmp = None

    # deals with geometries
    dr = r1 - r2
    dr = v_pbc_shift(dr, box, box_inv)
    norm_dr = jnp.linalg.norm(dr, axis=-1)
    Ri = build_quasi_internal(r1, r2, dr, norm_dr)
    qiQI = rot_global2local(Q_extendi, Ri, lmax)
    qiQJ = rot_global2local(Q_extendj, Ri, lmax)
    if lpol:
        qiUindI = rot_ind_global2local(Uind_extendi, Ri)
        qiUindJ = rot_ind_global2local(Uind_extendj, Ri)
    else:
        qiUindI = None
        qiUindJ = None

    # everything should be pair-specific now
    ene = jnp.sum(pme_real_kernel(norm_dr, qiQI, qiQJ, qiUindI, qiUindJ, thole1, thole2, dmp, mscales, pscales, dscales, kappa, lmax, lpol))

    return ene


@partial(vmap, in_axes=(0, 0), out_axes=(0))
@jit_condition(static_argnums=())
def get_pair_dmp(pol1, pol2):
    return (pol1*pol2) ** (1/6)


@jit_condition(static_argnums=(2))
def pme_self(Q_h, kappa, lmax=2):
    '''
    This function calculates the PME self energy

    Inputs:
        Q:
            Na * (lmax+1)^2: harmonic multipoles, local or global does not matter
        kappa:
            float: kappa used in PME

    Output:
        ene_self:
            float: the self energy
    '''
    n_harms = (lmax + 1) ** 2    
    l_list = np.array([0] + [1,]*3 + [2,]*5)[:n_harms]
    l_fac2 = np.array([1] + [3,]*3 + [15,]*5)[:n_harms]
    factor = kappa/np.sqrt(np.pi) * (2*kappa**2)**l_list / l_fac2
    return - jnp.sum(factor[np.newaxis] * Q_h**2) * DIELECTRIC


@jit_condition(static_argnums=())
def pol_penalty(U_ind, pol):
    '''
    The energy penalty for polarization of each site, currently only supports isotropic polarization:

    Inputs:
        U_ind:
            Na * 3 float: induced dipoles, in isotropic polarization case, cartesian or harmonic does not matter
        pol:
            (Na,) float: polarizability
    '''
    # this is to remove the singularity when pol=0
    pol_pi = trim_val_0(pol)
    # pol_pi = pol/(jnp.exp((-pol+1e-08)*1e10)+1) + 1e-08/(jnp.exp((pol-1e-08)*1e10)+1)
    return jnp.sum(0.5/pol_pi*(U_ind**2).T) * DIELECTRIC


def validation(pdb):
    xml = 'mpidwater.xml'
    pdbinfo = read_pdb(pdb)
    serials = pdbinfo['serials']
    names = pdbinfo['names']
    resNames = pdbinfo['resNames']
    resSeqs = pdbinfo['resSeqs']
    positions = pdbinfo['positions']
    box = pdbinfo['box'] # a, b, c, α, β, γ
    charges = pdbinfo['charges']
    positions = jnp.asarray(positions)
    lx, ly, lz, _, _, _ = box
    box = jnp.eye(3)*jnp.array([lx, ly, lz])

    mScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
    pScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
    dScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])

    rc = 4  # in Angstrom
    ethresh = 1e-4

    n_atoms = len(serials)

    atomTemplate, residueTemplate = read_xml(xml)
    atomDicts, residueDicts = init_residues(serials, names, resNames, resSeqs, positions, charges, atomTemplate, residueTemplate)

    Q = np.vstack(
        [(atom.c0, atom.dX*10, atom.dY*10, atom.dZ*10, atom.qXX*300, atom.qYY*300, atom.qZZ*300, atom.qXY*300, atom.qXZ*300, atom.qYZ*300) for atom in atomDicts.values()]
    )
    Q = jnp.array(Q)
    Q_local = convert_cart2harm(Q, 2)
    axis_type = np.array(
        [atom.axisType for atom in atomDicts.values()]
    )
    axis_indices = np.vstack(
        [atom.axis_indices for atom in atomDicts.values()]
    )
    covalent_map = assemble_covalent(residueDicts, n_atoms)

    
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
    nbr = neighbor_list_fn.allocate(positions)
    pairs = nbr.idx.T
    # pairs = pairs[pairs[:, 0] < pairs[:, 1]]

    lmax = 2


    # Finish data preparation
    # -------------------------------------------------------------------------------------
    # kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
    # # for debugging
    # kappa = 0.657065221219616
    # construct_local_frames_fn = generate_construct_local_frames(axis_type, axis_indices)
    # energy_force_pme = value_and_grad(energy_pme)
    # e, f = energy_force_pme(positions, box, pairs, Q_local, mScales, pScales, dScales, covalent_map, construct_local_frames_fn, kappa, K1, K2, K3, lmax)
    # print('ok')
    # e, f = energy_force_pme(positions, box, pairs, Q_local, mScales, pScales, dScales, covalent_map, construct_local_frames_fn, kappa, K1, K2, K3, lmax)
    # print(e)

    pme_force = ADMPPmeForce(box, axis_type, axis_indices, covalent_map, rc, ethresh, lmax)
    pme_force.update_env('kappa', 0.657065221219616)

    E, F = pme_force.get_forces(positions, box, pairs, Q_local, mScales)
    print('ok')
    E, F = pme_force.get_forces(positions, box, pairs, Q_local, mScales)
    print(E)
    return


# below is the validation code
if __name__ == '__main__':
    validation(sys.argv[1])
