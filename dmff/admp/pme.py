from typing import Tuple, Optional
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, vmap, jit
from jax.scipy.special import erf, erfc

from ..settings import DO_JIT
from ..common.constants import DIELECTRIC
from ..utils import jit_condition, regularize_pairs, pair_buffer_scales
from .settings import POL_CONV, MAX_N_POL
from .recip import generate_pme_recip, Ck_1
from .multipole import (
    C1_c2h,
    C1_h2c,
    C2_h2c,
    convert_cart2harm,
    rot_ind_global2local,
    rot_global2local,
    rot_local2global,
)
from .spatial import v_pbc_shift, generate_construct_local_frames, build_quasi_internal
from .pairwise import (
    distribute_scalar,
    distribute_v3,
    distribute_multipoles,
    distribute_matrix,
)


DEFAULT_THOLE_WIDTH = 5.0


class ADMPPmeForce:
    """
    This is a convenient wrapper for multipolar PME calculations
    It wrapps all the environment parameters of multipolar PME calculation
    The so called "environment paramters" means parameters that do not need to be differentiable
    """

    def __init__(
        self,
        box,
        map_atomtype,
        axis_type,
        axis_indices,
        rc,
        ethresh,
        lmax,
        lpol=False,
        lpme=True,
        steps_pol=None,
        has_aux=False,
    ):
        """
        Initialize the ADMPPmeForce calculator.

        Input:
            box:
                (3, 3) float, box size in row
            axis_type:
                (na,) int, types of local axis (bisector, z-then-x etc.)
            rc:
                float: cutoff distance
            ethresh:
                float: pme energy threshold
            lmax:
                int: max L for multipoles
            lpol:
                bool: polarize or not?
            lpme:
                bool: do pme or simple cutoff?
                if False, the kappa will be set to zero and the reciprocal part will not be computed
            steps:
                None or int: Whether do fixed number of dipole iteration steps?
                if None: converge dipoles until convergence threshold is met
                if int: optimize for this many steps and stop, this is useful if you want to jit the entire function

        Output:

        """
        self.map_atomtype = map_atomtype
        self.axis_type = axis_type
        self.axis_indices = axis_indices
        self.rc = rc
        self.ethresh = ethresh
        self.lmax = int(lmax)  # jichen: type checking
        # turn off pme if lpme is False, this is useful when doing cluster calculations
        self.lpme = lpme
        if self.lpme is False:
            self.kappa = 0.0
            self.K1 = 0
            self.K2 = 0
            self.K3 = 0
        else:
            kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
            self.kappa = kappa
            self.K1 = K1
            self.K2 = K2
            self.K3 = K3
        self.pme_order = 6
        self.lpol = lpol
        self.steps_pol = steps_pol
        # self.n_atoms = int(covalent_map.shape[0]) # len(axis_type)
        self.n_atoms = len(axis_type)
        self.has_aux = has_aux

        self.Q_local_to_global = self.generate_Q_global_function()

        # setup calculators
        self.refresh_calculators()
        return

    def generate_get_energy(self):
        # if the force field is not polarizable
        if not self.lpol:

            def get_energy(positions, box, pairs, Q_local, mScales):
                return energy_pme(
                    positions,
                    box,
                    pairs,
                    Q_local,
                    None,
                    None,
                    None,
                    mScales,
                    None,
                    None,
                    self.construct_local_frames,
                    self.pme_recip,
                    self.kappa,
                    self.K1,
                    self.K2,
                    self.K3,
                    self.lmax,
                    False,
                    lpme=self.lpme,
                )

            return get_energy
        else:
            # this is the bare energy calculator, with Uind as explicit input
            def energy_fn(
                positions,
                box,
                pairs,
                Q_local,
                Uind_global,
                pol,
                tholes,
                mScales,
                pScales,
                dScales,
            ):
                return energy_pme(
                    positions,
                    box,
                    pairs,
                    Q_local,
                    Uind_global,
                    pol,
                    tholes,
                    mScales,
                    pScales,
                    dScales,
                    self.construct_local_frames,
                    self.pme_recip,
                    self.kappa,
                    self.K1,
                    self.K2,
                    self.K3,
                    self.lmax,
                    True,
                    lpme=self.lpme,
                )

            self.energy_fn = energy_fn
            self.grad_U_fn = grad(energy_fn, argnums=(4))
            self.grad_pos_fn = grad(energy_fn, argnums=(0))
            self.U_ind = U_ind = jnp.zeros((self.n_atoms, 3))

            # this is the wrapper that include a Uind optimizer
            def get_energy(
                positions,
                box,
                pairs,
                Q_local,
                pol,
                tholes,
                mScales,
                pScales,
                dScales,
                U_init=U_ind,
                aux=None,
            ):
                U_ind, lconverg, n_cycle = self.optimize_Uind(
                    positions,
                    box,
                    pairs,
                    Q_local,
                    pol,
                    tholes,
                    mScales,
                    pScales,
                    dScales,
                    U_init=U_init * 10.0,
                    steps_pol=self.steps_pol,
                )  # nm to angstrom
                self.U_ind = U_ind
                # here we rely on Feynman-Hellman theorem, drop the term dV/dU*dU/dr !
                # self.U_ind = jax.lax.stop_gradient(U_ind)
                energy = energy_fn(
                    positions,
                    box,
                    pairs,
                    Q_local,
                    U_ind,
                    pol,
                    tholes,
                    mScales,
                    pScales,
                    dScales,
                )
                if aux is not None:
                    aux["U_ind"] = U_ind * 0.1  # Angstrom to nm
                    aux["lconverg"] = lconverg
                    aux["n_cycle"] = n_cycle
                    return energy, aux
                else:
                    return energy

            return get_energy

    def generate_esp(self):
        @jit_condition()
        def esp_kernel(particle, grid, Qtot, Uind):
            deltaR = particle - grid  # nm to A
            r2 = deltaR.dot(deltaR) + 1e-16
            r = jnp.sqrt(r2)
            rr1 = 1.0 / r
            rr2 = rr1 * rr1
            rr3 = rr1 * rr2
            charge = Qtot[0]
            potential = charge * rr1

            dipole = Qtot[1:4] / 10.0
            scd = dipole.dot(deltaR)
            scu = Uind.dot(deltaR)
            potential -= (scd + scu) * rr3

            rr5 = 3.0 * rr3 * rr2
            quad = Qtot[4:13] / 300.0
            QXX, QYY, QZZ, QXY, QXZ, QYZ = 0, 1, 2, 3, 4, 5
            scq = deltaR[0] * (
                quad[QXX] * deltaR[0] + quad[QXY] * deltaR[1] + quad[QXZ] * deltaR[2]
            )
            scq += deltaR[1] * (
                quad[QXY] * deltaR[0] + quad[QYY] * deltaR[1] + quad[QYZ] * deltaR[2]
            )
            scq += deltaR[2] * (
                quad[QXZ] * deltaR[0] + quad[QYZ] * deltaR[1] + quad[QZZ] * deltaR[2]
            )
            potential += scq * rr5
            return potential * DIELECTRIC * 0.1

        esp_point_kernel = jax.vmap(esp_kernel, in_axes=(0, None, 0, 0), out_axes=0)

        @jit_condition()
        def esp_point(positions, grid, Q, U):
            esp = esp_point_kernel(positions, grid, Q, U)
            return jnp.sum(esp)

        esp_grid = jax.vmap(esp_point, in_axes=(None, 0, None, None), out_axes=0)

        if self.lpol:

            @jit_condition()
            def get_esp(positions, grids, Q_local, U_ind):
                box = jnp.eye(3) * 1000.0
                Q_global = self.Q_local_to_global(positions, box, Q_local)
                esp = esp_grid(positions, grids, Q_global, U_ind)
                return esp.reshape((grids.shape[0],))

        else:

            @jit_condition()
            def get_esp(positions, grids, Q_local):
                U_ind = jnp.zeros(positions.shape)
                box = jnp.eye(3) * 1000.0
                Q_global = self.Q_local_to_global(positions, box, Q_local)
                esp = esp_grid(positions, grids, Q_global, U_ind)
                return esp.reshape((grids.shape[0],))

        return get_esp

    def generate_Q_global_function(self):
        @jit_condition()
        def get_Q_global(positions, box, Q_local):
            local_frames = self.construct_local_frames(positions, box)
            Q_global_h = rot_local2global(
                Q_local[self.map_atomtype], local_frames, self.lmax
            )
            C = Q_global_h[:, 0].reshape((-1, 1))
            D = C1_h2c.dot(Q_global_h[:, 1:4].T).T
            Q = C2_h2c.dot(Q_global_h[:, 4:9].T).T
            Q_global_c = jnp.hstack((C, D, Q))
            return Q_global_c

        return get_Q_global

    def update_env(self, attr, val):
        """
        Update the environment of the calculator
        """
        setattr(self, attr, val)
        self.refresh_calculators()

    def refresh_calculators(self):
        """
        refresh the energy and force calculators according to the current environment
        """
        if self.lmax > 0:
            self.construct_local_frames = generate_construct_local_frames(
                self.axis_type, self.axis_indices
            )
        else:
            self.construct_local_frames = None
        lmax = self.lmax
        # for polarizable monopole force field, need to increase lmax to 1, accomodating induced dipoles
        if self.lmax == 0 and self.lpol is True:
            lmax = 1
        self.pme_recip = generate_pme_recip(
            Ck_1, self.kappa, False, self.pme_order, self.K1, self.K2, self.K3, lmax
        )
        # generate the force calculator
        self.get_energy = self.generate_get_energy()
        self.get_forces = value_and_grad(self.get_energy)
        return

    def optimize_Uind(
        self,
        positions,
        box,
        pairs,
        Q_local,
        pol,
        tholes,
        mScales,
        pScales,
        dScales,
        U_init=None,
        steps_pol=None,
        maxiter=MAX_N_POL,
        thresh=POL_CONV,
    ):
        """
        This function converges the induced dipole
        Note that we cut all the gradient chain passing through this function as we assume Feynman-Hellman theorem
        Gradients related to Uind should be dropped
        """
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
        if steps_pol is None:
            site_filter = pol > 0.001  # focus on the actual polarizable sites

        if steps_pol is None:
            for i in range(maxiter):
                field = self.grad_U_fn(
                    positions,
                    box,
                    pairs,
                    Q_local,
                    U,
                    pol,
                    tholes,
                    mScales,
                    pScales,
                    dScales,
                )
                # E = self.energy_fn(positions, box, pairs, Q_local, U, pol, tholes, mScales, pScales, dScales)
                if jnp.max(jnp.abs(field[site_filter])) < thresh:
                    break
                U = U - field * pol[:, jnp.newaxis] / DIELECTRIC
            if i == maxiter - 1:
                flag = False
            else:  # converged
                flag = True
            n_cycles = i
        else:

            def update_U(i, U):
                field = self.grad_U_fn(
                    positions,
                    box,
                    pairs,
                    Q_local,
                    U,
                    pol,
                    tholes,
                    mScales,
                    pScales,
                    dScales,
                )
                U = U - field * pol[:, jnp.newaxis] / DIELECTRIC
                return U

            U = jax.lax.fori_loop(0, steps_pol, update_U, U)
            flag = True
            n_cycles = steps_pol
        return U, flag, n_cycles


def setup_ewald_parameters(
    rc: float,
    ethresh: float,
    box: Optional[jnp.ndarray] = None,
    spacing: Optional[float] = None,
    method: str = "openmm",
) -> Tuple[float, int, int, int]:
    """
    Given the cutoff distance, and the required precision, determine the parameters used in
    Ewald sum, including: kappa, K1, K2, and K3.


    Parameters:
    ----------
    rc: float
        The cutoff distance, in nm
    ethresh: float
        Required energy precision, in kJ/mol
    box: ndarray, optional
        3*3 matrix, box size, a, b, c arranged in rows, used in openmm method
    spacing: float, optional
        fourier spacing to determine K, used in gromacs method
    method: str
        Method to determine ewald parameters. Valid values: "openmm" or "gromacs".
        If openmm, the algorithm can refer to http://docs.openmm.org/latest/userguide/theory.html
        If gromacs, the algorithm is adapted from gromacs source code

    Returns
    -------
    kappa, K1, K2, K3: (float, int, int, int)
        float, the attenuation factor
    K1, K2, K3:
        integers, sizes of the k-points mesh
    """
    if method == "openmm":
        kappa = jnp.sqrt(-jnp.log(2 * ethresh)) / rc
        K1 = jnp.ceil(2 * kappa * box[0, 0] / 3 / ethresh**0.2)
        K2 = jnp.ceil(2 * kappa * box[1, 1] / 3 / ethresh**0.2)
        K3 = jnp.ceil(2 * kappa * box[2, 2] / 3 / ethresh**0.2)

        return kappa, int(K1), int(K2), int(K3)
    elif method == "gromacs":
        # determine kappa
        kappa = 5.0
        i = 0
        while erfc(kappa * rc) > ethresh:
            i += 1
            kappa *= 2

        n = i + 60
        low = 0.0
        high = kappa
        for k in range(n):
            kappa = (low + high) / 2
            if erfc(kappa * rc) > ethresh:
                low = kappa
            else:
                high = kappa
        # determine K
        K1 = int(jnp.ceil(box[0, 0] / spacing))
        K2 = int(jnp.ceil(box[1, 1] / spacing))
        K3 = int(jnp.ceil(box[2, 2] / spacing))
        return kappa, K1, K2, K3
    else:
        raise ValueError(
            f"Invalid method: {method}." "Valid methods: 'openmm', 'gromacs'"
        )


# @jit_condition(static_argnums=())
def energy_pme(
    positions,
    box,
    pairs,
    Q_local,
    Uind_global,
    pol,
    tholes,
    mScales,
    pScales,
    dScales,
    construct_local_frame_fn,
    pme_recip_fn,
    kappa,
    K1,
    K2,
    K3,
    lmax,
    lpol,
    lpme=True,
):
    """
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
            Np * 3: interacting pair indices and topology distance
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
        lpol:
            bool: if polarizable or not? if yes, 1, otherwise 0
        lpme:
            bool: doing pme? If false, then turn off reciprocal space and set kappa = 0

    Output:
        energy: total pme energy
    """
    # if doing a multipolar calculation
    if lmax > 0:
        local_frames = construct_local_frame_fn(positions, box)
        Q_global = rot_local2global(Q_local, local_frames, lmax)
    else:
        if lpol:
            # if fixed multipole only contains charge, and it's polarizable, then expand Q matrix
            dips = jnp.zeros((Q_local.shape[0], 3))
            Q_global = jnp.hstack((Q_local, dips))
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

    if lpme is False:
        kappa = 0

    if lpol:
        ene_real = pme_real(
            positions,
            box,
            pairs,
            Q_global,
            U_ind,
            pol,
            tholes,
            mScales,
            pScales,
            dScales,
            kappa,
            lmax,
            True,
        )
    else:
        ene_real = pme_real(
            positions,
            box,
            pairs,
            Q_global,
            None,
            None,
            None,
            mScales,
            None,
            None,
            kappa,
            lmax,
            False,
        )

    if lpme:
        ene_recip = pme_recip_fn(positions, box, Q_global_tot)
        ene_self = pme_self(Q_global_tot, kappa, lmax)

        if lpol:
            ene_self += pol_penalty(U_ind, pol)
        return ene_real + ene_recip + ene_self

    else:
        if lpol:
            ene_self = pol_penalty(U_ind, pol)
        else:
            ene_self = 0.0
        return ene_real + ene_self


# @partial(vmap, in_axes=(0, 0, None, None), out_axes=0)
@jit_condition(static_argnums=(3))
def calc_e_perm(dr, mscales, kappa, lmax=2):
    r"""
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
    """

    # be aware of unit and dimension !!
    rInv = 1 / dr
    rInvVec = jnp.array([DIELECTRIC * jnp.power(rInv + 1e-16, i) for i in range(0, 9)])
    alphaRVec = jnp.array([jnp.power(kappa * dr + 1e-16, i) for i in range(0, 10)])
    X = 2 * jnp.exp(-alphaRVec[2]) / jnp.sqrt(np.pi)
    tmp = jnp.array(alphaRVec[1])
    doubleFactorial = 1
    facCount = 1
    erfAlphaR = erf(alphaRVec[1])

    # bVec = jnp.empty((6, len(erfAlphaR)))
    bVec = jnp.empty(6)

    bVec = bVec.at[1].set(-erfAlphaR)
    for i in range(2, 6):
        bVec = bVec.at[i].set((bVec[i - 1] + (tmp * X / doubleFactorial)))
        facCount += 2
        doubleFactorial *= facCount
        tmp *= 2 * alphaRVec[2]

    # C-C: 1
    cc = rInvVec[1] * (mscales + bVec[2] - alphaRVec[1] * X)
    if lmax >= 1:
        # C-D
        cd = rInvVec[2] * (mscales + bVec[2])
        # D-D: 2
        dd_m0 = -2 / 3 * rInvVec[3] * (3 * (mscales + bVec[3]) + alphaRVec[3] * X)
        dd_m1 = rInvVec[3] * (mscales + bVec[3] - (2 / 3) * alphaRVec[3] * X)
    else:
        cd = 0
        dd_m0 = 0
        dd_m1 = 0

    if lmax >= 2:
        ## C-Q: 1
        cq = (mscales + bVec[3]) * rInvVec[3]
        ## D-Q: 2
        dq_m0 = rInvVec[4] * (3 * (mscales + bVec[3]) + (4 / 3) * alphaRVec[5] * X)
        dq_m1 = -jnp.sqrt(3) * rInvVec[4] * (mscales + bVec[3])
        ## Q-Q
        qq_m0 = rInvVec[5] * (
            6 * (mscales + bVec[4])
            + (4 / 45) * (-3 + 10 * alphaRVec[2]) * alphaRVec[5] * X
        )
        qq_m1 = -(4 / 15) * rInvVec[5] * (15 * (mscales + bVec[4]) + alphaRVec[5] * X)
        qq_m2 = rInvVec[5] * (mscales + bVec[4] - (4 / 15) * alphaRVec[5] * X)
    else:
        cq = 0
        dq_m0 = 0
        dq_m1 = 0
        qq_m0 = 0
        qq_m1 = 0
        qq_m1 = 0
        qq_m2 = 0

    if lmax >= 3:
        ## C-O
        co = rInvVec[4] * (-mscales - bVec[3] - (4/15)*alphaRVec[5]*X)
        ## D-O
        do_m0 = -4.0 * rInvVec[5] * (mscales + bVec[4] + (2/15)*alphaRVec[7]*X)
        do_m1 = jnp.sqrt(6) * (mscales + bVec[4]) * rInvVec[5]
        ## Q-O
        qo_m0 = rInvVec[6] * (-10.0 * (mscales + bVec[4]) - (8/45) * (3.0 + 2.0*alphaRVec[2])*alphaRVec[7]*X)
        qo_m1 = 5.0 * jnp.sqrt(2) * rInvVec[6] * (mscales + bVec[4] + (8/75)*alphaRVec[7]*X)
        qo_m2 = -jnp.sqrt(5) * (mscales + bVec[4]) * rInvVec[6] 
        ## O-O
        oo_m0 = rInvVec[7] * (-20.0*(mscales+bVec[5]) - (8/1575)*(15.0+28.0*alphaRVec[2] + 28.0*alphaRVec[4])*alphaRVec[7]*X)
        oo_m1 = rInvVec[7] * (15.0*(mscales+bVec[5]) + (8/525)*(-5.0 + 28.0*alphaRVec[2])*alphaRVec[7]*X)
        oo_m2 = rInvVec[7] * (-6.0*(mscales+bVec[5]) - (8/105)*alphaRVec[7]*X)
        oo_m3 = rInvVec[7] * ((mscales+bVec[5]) - (8/105)*alphaRVec[7]*X)
    else:
        co = 0
        do_m0 = 0
        do_m1 = 0
        qo_m0 = 0
        qo_m1 = 0
        qo_m2 = 0
        oo_m0 = 0
        oo_m1 = 0
        oo_m2 = 0
        oo_m3 = 0


    return cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2, co, do_m0, do_m1, qo_m0, qo_m1, qo_m2, oo_m0, oo_m1, oo_m2, oo_m3


@jit_condition(static_argnums=())
def switch_val(x, x0, sigma, y0, y1):
    """
    This is a Fermi function switches between y0 and y1, according to the value of x
    y = y0 when x << x0
    y = y1 when x >> x1
    sigma control sthe switch width
    """
    u = (x - x0) / sigma
    w0 = 1 / (jnp.exp(u) + 1)
    w1 = 1 - w0
    return w0 * y0 + w1 * y1


def gen_trim_val_0(thresh):
    """
    Trim the value at zero point to avoid singularity
    """

    def trim_val_0(x):
        return jnp.piecewise(
            x, [x < thresh, x >= thresh], [lambda x: jnp.array(thresh), lambda x: x]
        )

    if DO_JIT:
        return jit(trim_val_0)
    else:
        return trim_val_0


trim_val_0 = gen_trim_val_0(1e-8)


def gen_trim_val_infty(thresh):
    """
    Trime the value at infinity to avoid divergence
    """

    def trim_val_infty(x):
        return jnp.piecewise(
            x, [x < thresh, x >= thresh], [lambda x: x, lambda x: jnp.array(thresh)]
        )

    if DO_JIT:
        return jit(trim_val_infty)
    else:
        return trim_val_infty


trim_val_infty = gen_trim_val_infty(1e8)


@jit_condition(static_argnums=(7))
def calc_e_ind(dr, thole1, thole2, dmp, pscales, dscales, kappa, lmax=2):
    r"""
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
    """
    ## pscale == 0 ? thole1 + thole2 : DEFAULT_THOLE_WIDTH
    w = jnp.heaviside(pscales, 0)
    a = w * DEFAULT_THOLE_WIDTH + (1 - w) * (thole1 + thole2)

    dmp = trim_val_0(dmp)
    u = trim_val_infty(dr / dmp)

    ## au <= 50 aupi = au ;au> 50 aupi = 50
    au = a * u
    expau = jnp.piecewise(
        au, [au < 50, au >= 50], [lambda au: jnp.exp(-au), lambda au: jnp.array(0)]
    )

    ## compute the Thole damping factors for energies
    au2 = trim_val_infty(au * au)
    au3 = trim_val_infty(au2 * au)
    au4 = trim_val_infty(au3 * au)
    au5 = trim_val_infty(au4 * au)
    au6 = trim_val_infty(au5 * au)

    ##  Thole damping factors for energies
    thole_c = 1.0 - expau * (1.0 + au + 0.5 * au2)
    thole_d0 = 1.0 - expau * (1.0 + au + 0.5 * au2 + au3 / 4.0)
    thole_d1 = 1.0 - expau * (1.0 + au + 0.5 * au2)
    thole_q0 = 1.0 - expau * (1.0 + au + 0.5 * au2 + au3 / 6.0 + au4 / 18.0)
    thole_q1 = 1.0 - expau * (1.0 + au + 0.5 * au2 + au3 / 6.0)
    thole_o0  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/24.0 + au5/120.0)
    thole_o1  = 1.0 - expau*(1.0 + au + 0.5*au2 + au3/6.0 + au4/30.0)
    
    # copied from calc_e_perm
    # be aware of unit and dimension !!
    rInv = 1 / dr
    rInvVec = jnp.array([DIELECTRIC * jnp.power(rInv + 1e-16, i) for i in range(0, 9)])
    alphaRVec = jnp.array([jnp.power(kappa * dr + 1e-16, i) for i in range(0, 10)])
    X = 2 * jnp.exp(-alphaRVec[2]) / jnp.sqrt(np.pi)
    tmp = jnp.array(alphaRVec[1])
    doubleFactorial = 1
    facCount = 1
    erfAlphaR = erf(alphaRVec[1])

    # bVec = jnp.empty((6, len(erfAlphaR)))
    bVec = jnp.empty(6)

    bVec = bVec.at[1].set(-erfAlphaR)
    for i in range(2, 6):
        bVec = bVec.at[i].set((bVec[i - 1] + (tmp * X / doubleFactorial)))
        facCount += 2
        doubleFactorial *= facCount
        tmp *= 2 * alphaRVec[2]

    ## C-Uind
    cud = 2.0 * rInvVec[2] * (pscales * thole_c + bVec[2])
    if lmax >= 1:
        ##  D-Uind terms
        dud_m0 = (
            -2.0
            * 2.0
            / 3.0
            * rInvVec[3]
            * (3.0 * (pscales * thole_d0 + bVec[3]) + alphaRVec[3] * X)
        )
        dud_m1 = (
            2.0
            * rInvVec[3]
            * (pscales * thole_d1 + bVec[3] - 2.0 / 3.0 * alphaRVec[3] * X)
        )
    else:
        dud_m0 = 0.0
        dud_m1 = 0.0

    if lmax >= 2:
        ## Uind-Q
        udq_m0 = (
            2.0
            * rInvVec[4]
            * (3.0 * (pscales * thole_q0 + bVec[3]) + 4 / 3 * alphaRVec[5] * X)
        )
        udq_m1 = -2.0 * jnp.sqrt(3) * rInvVec[4] * (pscales * thole_q1 + bVec[3])
    else:
        udq_m0 = 0.0
        udq_m1 = 0.0
        
    if lmax >= 3:
        ## Uind-O
        udo_m0 = (
            -8.0 * 
            rInvVec[5] * 
            (pscales * thole_o0 + bVec[4] + (2/15)*alphaRVec[7]*X)
        )
        udo_m1 = (
            2.0 * 
            jnp.sqrt(6) * 
            (pscales * thole_o1+bVec[4]) * 
            rInvVec[5]
        )
    else:
        udo_m0 = 0.0
        udo_m1 = 0.0 
    
    ## Uind-Uind
    udud_m0 = (
        -2.0
        / 3.0
        * rInvVec[3]
        * (3.0 * (dscales * thole_d0 + bVec[3]) + alphaRVec[3] * X)
    )
    udud_m1 = rInvVec[3] * (dscales * thole_d1 + bVec[3] - 2.0 / 3.0 * alphaRVec[3] * X)
    return cud, dud_m0, dud_m1, udq_m0, udq_m1, udo_m0, udo_m1, udud_m0, udud_m1


@partial(vmap, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None), out_axes=0)
@jit_condition(static_argnums=(12, 13))
def pme_real_kernel(
    dr,
    qiQI,
    qiQJ,
    qiUindI,
    qiUindJ,
    thole1,
    thole2,
    dmp,
    mscales,
    pscales,
    dscales,
    kappa,
    lmax=2,
    lpol=False,
):
    """
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
    """
    cc, cd, dd_m0, dd_m1, cq, dq_m0, dq_m1, qq_m0, qq_m1, qq_m2, co, do_m0, do_m1, qo_m0, qo_m1, qo_m2, oo_m0, oo_m1, oo_m2, oo_m3 = calc_e_perm(
        dr, mscales, kappa, lmax
    )
    if lpol:
        cud, dud_m0, dud_m1, udq_m0, udq_m1, udo_m0, udo_m1, udud_m0, udud_m1 = calc_e_ind(
            dr, thole1, thole2, dmp, pscales, dscales, kappa, lmax
        )

    Vij0 = cc * qiQI[0]
    Vji0 = cc * qiQJ[0]
    # C-Uind
    if lpol:
        Vij0 -= cud * qiUindI[0]
        Vji0 += cud * qiUindJ[0]

    if lmax >= 1:
        # C-D
        Vij0 = Vij0 - cd * qiQI[1]
        Vji1 = -cd * qiQJ[0]
        Vij1 = cd * qiQI[0]
        Vji0 = Vji0 + cd * qiQJ[1]
        # D-D m0
        Vij1 += dd_m0 * qiQI[1]
        Vji1 += dd_m0 * qiQJ[1]
        # D-D m1
        Vij2 = dd_m1 * qiQI[2]
        Vji2 = dd_m1 * qiQJ[2]
        Vij3 = dd_m1 * qiQI[3]
        Vji3 = dd_m1 * qiQJ[3]
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
        Vij0 = Vij0 + cq * qiQI[4]
        Vji4 = cq * qiQJ[0]
        Vij4 = cq * qiQI[0]
        Vji0 = Vji0 + cq * qiQJ[4]
        # D-Q m0
        Vij1 += dq_m0 * qiQI[4]
        Vji4 += dq_m0 * qiQJ[1]
        # Q-D m0
        Vij4 -= dq_m0 * qiQI[1]
        Vji1 -= dq_m0 * qiQJ[4]
        # D-Q m1
        Vij2 = Vij2 + dq_m1 * qiQI[5]
        Vji5 = dq_m1 * qiQJ[2]
        Vij3 += dq_m1 * qiQI[6]
        Vji6 = dq_m1 * qiQJ[3]
        Vij5 = -(dq_m1 * qiQI[2])
        Vji2 += -(dq_m1 * qiQJ[5])
        Vij6 = -(dq_m1 * qiQI[3])
        Vji3 += -(dq_m1 * qiQJ[6])
        # Q-Q m0
        Vij4 += qq_m0 * qiQI[4]
        Vji4 += qq_m0 * qiQJ[4]
        # Q-Q m1
        Vij5 += qq_m1 * qiQI[5]
        Vji5 += qq_m1 * qiQJ[5]
        Vij6 += qq_m1 * qiQI[6]
        Vji6 += qq_m1 * qiQJ[6]
        # Q-Q m2
        Vij7 = qq_m2 * qiQI[7]
        Vji7 = qq_m2 * qiQJ[7]
        Vij8 = qq_m2 * qiQI[8]
        Vji8 = qq_m2 * qiQJ[8]
        # Q-Uind
        if lpol:
            Vji4 += udq_m0 * qiUindJ[0]
            Vij4 -= udq_m0 * qiUindI[0]
            Vji5 += udq_m1 * qiUindJ[1]
            Vji6 += udq_m1 * qiUindJ[2]
            Vij5 -= udq_m1 * qiUindI[1]
            Vij6 -= udq_m1 * qiUindI[2]

    if lmax >= 3:
        # C-O
        Vij0 = Vij0 + co*qiQI[9]
        Vji9 = co*qiQJ[0]
        Vij9 = -co*qiQI[0]
        Vji0 = Vji0 - co*qiQJ[9]
        # D-O m0
        Vij1 += do_m0*qiQI[9]
        Vji9 += do_m0*qiQJ[1]
        # O-D m0
        Vij9 += do_m0*qiQI[1]
        Vji1 += do_m0*qiQJ[9]
        # D-O m1
        Vij2 = Vij2 + do_m1*qiQI[10]
        Vji10 = do_m1*qiQJ[2]
        Vij3 += do_m1*qiQI[11]
        Vji11 = do_m1*qiQJ[3]
        # O-D m1
        Vij10 = do_m1*qiQI[2]
        Vji2 += do_m1*qiQJ[10]
        Vij11 = do_m1*qiQI[3]
        Vji3 += do_m1*qiQJ[11]
        # Q-O m0
        Vij4 += qo_m0*qiQI[9]
        Vji9 += qo_m0*qiQJ[4]
        # O-Q m0
        Vij9 -= qo_m0*qiQI[4]
        Vji4 -= qo_m0*qiQJ[9]
        # Q-O m1
        Vij5 += qo_m1*qiQI[10]
        Vji10 += qo_m1*qiQJ[5]
        Vij6 += qo_m1*qiQI[11]
        Vji11 += qo_m1*qiQJ[6]
        # O-Q m1
        Vij10 -= qo_m1*qiQI[5]
        Vji5  -= qo_m1*qiQJ[10]
        Vij11 -= qo_m1*qiQI[6]
        Vji6  -= qo_m1*qiQJ[11]
        # Q-O m2
        Vij7 += qo_m2*qiQI[12]
        Vji12 = qo_m2*qiQJ[7]
        Vij8 += qo_m2*qiQI[13]
        Vji13 = qo_m2*qiQJ[8]
        # O-Q m2
        Vij12 = -qo_m2*qiQI[7]
        Vji7 -=  qo_m2*qiQJ[12]
        Vij13 = -qo_m2*qiQI[8]
        Vji8 -=  qo_m2*qiQJ[13]
        # O-O m0
        Vij9 += oo_m0*qiQI[9]
        Vji9 += oo_m0*qiQJ[9]
        # O-O m1
        Vij10 += oo_m1*qiQI[10]
        Vji10 += oo_m1*qiQJ[10]
        Vij11 += oo_m1*qiQI[11]
        Vji11 += oo_m1*qiQJ[11]
        # O-O m2
        Vij12 += oo_m2*qiQI[12]
        Vji12 += oo_m2*qiQJ[12]
        Vij13 += oo_m2*qiQI[13]
        Vji13 += oo_m2*qiQJ[13]
        # O-O m3
        Vij14 = oo_m3*qiQI[14]
        Vji14 = oo_m3*qiQJ[14]
        Vij15 = oo_m3*qiQI[15]
        Vji15 = oo_m3*qiQJ[15]
        if lpol:
            # m = 0
            Vji9 += udo_m0*qiUindJ[0]
            Vij9 += udo_m0*qiUindI[0]
            # m = 1
            Vji10 += udo_m1*qiUindJ[1]
            Vji11 += udo_m1*qiUindJ[2]
            
            Vij10 += udo_m1*qiUindI[1]
            Vij11 += udo_m1*qiUindI[2]            

    # Uind - Uind
    if lpol:
        Vij1dd = udud_m0 * qiUindI[0]
        Vji1dd = udud_m0 * qiUindJ[0]
        Vij2dd = udud_m1 * qiUindI[1]
        Vji2dd = udud_m1 * qiUindJ[1]
        Vij3dd = udud_m1 * qiUindI[2]
        Vji3dd = udud_m1 * qiUindJ[2]
        Vijdd = jnp.stack((Vij1dd, Vij2dd, Vij3dd))
        Vjidd = jnp.stack((Vji1dd, Vji2dd, Vji3dd))

    if lmax == 0:
        Vij = Vij0
        Vji = Vji0
    elif lmax == 1:
        Vij = jnp.stack((Vij0, Vij1, Vij2, Vij3))
        Vji = jnp.stack((Vji0, Vji1, Vji2, Vji3))
    elif lmax == 2:
        Vij = jnp.stack((Vij0, Vij1, Vij2, Vij3, Vij4, Vij5, Vij6, Vij7, Vij8))
        Vji = jnp.stack((Vji0, Vji1, Vji2, Vji3, Vji4, Vji5, Vji6, Vji7, Vji8))
    elif lmax == 3:
        Vij = jnp.stack((Vij0, Vij1, Vij2, Vij3, Vij4, Vij5, Vij6, Vij7, Vij8, Vij9, Vij10, Vij11, Vij12, Vij13, Vij14, Vij15))
        Vji = jnp.stack((Vji0, Vji1, Vji2, Vji3, Vji4, Vji5, Vji6, Vji7, Vji8, Vji9, Vji10, Vji11, Vji12, Vji13, Vji14, Vji15))
    else:
        raise ValueError(f"Invalid lmax {lmax}. Valid values are 0, 1, 2")

    if lpol:
        # return jnp.array(0.5) * (jnp.sum(qiQJ*Vij) + jnp.sum(qiQI*Vji)) + jnp.array(0.5) * (jnp.sum(qiUindJ*Vijdd) + jnp.sum(qiUindI*Vjidd))
        return jnp.array(0.5) * (jnp.sum(qiQJ * Vij) + jnp.sum(qiQI * Vji)) + jnp.array(
            0.5
        ) * (jnp.sum(qiUindJ * Vijdd) + jnp.sum(qiUindI * Vjidd))
    else:
        return jnp.array(0.5) * (jnp.sum(qiQJ * Vij) + jnp.sum(qiQI * Vji))


# @jit_condition(static_argnums=(7))
def pme_real(
    positions,
    box,
    pairs,
    Q_global,
    Uind_global,
    pol,
    tholes,
    mScales,
    pScales,
    dScales,
    kappa,
    lmax,
    lpol,
):
    """
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
            Np * 3: interacting pair indices and topology distance
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
    """
    pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))
    buffer_scales = pair_buffer_scales(pairs[:, :2])
    box_inv = jnp.linalg.inv(box + jnp.eye(3) * 1e-36)
    r1 = distribute_v3(positions, pairs[:, 0])
    r2 = distribute_v3(positions, pairs[:, 1])
    Q_extendi = distribute_multipoles(Q_global, pairs[:, 0])
    Q_extendj = distribute_multipoles(Q_global, pairs[:, 1])
    nbonds = pairs[:, 2]
    # nbonds = covalent_map[pairs[:, 0], pairs[:, 1]]
    indices = nbonds - 1
    mscales = distribute_scalar(mScales, indices)
    mscales = mscales * buffer_scales
    if lpol:
        pol1 = distribute_scalar(pol, pairs[:, 0])
        pol2 = distribute_scalar(pol, pairs[:, 1])
        thole1 = distribute_scalar(tholes, pairs[:, 0])
        thole2 = distribute_scalar(tholes, pairs[:, 1])
        Uind_extendi = distribute_v3(Uind_global, pairs[:, 0])
        Uind_extendj = distribute_v3(Uind_global, pairs[:, 1])
        pscales = distribute_scalar(pScales, indices)
        pscales = pscales * buffer_scales
        dscales = distribute_scalar(dScales, indices)
        dscales = dscales * buffer_scales
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
    norm_dr = jnp.linalg.norm(dr + 1e-64, axis=-1)  # add eta to avoid division by zero
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
    elist = (
        pme_real_kernel(
            norm_dr,
            qiQI,
            qiQJ,
            qiUindI,
            qiUindJ,
            thole1,
            thole2,
            dmp,
            mscales,
            pscales,
            dscales,
            kappa,
            lmax,
            lpol,
        )
        * buffer_scales
    )
    ene = jnp.sum(elist)

    return ene


@partial(vmap, in_axes=(0, 0), out_axes=(0))
@jit_condition(static_argnums=())
def get_pair_dmp(pol1, pol2):
    p12 = pol1 * pol2
    return jnp.power(p12 + 1e-50, 1 / 6)


@jit_condition(static_argnums=(2))
def pme_self(Q_h, kappa, lmax=2):
    """
    This function calculates the PME self energy

    Inputs:
        Q:
            Na * (lmax+1)^2: harmonic multipoles, local or global does not matter
        kappa:
            float: kappa used in PME

    Output:
        ene_self:
            float: the self energy
    """
    n_harms = (lmax + 1) ** 2
    l_list = np.array(
        [0]
        + [
            1,
        ]
        * 3
        + [
            2,
        ]
        * 5
    )[:n_harms]
    l_fac2 = np.array(
        [1]
        + [
            3,
        ]
        * 3
        + [
            15,
        ]
        * 5
    )[:n_harms]
    factor = kappa / np.sqrt(np.pi) * (2 * kappa**2) ** l_list / l_fac2
    return -jnp.sum(factor[np.newaxis] * Q_h**2) * DIELECTRIC


@jit_condition(static_argnums=())
def pol_penalty(U_ind, pol):
    """
    The energy penalty for polarization of each site, currently only supports isotropic polarization:

    Inputs:
        U_ind:
            Na * 3 float: induced dipoles, in isotropic polarization case, cartesian or harmonic does not matter
        pol:
            (Na,) float: polarizability
    """
    # this is to remove the singularity when pol=0
    pol_pi = trim_val_0(pol)
    Uind_norm = jnp.linalg.norm(U_ind + 1e-16, axis=1)
    # pol_pi = pol/(jnp.exp((-pol+1e-08)*1e10)+1) + 1e-08/(jnp.exp((pol-1e-08)*1e10)+1)
    return jnp.sum(0.5 / pol_pi * jnp.power(U_ind + 1e-16, 2).sum(axis=1)) * DIELECTRIC
