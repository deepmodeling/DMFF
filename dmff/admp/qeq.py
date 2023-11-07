<<<<<<< HEAD
#!/usr/bin/env python
import sys
import absl 
import numpy as np 
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
from dmff.settings import DO_JIT
from dmff.common.constants import DIELECTRIC
from dmff.common import nblist 
from jax_md import space, partition
from jax import grad, value_and_grad, vmap, jit
from jaxopt import OptaxSolver
from itertools import combinations
import jaxopt
import jax
import scipy
import pickle
=======
import numpy as np
import jax.numpy as jnp
from ..common.constants import DIELECTRIC
from jax import grad, vmap
from ..classical.inter import CoulNoCutoffForce, CoulombPMEForce
from typing import Tuple, List
from ..settings import PRECISION
from .pme import energy_pme
from .recip import generate_pme_recip, Ck_1

if PRECISION == "double":
    CONST_0 = jnp.array(0, dtype=jnp.float64)
    CONST_1 = jnp.array(1, dtype=jnp.float64)
else:
    CONST_0 = jnp.array(0, dtype=jnp.float32)
    CONST_1 = jnp.array(1, dtype=jnp.float32)

try:
    import jaxopt

    try:
        from jaxopt import Broyden

        JAXOPT_OLD = False
    except ImportError:
        JAXOPT_OLD = True
        import warnings
        warnings.warn(
            "jaxopt is too old. The QEQ potential function cannot be jitted. Please update jaxopt to the latest version for speed concern."
        )
except ImportError:
    import warnings
    warnings.warn("jaxopt not found, QEQ cannot be used.")
import jax
>>>>>>> wangxy/v1.0.0-devel

from jax.scipy.special import erf, erfc

from dmff.utils import jit_condition, regularize_pairs, pair_buffer_scales


<<<<<<< HEAD
jax.config.update("jax_enable_x64", True)

class ADMPQeqForce:

    def __init__(self, q, lagmt, damp_mod=3, neutral_flag=True, slab_flag=False,  constQ=True, pbc_flag = True):
        
        self.damp_mod = damp_mod
        self.neutral_flag = neutral_flag
        self.slab_flag = slab_flag
        self.constQ = constQ
        self.pbc_flag = pbc_flag
        self.q = q
        self.lagmt = lagmt
        return

    def generate_get_energy(self):
      #  q = self.q
        damp_mod = self.damp_mod
        neutral_flag = self.neutral_flag
        constQ = self.constQ
        pbc_flag = self.pbc_flag
      #  lagmt = self.lagmt

        if eval(constQ) is True:
            e_constraint = E_constQ
        else:
            e_constraint = E_constP
        self.e_constraint = e_constraint

        if eval(damp_mod) is False:
            e_sr = E_sr0
            e_site = E_site
        elif eval(damp_mod) == 2:
            e_sr = E_sr2
            e_site = E_site2
        elif eval(damp_mod) == 3:
            e_sr = E_sr3
            e_site = E_site3

      #  if pbc_flag is False:
      #      e_coul = E_CoulNocutoff
      #  else:
      #      e_coul = E_coul 
        def get_energy(positions, box, pairs, q, lagmt, eta, chi, J, const_list, const_vals,pme_generator):
            
            pos = positions
            ds = ds_pairs(pos, box, pairs, pbc_flag)
            buffer_scales = pair_buffer_scales(pairs)
            kappa = pme_generator.coulforce.kappa
            def E_full(q, lagmt, const_vals, chi, J, pos, box, pairs, eta, ds, buffer_scales):
                e1 = e_constraint(q, lagmt, const_list, const_vals)
                e2 = e_sr(pos*10, box*10 ,pairs , q , eta, ds*10, buffer_scales)
                e3 = e_site( chi, J , q)
                e4 = pme_generator.coulenergy(pos, box ,pairs, q, pme_generator.mscales_coul)
                e5 = E_corr(pos*10, box*10, pairs, q, kappa/10, neutral_flag)
                return e1 + e2 + e3 + e4 + e5 
            @jit
            def E_grads(b_value, const_vals, chi, J, positions, box, pairs, eta, ds, buffer_scales):
                n_const = len(const_vals)
                q = b_value[:-n_const]
                lagmt = b_value[-n_const:]
                g1,g2 = grad(E_full,argnums=(0,1))(q, lagmt, const_vals, chi, J, positions, box, pairs, eta, ds, buffer_scales)
                g = jnp.concatenate((g1,g2))
                return g
            
            def Q_equi(b_value, const_vals, chi, J, positions, box, pairs, eta, ds, buffer_scales):
                rf=jaxopt.ScipyRootFinding(optimality_fun=E_grads,method='hybr',jit=False,tol=1e-10)
                q0,state1 = rf.run(b_value, const_vals, chi, J, positions, box, pairs, eta, ds, buffer_scales)
                return q0,state1
    
            def get_chgs():
                n_const = len(self.lagmt)
                b_value = jnp.concatenate((self.q,self.lagmt))
                q0,state1 = Q_equi(b_value, const_vals, chi, J, positions, box, pairs, eta, ds, buffer_scales)
                self.q = q0[:-n_const]
                self.lagmt = q0[-n_const:]
                return q0,state1

            q0,state1 = get_chgs()
            self.q0 = q0
            self.state1 = state1
            energy = E_full(self.q, self.lagmt, const_vals, chi, J, positions, box, pairs, eta, ds , buffer_scales)
            self.e_grads = E_grads(q0, const_vals, chi, J, positions, box, pairs, eta, ds, buffer_scales)
            self.e_full = E_full
            return  energy 

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
        # generate the force calculator
        self.get_energy = self.generate_get_energy()
        self.get_forces = value_and_grad(self.get_energy)
        return

def E_constQ(q, lagmt, const_list, const_vals):
    q_sum = []
    for i in range(len(const_list)):
        q_sum.append(np.sum(q[const_list[i]]))
    constraint = (jnp.array(q_sum) - const_vals) * lagmt
    return np.sum(constraint)
def E_constP(q, lagmt, const_list, const_vals):
    q_sum = []
    for i in range(len(const_list)):
        q_sum.append(np.sum(q[const_list[i]]))
    constraint = jnp.array(q_sum) * const_vals
    return np.sum(constraint)

def E_sr(pos, box, pairs, q, eta, ds, buffer_scales ):
    return 0
def E_sr2(pos, box, pairs, q, eta, ds, buffer_scales ):
    etasqrt = jnp.sqrt( 2 * ( jnp.array(eta)[pairs[:,0]] **2 + jnp.array(eta)[pairs[:,1]] **2))
    pre_pair = - eta_piecewise(etasqrt,ds) * DIELECTRIC
    pre_self = etainv_piecewise(eta) /( jnp.sqrt(2 * jnp.pi)) * DIELECTRIC
    e_sr_pair = pre_pair * q[pairs[:,0]] * q[pairs[:,1]] /ds * buffer_scales
    e_sr_self = pre_self * q * q
    e_sr = jnp.sum(e_sr_pair) + jnp.sum(e_sr_self)
    return e_sr
def E_sr3(pos, box, pairs, q, eta, ds, buffer_scales ):
    etasqrt = jnp.sqrt( jnp.array(eta)[pairs[:,0]] **2 +  jnp.array(eta)[pairs[:,1]] **2 )
    pre_pair = - eta_piecewise(etasqrt,ds) * DIELECTRIC
    pre_self = etainv_piecewise(eta) /( jnp.sqrt(2 * jnp.pi)) * DIELECTRIC
    e_sr_pair = pre_pair * q[pairs[:,0]] * q[pairs[:,1]] /ds * buffer_scales
=======
@jit_condition()
def mask_index(idx, max_idx):
    return jnp.piecewise(
        idx, [idx < max_idx, idx >= max_idx], [lambda x: CONST_1, lambda x: CONST_0]
    )


mask_index = jax.vmap(mask_index, in_axes=(0, None))


@jit_condition()
def group_sum(val_list, indices):
    max_idx = val_list.shape[0]
    mask = mask_index(indices, max_idx)
    return jnp.sum(val_list[indices] * mask)


group_sum = jax.vmap(group_sum, in_axes=(None, 0))


def padding_consts(const_list, max_idx):
    max_length = max([len(i) for i in const_list])
    new_const_list = np.zeros((len(const_list), max_length)) + max_idx
    for ncl, cl in enumerate(const_list):
        for nitem, item in enumerate(cl):
            new_const_list[ncl, nitem] = item
    return jnp.array(new_const_list, dtype=int)


@jit_condition()
def E_constQ(q, lagmt, const_list, const_vals):
    constraint = (group_sum(q, const_list) - const_vals) * lagmt
    return jnp.sum(constraint)


@jit_condition()
def E_constP(q, lagmt, const_list, const_vals):
    constraint = group_sum(q, const_list) * const_vals
    return jnp.sum(constraint)


@vmap
@jit_condition()
def mask_to_zero(v, mask):
    return jnp.piecewise(
        v, [mask < 1e-4, mask >= 1e-4], [lambda x: CONST_0, lambda x: v]
    )


@jit_condition()
def E_sr(pos, box, pairs, q, eta, ds, buffer_scales):
    return 0.0


@jit_condition()
def E_sr2(pos, box, pairs, q, eta, ds, buffer_scales):
    etasqrt = jnp.sqrt(2 * (eta[pairs[:, 0]] ** 2 + eta[pairs[:, 1]] ** 2))
    pre_pair = -eta_piecewise(etasqrt, ds) * DIELECTRIC
    pre_self = etainv_piecewise(eta) / (jnp.sqrt(2 * jnp.pi)) * DIELECTRIC
    e_sr_pair = pre_pair * q[pairs[:, 0]] * q[pairs[:, 1]] / ds * buffer_scales
    e_sr_pair = mask_to_zero(e_sr_pair, buffer_scales)
    e_sr_self = pre_self * q * q
    e_sr = jnp.sum(e_sr_pair) + jnp.sum(e_sr_self)
    return e_sr


@jit_condition()
def E_sr3(pos, box, pairs, q, eta, ds, buffer_scales):
    etasqrt = jnp.sqrt(
        eta[pairs[:, 0]] ** 2 + eta[pairs[:, 1]] ** 2 + 1e-64
    )  # add eta to avoid division by zero
    epiece = eta_piecewise(etasqrt, ds)
    pre_pair = -epiece * DIELECTRIC
    pre_self = etainv_piecewise(eta) / (jnp.sqrt(2 * jnp.pi)) * DIELECTRIC
    e_sr_pair = pre_pair * q[pairs[:, 0]] * q[pairs[:, 1]] / ds
    e_sr_pair = mask_to_zero(e_sr_pair, buffer_scales)
>>>>>>> wangxy/v1.0.0-devel
    e_sr_self = pre_self * q * q
    e_sr = jnp.sum(e_sr_pair) + jnp.sum(e_sr_self)
    return e_sr

<<<<<<< HEAD
def E_site(chi, J , q ):
    return 0
def E_site2(chi, J , q ):
    ene = (chi * q + 0.5 * J * q **2 ) * 96.4869
    return np.sum(ene)
def E_site3(chi, J , q ):
    ene =  chi * q *4.184 + J * q **2 *DIELECTRIC * 2 * jnp.pi
    return np.sum(ene)

def E_corr(pos, box, pairs, q, kappa, neutral_flag = True):
   # def E_corr():
    V = jnp.linalg.det(box)
    pre_corr = 2 * jnp.pi / V * DIELECTRIC
    Mz = jnp.sum(q * pos[:,2])
    Q_tot = jnp.sum(q)
    Lz = jnp.linalg.norm(box[3])
    e_corr = pre_corr * (Mz **2 - Q_tot * (jnp.sum(q * pos[:,2] **2)) - Q_tot **2 * Lz **2 /12)
    if eval(neutral_flag) is True:
      #  kappa = pme_potential.pme_force.kappa
        pre_corr_non = - jnp.pi / (2 * V * kappa **2) * DIELECTRIC
        e_corr_non = pre_corr_non * Q_tot **2 
        e_corr += e_corr_non
    return np.sum( e_corr)

def E_CoulNocutoff(pos, box, pairs, q, ds):
    e = q[pairs[:,0]] * q[pairs[:,1]] /ds * DIELECTRIC
    return jnp.sum(e)

def E_Coul(pos, box, pairs, q, ds):
    return 0 

@jit_condition(static_argnums=(3))
def ds_pairs(positions, box, pairs, pbc_flag):
    pos1 = positions[pairs[:,0].astype(int)]
    pos2 = positions[pairs[:,1].astype(int)]
=======

@jit_condition()
def E_site(chi, J, q):
    return 0.0


@jit_condition()
def E_site2(chi, J, q):
    ene = (chi * q + 0.5 * J * q**2) * 96.4869
    return jnp.sum(ene)


@jit_condition()
def E_site3(chi, J, q):
    ene = chi * q * 4.184 + J * q**2 * DIELECTRIC * 2 * jnp.pi
    return jnp.sum(ene)


@jit_condition(static_argnums=[5])
def E_corr(pos, box, pairs, q, kappa, neutral_flag=True):
    # def E_corr():
    V = jnp.linalg.det(box)
    pre_corr = 2 * jnp.pi / V * DIELECTRIC
    Mz = jnp.sum(q * pos[:, 2])
    Q_tot = jnp.sum(q)
    Lz = jnp.linalg.norm(box[3])
    e_corr = pre_corr * (
        Mz**2
        - Q_tot * (jnp.sum(q * pos[:, 2] ** 2))
        - jnp.power(Q_tot, 2) * jnp.power(Lz, 2) / 12
    )
    if neutral_flag:
        #  kappa = pme_potential.pme_force.kappa
        pre_corr_non = -jnp.pi / (2 * V * kappa**2) * DIELECTRIC
        e_corr_non = pre_corr_non * Q_tot**2
        e_corr += e_corr_non
    return jnp.sum(e_corr)


@jit_condition(static_argnums=[3])
def ds_pairs(positions, box, pairs, pbc_flag):
    pos1 = positions[pairs[:, 0]]
    pos2 = positions[pairs[:, 1]]
>>>>>>> wangxy/v1.0.0-devel
    if pbc_flag is False:
        dr = pos1 - pos2
    else:
        box_inv = jnp.linalg.inv(box)
        dpos = pos1 - pos2
        dpos = dpos.dot(box_inv)
<<<<<<< HEAD
        dpos -= jnp.floor(dpos+0.5)
        dr = dpos.dot(box)
    ds = jnp.linalg.norm(dr,axis=1)
    return ds

@jit_condition()
@vmap
def eta_piecewise(eta,ds):
    return jnp.piecewise(eta, (eta > 1e-4, eta <= 1e-4),
                        (lambda x: jnp.array(erfc( ds / eta)), lambda x:jnp.array(0))) 
                         
@jit_condition()
@vmap
def etainv_piecewise(eta):
    return jnp.piecewise(eta, (eta > 1e-4, eta <= 1e-4),
                        (lambda x: jnp.array(1/eta), lambda x:jnp.array(0))) 
                         

=======
        dpos -= jnp.floor(dpos + 0.5)
        dr = dpos.dot(box)
    ds = jnp.linalg.norm(dr + 1e-64, axis=1)  # add eta to avoid division by zero
    return ds


@jit_condition()
def eta_piecewise(eta, ds):
    return jnp.piecewise(
        eta,
        (eta > 1e-4, eta <= 1e-4),
        (lambda x: erfc(ds / x), lambda x: x - x),
    )


eta_piecewise = jax.vmap(eta_piecewise, in_axes=(0, 0))


@jit_condition()
def etainv_piecewise(eta):
    return jnp.piecewise(
        eta,
        (eta > 1e-4, eta <= 1e-4),
        (lambda x: 1 / x, lambda x: x - x),
    )


etainv_piecewise = jax.vmap(etainv_piecewise, in_axes=0)


class ADMPQeqForce:
    def __init__(
        self,
        init_q,
        r_cut: float,
        kappa: float,
        K: Tuple[int, int, int],
        damp_mod: int = 3,
        const_list: List = [],
        const_vals: List = [],
        neutral_flag: bool = True,
        slab_flag: bool = False,
        constQ: bool = True,
        pbc_flag: bool = True,
        has_aux=False,
    ):
        self.has_aux = has_aux
        const_vals = np.array(const_vals)
        if neutral_flag:
            const_vals = const_vals - np.sum(const_vals) / len(const_vals)
        self.const_vals = jnp.array(const_vals)
        assert len(const_list) == len(
            const_vals
        ), "const_list and const_vals must have the same length"
        n_atoms = len(init_q)
        self.const_list = padding_consts(const_list, n_atoms)
        self.init_q = jnp.array(init_q)
        self.init_lagmt = jnp.ones((len(const_list),))

        self.damp_mod = damp_mod
        self.neutral_flag = neutral_flag
        self.slab_flag = slab_flag
        self.constQ = constQ
        self.pbc_flag = pbc_flag

        if constQ:
            e_constraint = E_constQ
        else:
            e_constraint = E_constP
        self.e_constraint = e_constraint

        if damp_mod == 1:
            self.e_sr = E_sr
            self.e_site = E_site
        elif damp_mod == 2:
            self.e_sr = E_sr2
            self.e_site = E_site2
        elif damp_mod == 3:
            self.e_sr = E_sr3
            self.e_site = E_site3
        else:
            raise ValueError("damp_mod must be 1, 2 or 3")

        if pbc_flag:
            pme_recip_fn = generate_pme_recip(
                Ck_fn=Ck_1,
                kappa=kappa / 10,
                gamma=False,
                pme_order=6,
                K1=K[0],
                K2=K[1],
                K3=K[2],
                lmax=0,
            )

            def coul_energy(positions, box, pairs, q, mscales):
                atomCharges = q
                atomChargesT = jnp.reshape(atomCharges, (-1, 1))
                return energy_pme(
                    positions * 10,
                    box * 10,
                    pairs,
                    atomChargesT,
                    None,
                    None,
                    None,
                    mscales,
                    None,
                    None,
                    None,
                    pme_recip_fn,
                    kappa / 10,
                    K[0],
                    K[1],
                    K[2],
                    0,
                    False,
                )

            self.kappa = kappa

        else:

            def get_coul_energy(dr_vec, chrgprod, box):
                dr_norm = jnp.linalg.norm(dr_vec + 1e-64, axis=1) # add eta to avoid division by zero

                dr_inv = 1.0 / dr_norm
                E = chrgprod * DIELECTRIC * 0.1 * dr_inv

                return E

            def coul_energy(positions, box, pairs, q, mscales):
                pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))
                mask = pair_buffer_scales(pairs[:, :2])
                cov_pair = pairs[:, 2]
                mscale_pair = mscales[cov_pair - 1]

                charge0 = q[pairs[:, 0]]
                charge1 = q[pairs[:, 1]]
                chrgprod = charge0 * charge1
                chrgprod_scale = chrgprod * mscale_pair
                dr_vec = positions[pairs[:, 0]] - positions[pairs[:, 1]]

                E_inter = get_coul_energy(dr_vec, chrgprod_scale, box)

                return jnp.sum(E_inter * mask)

            self.kappa = 0.0

        self.coul_energy = coul_energy

    def generate_get_energy(self):
        @jit_condition()
        def E_full(q, lagmt, chi, J, pos, box, pairs, eta, ds, buffer_scales, mscales):
            e1 = self.e_constraint(q, lagmt, self.const_list, self.const_vals)
            e2 = self.e_sr(pos * 10, box * 10, pairs, q, eta, ds * 10, buffer_scales)
            e3 = self.e_site(chi, J, q)
            e4 = self.coul_energy(pos, box, pairs, q, mscales)
            if self.slab_flag:
                e5 = E_corr(
                    pos * 10.0, box * 10.0, pairs, q, self.kappa / 10, self.neutral_flag
                )
                return e1 + e2 + e3 + e4 + e5
            else:
                return e1 + e2 + e3 + e4

        grad_E_full = grad(E_full, argnums=(0, 1))

        @jit_condition()
        def E_grads(
            b_value, chi, J, positions, box, pairs, eta, ds, buffer_scales, mscales
        ):
            n_const = len(self.const_vals)
            q = b_value[:-n_const]
            lagmt = b_value[-n_const:]

            g1, g2 = grad_E_full(
                q, lagmt, chi, J, positions, box, pairs, eta, ds, buffer_scales, mscales
            )
            g = jnp.concatenate((g1, g2))
            return g

        def get_energy(positions, box, pairs, mscales, eta, chi, J, aux=None):
            pos = positions
            ds = ds_pairs(pos, box, pairs, self.pbc_flag)
            buffer_scales = pair_buffer_scales(pairs)

            n_const = len(self.init_lagmt)
            if self.has_aux:
                b_value = jnp.concatenate((aux["q"], aux["lagmt"]))
            else:
                b_value = jnp.concatenate([self.init_q, self.init_lagmt])
            # if JAXOPT_OLD:
            if True:
                rf = jaxopt.ScipyRootFinding(
                    optimality_fun=E_grads, method="hybr", jit=False, tol=1e-10
                )
            else:
                rf = jaxopt.Broyden(fun=E_grads, tol=1e-10)
            b_0, _ = rf.run(
                b_value,
                chi,
                J,
                positions,
                box,
                pairs,
                eta,
                ds,
                buffer_scales,
                mscales,
            )
            b_0 = jax.lax.stop_gradient(b_0)
            q_0 = b_0[:-n_const]
            lagmt_0 = b_0[-n_const:]

            energy = E_full(
                q_0,
                lagmt_0,
                chi,
                J,
                positions,
                box,
                pairs,
                eta,
                ds,
                buffer_scales,
                mscales,
            )
            if self.has_aux:
                aux["q"] = q_0
                aux["lagmt"] = lagmt_0
                return energy, aux
            else:
                return energy

        return get_energy
>>>>>>> wangxy/v1.0.0-devel
