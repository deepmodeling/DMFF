import numpy as np
import jax.numpy as jnp
from ..common.constants import DIELECTRIC
from jax import grad, value_and_grad, vmap, jacfwd, jacrev
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

from jax.scipy.special import erf, erfc

from dmff.utils import jit_condition, regularize_pairs, pair_buffer_scales


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
    constraint = - group_sum(q, const_list) * const_vals * 96.4869
    return jnp.sum(constraint)

@jit_condition()
def E_noconst(q, lagmt, const_list, const_vals):
    return 0.0

@vmap
@jit_condition()
def mask_to_zero(v, mask):
    return jnp.piecewise(
        v, [mask < 1e-4, mask >= 1e-4], [lambda x: CONST_0, lambda x: v]
    )


@jit_condition(static_argnums=[6])
def E_sr(pos, box, pairs, q, eta, buffer_scales, pbc_flag):
    return 0.0


@jit_condition(static_argnums=[6])
def E_sr2(pos, box, pairs, q, eta, buffer_scales, pbc_flag):
    ds = ds_pairs(pos, box, pairs, pbc_flag)
    etasqrt = jnp.sqrt(2 * (eta[pairs[:, 0]] ** 2 + eta[pairs[:, 1]] ** 2))
    pre_pair = -eta_piecewise(etasqrt, ds) * DIELECTRIC
    pre_self = etainv_piecewise(eta) / (jnp.sqrt(2 * jnp.pi)) * DIELECTRIC
    e_sr_pair = pre_pair * q[pairs[:, 0]] * q[pairs[:, 1]] / ds * buffer_scales
    e_sr_pair = mask_to_zero(e_sr_pair, buffer_scales)
    e_sr_self = pre_self * q * q
    e_sr = jnp.sum(e_sr_pair) + jnp.sum(e_sr_self)
    return e_sr


@jit_condition(static_argnums=[6])
def E_sr3(pos, box, pairs, q, eta, buffer_scales, pbc_flag):
    ds = ds_pairs(pos, box, pairs, pbc_flag)
    etasqrt = jnp.sqrt(
        eta[pairs[:, 0]] ** 2 + eta[pairs[:, 1]] ** 2 + 1e-64
    )  # add eta to avoid division by zero
    epiece = eta_piecewise(etasqrt, ds)
    pre_pair = -epiece * DIELECTRIC
    pre_self = etainv_piecewise(eta) / (jnp.sqrt(2 * jnp.pi)) * DIELECTRIC
    e_sr_pair = pre_pair * q[pairs[:, 0]] * q[pairs[:, 1]] / ds
    e_sr_pair = mask_to_zero(e_sr_pair, buffer_scales)
    e_sr_self = pre_self * q * q
    e_sr = jnp.sum(e_sr_pair) + jnp.sum(e_sr_self)
    return e_sr


@jit_condition()
def E_site(chi, J, q):
    return 0.0


@jit_condition()
def E_site2(chi, J, q):
    ene = (chi * q + 0.5 * J * q**2) * 96.4869  #ev to kj/mol
    return jnp.sum(ene)


@jit_condition()
def E_site3(chi, J, q):
    ene = chi * q +  J* q**2  # kj/mol 
    return jnp.sum(ene)


@jit_condition(static_argnums=[5])
def E_corr(pos, box, pairs, q, kappa, slab_flag=False):
    V = jnp.linalg.det(box)
    Q_tot = jnp.sum(q)
    pre_corr_non = -jnp.pi / (2 * V * kappa**2) * DIELECTRIC
    e_corr_non = pre_corr_non * Q_tot**2
    e_corr = e_corr_non

    if slab_flag:
        Mz = jnp.sum(q * pos[:, 2])
        pre_corr = 2 * jnp.pi / V * DIELECTRIC
        Lz = jnp.linalg.norm(box[3])
        e_corr_slab = pre_corr * (
            Mz**2
            - Q_tot * (jnp.sum(q * pos[:, 2] ** 2))
            - jnp.power(Q_tot, 2) * jnp.power(Lz, 2) / 12
            )
        e_corr += e_corr_slab

    return jnp.sum(e_corr)


@jit_condition(static_argnums=[3])
def ds_pairs(positions, box, pairs, pbc_flag):
    pos1 = positions[pairs[:, 0]]
    pos2 = positions[pairs[:, 1]]
    if pbc_flag is False:
        dr = pos1 - pos2
    else:
        box_inv = jnp.linalg.inv(box)
        dpos = pos1 - pos2
        dpos = dpos.dot(box_inv)
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
        neutral_flag: bool = False,
        slab_flag: bool = False,
        constQ: bool = True,
        pbc_flag: bool = True,
        part_const:bool = True,
        has_aux=False,
    ):
        self.has_aux = has_aux
        self.part_const = part_const
        const_vals = np.array(const_vals)
        #if neutral_flag:
        #    const_vals = const_vals - np.sum(const_vals) / len(const_vals)
        self.const_vals = jnp.array(const_vals)
        assert len(const_list) == len(
            const_vals
        ), "const_list and const_vals must have the same length"
        n_atoms = len(init_q)

        const_mat = np.zeros((len(const_list), n_atoms))
        for ncl, cl in enumerate(const_list):
            const_mat[ncl][cl] = 1
        self.const_mat = jnp.array(const_mat)
       
        if len(const_list) != 0:
            self.const_list = padding_consts(const_list, n_atoms)
            #if fix part charges
            self.all_const_list = self.const_list[jnp.where(self.const_list < n_atoms)]
        else:
            self.const_list = np.array(const_list)
            self.all_const_list =  self.const_list

        all_fix_list = jnp.setdiff1d(jnp.array(range(n_atoms)),self.all_const_list)        
        fix_mat = np.zeros((len(all_fix_list),n_atoms))
        for i, j in enumerate(all_fix_list):
            fix_mat[i][j] = 1
        self.all_fix_list = jnp.array(all_fix_list)
        self.fix_mat = jnp.array(fix_mat)

        self.init_q = jnp.array(init_q)
        self.init_lagmt = jnp.ones((len(const_list),))
        
        self.init_energy = True #init charge by hession inversion method
        self.icount = 0
        self.hessinv_stride = 1
        self.qupdate_stride = 1

        self.damp_mod = damp_mod
        self.neutral_flag = neutral_flag
        self.slab_flag = slab_flag
        self.constQ = constQ
        self.pbc_flag = pbc_flag

        if constQ:
            e_constraint = E_constQ
        elif not part_const:
            e_constraint = E_noconst
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
        def E_full(q, lagmt, chi, J, pos, box, pairs, eta, buffer_scales, mscales):
            if self.part_const:
                e1 = self.e_constraint(q, lagmt, self.const_list, self.const_vals)
            else:
                e1 = 0
            e2 = self.e_sr(pos * 10, box * 10, pairs, q, eta, buffer_scales, self.pbc_flag)
            e3 = self.e_site(chi, J, q)
            e4 = self.coul_energy(pos, box, pairs, q, mscales)
            if not self.neutral_flag:
                e5 = E_corr(
                    pos * 10.0, box * 10.0, pairs, q, self.kappa / 10, self.slab_flag
                )
                return e1 + e2 + e3 + e4 + e5
            else:
                return e1 + e2 + e3 + e4

        grad_E_full = grad(E_full, argnums=(0, 1))

        @jit_condition()
        def E_hession(q, lagmt, chi, J, pos, box, pairs, eta, buffer_scales, mscales):
            h = jacfwd(jacrev(E_full, argnums=(0)))(q, lagmt, chi, J, pos, box, pairs, eta,  buffer_scales, mscales)
            return h

        @jit_condition()
        def get_init_energy(positions, box, pairs, mscales, eta, chi, J, aux=None):
            pos = positions
            buffer_scales = pair_buffer_scales(pairs)

            n_const = len(self.init_lagmt)
            b_vector = jnp.concatenate((-chi, self.const_vals)) #For E_site3
           
            if self.has_aux:
                q = aux["q"][:len(pos)]
                lagmt = aux["q"][len(pos):]
            else:
                q = self.init_q
                lagmt = self.init_lagmt
            B = E_hession(q, lagmt, chi, J, pos, box, pairs, eta, buffer_scales, mscales)
            
            if self.part_const:
                C = jnp.eye(len(q))
                A = C.at[self.all_const_list].set(B[self.all_const_list])
            else:
                A = self.fix_mat

            if self.constQ:
                b_vector = jnp.concatenate((-chi, self.const_vals)) #For E_site3
                b_vector = b_vector.at[self.all_fix_list].set(q[self.all_fix_list]) 

                m0 = jnp.concatenate((A,self.const_mat),axis=0)
                n0 = jnp.concatenate((jnp.transpose(self.const_mat),jnp.zeros((n_const,n_const))),axis=0)

                M = jnp.concatenate((m0,n0),axis=1)
                q_0 = jnp.linalg.solve(M,b_vector)
                q = q_0[:len(pos)]
                lagmt =  q_0[len(pos):]
            else: #constP
                b_vector = jnp.array((-chi  + jnp.sum(self.const_vals * self.const_mat,axis=0) * 96.4869))
                b_vector = b_vector.at[self.all_fix_list].set(q[self.all_fix_list]) 
                M = A
                q = jnp.linalg.solve(M,b_vector)
                q_0 = jnp.concatenate((q.reshape(-1),lagmt.reshape(-1)),axis=0)
               

            energy = E_full(
                q,
                lagmt,
                chi,
                J,
                positions,
                box,
                pairs,
                eta,
                buffer_scales,
                mscales,
            )
            self.init_energy = False
          #  self.icount = self.icount + 1
            if self.has_aux:
                aux["q"] = q_0
                aux["A"] = A
               # aux["m0"] = m0
               # aux["n0"] = n0
                aux["b_vector"] = b_vector
               # aux["init_energy"] = self.init_energy 
                aux["icount"] = self.icount
                return energy, aux
            else:
                return energy


       # @jit_condition()
        def get_proj_grad(func, constraint_matrix, has_aux=False):
            def value_and_proj_grad(*arg, **kwargs):
                value, grad = value_and_grad(func, has_aux=has_aux)(*arg, **kwargs)
                a = jnp.matmul(constraint_matrix, grad.reshape(-1, 1))
                b = jnp.sum(constraint_matrix * constraint_matrix, axis=1, keepdims=True)
                delta_grad = jnp.matmul((a / b).T, constraint_matrix)
                proj_grad = grad - delta_grad.reshape(-1)
                return value, proj_grad
            return value_and_proj_grad

        @jit_condition()
        def get_step_energy(positions, box, pairs, mscales, eta, chi, J, aux=None):
            if self.init_energy:
                if self.has_aux:
                    energy,aux = get_init_energy(positions, box, pairs, mscales, eta, chi, J, aux)
                    return energy, aux
                else:
                    energy = get_init_energy(positions, box, pairs, mscales, eta, chi, J, aux)
                    return energy
            if not self.icount % self.hessinv_stride :
                if self.has_aux:
                    energy,aux = get_init_energy(positions, box, pairs, mscales, eta, chi, J, aux)
                    return energy, aux
                else:
                    energy = get_init_energy(positions, box, pairs, mscales, eta, chi, J, aux)
                    return energy

            func = get_proj_grad(E_full,self.const_mat)
            solver = jaxopt.LBFGS(
                    fun=func,
                    value_and_grad=True,
                    tol=1e-2,
                    )
            pos = positions
            buffer_scales = pair_buffer_scales(pairs)
            if self.has_aux:
                q = aux["q"][:len(pos)]
                lagmt = aux["q"][len(pos):]
            else:
                q = self.init_q
                lagmt = self.init_lagmt

            res = solver.run(
                q,
                lagmt,
                chi,
                J,
                positions,
                box,
                pairs,
                eta,
                buffer_scales,
                mscales,
            )
            q_opt = res.params
            energy = E_full(
                q_opt,
                lagmt,
                chi,
                J,
                positions,
                box,
                pairs,
                eta,
                buffer_scales,
                mscales,
            )
            if self.has_aux:
                aux["q"] = aux['q'].at[:len(pos)].set(q_opt)
                return energy, aux
            else:
                return energy
       # @jit_condition()
        def get_energy(positions, box, pairs, mscales, eta, chi, J, aux=None):
            if self.has_aux :
                if "const_vals" in aux.keys():
                    self.const_vals = aux["const_vals"]
                if "hessinv_stride" in aux.keys(): 
                    self.hessinv_stride = aux["hessinv_stride"]
                if "qupdate_stride" in aux.keys():
                    self.qupdate_stride = aux["qupdate_stride"]
            if not self.icount % self.qupdate_stride :
                if self.has_aux:
                   # aux["q"] = aux['q'].at[:len(pos)].set(q)
                    energy, aux = get_step_energy(positions, box, pairs, mscales, eta, chi, J, aux) 
                    self.icount = self.icount + 1
                    aux["icount"] = self.icount 
                    return energy, aux
                else:
                    self.icount = self.icount + 1
                    energy = get_step_energy(positions, box, pairs, mscales, eta, chi, J ) 
                    return energy

            else:
                self.icount = self.icount + 1
               # print(self.icount)
                pos = positions
                buffer_scales = pair_buffer_scales(pairs)
                if self.has_aux:
                    q = aux["q"][:len(pos)]
                    lagmt = aux["q"][len(pos):]
                else:
                    q = self.init_q
                    lagmt = self.init_lagmt
                energy = E_full(
                    q,
                    lagmt,
                    chi,
                    J,
                    positions,
                    box,
                    pairs,
                    eta,
                    buffer_scales,
                    mscales,
                )

                if self.has_aux:
                    aux = aux
                   # aux["q"] = aux['q'].at[:len(pos)].set(q)
                    aux["icount"] = self.icount 
                    return energy, aux
                else:
                    return energy
       
        return get_energy

