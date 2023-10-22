import numpy as np
import jax.numpy as jnp
from ..common.constants import DIELECTRIC
from jax import grad, vmap
from ..classical.inter import CoulNoCutoffForce, CoulombPMEForce
from typing import Tuple, List
from ..settings import PRECISION

if PRECISION == "double":
    CONST_0 = jnp.array(0, dtype=jnp.float64)
    CONST_1 = jnp.array(1, dtype=jnp.float64)
else:
    CONST_0 = jnp.array(0, dtype=jnp.float32)
    CONST_1 = jnp.array(1, dtype=jnp.float32)

try:
    import jaxopt
except ImportError:
    print("jaxopt not found, QEQ cannot be used.")
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
    etasqrt = jnp.sqrt(eta[pairs[:, 0]] ** 2 + eta[pairs[:, 1]] ** 2)
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


@jit_condition
def E_CoulNocutoff(pos, box, pairs, q, ds):
    e = q[pairs[:, 0]] * q[pairs[:, 1]] / ds * DIELECTRIC
    return jnp.sum(e)


@jit_condition
def E_Coul(pos, box, pairs, q, ds):
    return 0.0


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
    ds = jnp.linalg.norm(dr, axis=1)
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
    ):
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
            force = CoulombPMEForce(r_cut, kappa, K)
            self.kappa = kappa
        else:
            force = CoulNoCutoffForce()
            self.kappa = 1.0
        self.coul_energy = force.generate_get_energy()

    def generate_get_energy(self):
        @jit_condition()
        def E_full(q, lagmt, chi, J, pos, box, pairs, eta, ds, buffer_scales, mscales):
            e1 = self.e_constraint(q, lagmt, self.const_list, self.const_vals)
            e2 = self.e_sr(pos * 10, box * 10, pairs, q, eta, ds * 10, buffer_scales)
            e3 = self.e_site(chi, J, q)
            e4 = self.coul_energy(pos, box, pairs, q, mscales)
            e5 = E_corr(
                pos * 10.0, box * 10.0, pairs, q, self.kappa / 10, self.neutral_flag
            )
            return e1 + e2 + e3 + e4 + e5

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

        def get_energy(positions, box, pairs, mscales, eta, chi, J):
            pos = positions
            ds = ds_pairs(pos, box, pairs, self.pbc_flag)
            buffer_scales = pair_buffer_scales(pairs)

            n_const = len(self.init_lagmt)
            b_value = jnp.concatenate((self.init_q, self.init_lagmt))
            rf = jaxopt.ScipyRootFinding(
                optimality_fun=E_grads, method="hybr", jit=False, tol=1e-10
            )
            b_0, _ = rf.run(
                b_value, chi, J, positions, box, pairs, eta, ds, buffer_scales, mscales
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
            return energy

        return get_energy
