from typing import Iterable

import jax.numpy as jnp
from jax.scipy.special import erfc, erf
import numpy as np

from dmff.utils import pair_buffer_scales, regularize_pairs
from dmff.admp.pme import energy_pme, setup_ewald_parameters, pme_real
from dmff.admp.recip import generate_pme_recip
from dmff.admp.spatial import v_pbc_shift
from dmff.admp.recip import generate_pme_recip, Ck_1
from dmff.admp.pme import DIELECTRIC 


ONE_4PI_EPS0 = DIELECTRIC * 0.1


class LennardJonesForce:
    def __init__(
        self,
        r_switch,
        r_cut,
        map_prm,
        map_nbfix,
        colvmap,
        isSwitch: bool = False,
        isPBC: bool = True,
        isNoCut: bool = False
    ) -> None:
        self.isSwitch = isSwitch
        self.r_switch = r_switch
        self.r_cut = r_cut

        self.map_prm = jnp.array(map_prm)
        self.map_nbfix = map_nbfix
        self.ifPBC = isPBC
        self.ifNoCut = isNoCut
        self.colvmap = colvmap

    def generate_get_energy(self):
        def get_LJ_energy(dr_vec, sig, eps, box):
            if self.ifPBC:
                dr_vec = v_pbc_shift(dr_vec, box, jnp.linalg.inv(box))
            dr_norm = jnp.linalg.norm(dr_vec, axis=1)
            if not self.ifNoCut:
                msk = dr_norm <= self.r_cut
                sig = sig[msk]
                eps = eps[msk]
                dr_norm = dr_norm[msk]

            dr_inv = 1.0 / dr_norm
            sig_dr = sig * dr_inv
            sig_dr6 = jnp.power(sig_dr, 6)
            sig_dr12 = jnp.power(sig_dr6, 2)
            E = 4.0 * eps * (sig_dr12 - sig_dr6)

            if self.isSwitch:
                x = (dr_norm - self.r_switch) / (self.r_cut - self.r_switch)
                S = 1 - 6. * x ** 5 + 15. * x ** 4 - 10. * x ** 3
                jnp.where(dr_norm > self.r_switch, E, E * S)
            
            return E

        def get_energy(positions, box, pairs, epsilon, sigma, epsfix, sigfix, mscales):
            
            pairs = regularize_pairs(pairs)
            mask = pair_buffer_scales(pairs)
            map_prm = self.map_prm

            eps_m1 = jnp.repeat(epsilon.reshape((-1, 1)), epsilon.shape[0], axis=1)
            eps_m2 = eps_m1.T
            eps_mat = jnp.sqrt(eps_m1 * eps_m2)
            sig_m1 = jnp.repeat(sigma.reshape((-1, 1)), sigma.shape[0], axis=1)
            sig_m2 = sig_m1.T
            sig_mat = (sig_m1 + sig_m2) * 0.5

            eps_mat = eps_mat.at[self.map_nbfix[:, 0], self.map_nbfix[:, 1]].set(epsfix)
            eps_mat = eps_mat.at[self.map_nbfix[:, 1], self.map_nbfix[:, 0]].set(epsfix)
            sig_mat = sig_mat.at[self.map_nbfix[:, 0], self.map_nbfix[:, 1]].set(sigfix)
            sig_mat = sig_mat.at[self.map_nbfix[:, 1], self.map_nbfix[:, 0]].set(sigfix)

            colv_pair = self.colvmap[pairs[:,0],pairs[:,1]]
            mscale_pair = mscales[colv_pair-1] # in mscale vector, the 0th item is 1-2 scale, the 1st item is 1-3 scale, etc...

            dr_vec = positions[pairs[:, 0]] - positions[pairs[:, 1]]
            prm_pair0 = map_prm[pairs[:, 0]]
            prm_pair1 = map_prm[pairs[:, 1]]
            eps = eps_mat[prm_pair0, prm_pair1]
            sig = sig_mat[prm_pair0, prm_pair1]

            eps_scale = eps * mscale_pair

            E_inter = get_LJ_energy(dr_vec, sig, eps_scale, box)

            return jnp.sum(E_inter * mask)

        return get_energy


class LennardJonesLongRangeForce:
    def __init__(
        self,
        r_cut: float,
        map_prm: Iterable[int],
        map_nbfix: Iterable[int],
        countMat: np.ndarray
    ):
        self.r_cut = r_cut
        self.map_prm = map_prm
        self.map_nbfix = map_nbfix
        self.countMat = countMat
        self.numParticles = len(map_prm)
    
    def generate_get_energy(self):
        def get_energy(box, epsilon, sigma, epsfix, sigfix):

            eps_m1 = jnp.repeat(epsilon.reshape((-1, 1)), epsilon.shape[0], axis=1)
            eps_m2 = eps_m1.T
            eps_mat = jnp.sqrt(eps_m1 * eps_m2)
            sig_m1 = jnp.repeat(sigma.reshape((-1, 1)), sigma.shape[0], axis=1)
            sig_m2 = sig_m1.T
            sig_mat = (sig_m1 + sig_m2) * 0.5

            eps_mat = eps_mat.at[self.map_nbfix[:, 0], self.map_nbfix[:, 1]].set(epsfix)
            eps_mat = eps_mat.at[self.map_nbfix[:, 1], self.map_nbfix[:, 0]].set(epsfix)
            sig_mat = sig_mat.at[self.map_nbfix[:, 0], self.map_nbfix[:, 1]].set(sigfix)
            sig_mat = sig_mat.at[self.map_nbfix[:, 1], self.map_nbfix[:, 0]].set(sigfix)

            volume = jnp.linalg.det(box)

            c6Mat = 4 * eps_mat * jnp.power(sig_mat, 6)
            c6 = jnp.sum(c6Mat * self.countMat) / jnp.sum(self.countMat)
            dispCorrEnergy = -2 / 3 * jnp.pi * self.numParticles * self.numParticles / volume * c6 / jnp.power(self.r_cut, 3)
            return dispCorrEnergy
        
        return get_energy
    

class CoulNoCutoffForce:
    # E=\frac{{q}_{1}{q}_{2}}{4\pi\epsilon_0\epsilon_1 r}

    def __init__(self, map_prm, colvmap, epsilon_1=1.0) -> None:

        self.eps_1 = epsilon_1
        self.map_prm = map_prm
        self.colvmap = colvmap

    def generate_get_energy(self):
        def get_coul_energy(dr_vec, chrgprod, box):
            dr_norm = jnp.linalg.norm(dr_vec, axis=1)

            dr_inv = 1.0 / dr_norm
            E = chrgprod * ONE_4PI_EPS0 / self.eps_1 * dr_inv

            return E

        def get_energy(positions, box, pairs, charges, mscales):
            
            pairs = regularize_pairs(pairs)
            mask = pair_buffer_scales(pairs)
            map_prm = jnp.array(self.map_prm)

            colv_pair = self.colvmap[pairs[:,0],pairs[:,1]]
            mscale_pair = mscales[colv_pair-1]

            chrg_map0 = map_prm[pairs[:, 0]]
            chrg_map1 = map_prm[pairs[:, 1]]
            charge0 = charges[chrg_map0]
            charge1 = charges[chrg_map1]
            chrgprod = charge0 * charge1
            chrgprod_scale = chrgprod * mscale_pair
            dr_vec = positions[pairs[:, 0]] - positions[pairs[:, 1]]

            E_inter = get_coul_energy(dr_vec, chrgprod_scale, box)

            return jnp.sum(E_inter * mask) 

        return get_energy


class CoulReactionFieldForce:
    # E=\frac{{q}_{1}{q}_{2}}{4\pi\epsilon_0\epsilon_1}\left(\frac{1}{r}+{k}_{\mathit{rf}}{r}^{2}-{c}_{\mathit{rf}}\right)
    def __init__(
        self,
        r_cut,
        map_prm,
        colvmap,
        epsilon_1=1.0,
        epsilon_solv=78.5,
        isPBC=True,
    ) -> None:

        self.r_cut = r_cut
        self.krf = (1.0 / r_cut ** 3) * (epsilon_solv - 1) / (2.0 * epsilon_solv + 1)
        self.crf = (1.0 / r_cut) * 3.0 * epsilon_solv / (2.0 * epsilon_solv + 1)
        self.exp_solv = epsilon_solv
        self.eps_1 = epsilon_1
        self.map_prm = map_prm
        self.colvmap = colvmap
        self.ifPBC = isPBC

    def generate_get_energy(self):
        def get_rf_energy(dr_vec, chrgprod, box):
            if self.ifPBC:
                dr_vec = v_pbc_shift(dr_vec, box, jnp.linalg.inv(box))
            dr_norm = jnp.linalg.norm(dr_vec, axis=1)
            chrgprod = chrgprod[dr_norm <= self.r_cut]
            dr_norm = dr_norm[dr_norm <= self.r_cut]

            dr_inv = 1.0 / dr_norm
            E = (
                chrgprod
                * ONE_4PI_EPS0
                / self.eps_1
                * (dr_inv + self.krf * dr_norm * dr_norm - self.crf)
            )

            return E

        def get_energy(positions, box, pairs, charges, mscales):
            
            pairs = regularize_pairs(pairs)
            mask = pair_buffer_scales(pairs)

            colv_pair = self.colvmap[pairs[:,0],pairs[:,1]]
            mscale_pair = mscales[colv_pair-1]

            chrg_map0 = self.map_prm[pairs[:, 0]]
            chrg_map1 = self.map_prm[pairs[:, 1]]
            charge0 = charges[chrg_map0]
            charge1 = charges[chrg_map1]
            chrgprod = charge0 * charge1
            chrgprod_scale = chrgprod * mscale_pair
            dr_vec = positions[pairs[:, 0]] - positions[pairs[:, 1]]

            E_inter = get_rf_energy(dr_vec, chrgprod_scale, box)

            return jnp.sum(E_inter * mask)

        return get_energy


class CoulombPMEForce:
    def __init__(
        self,
        r_cut: float,
        map_prm: Iterable[int],
        colv_map: np.ndarray,
        ethresh: float,
        pme_order: int = 6
    ):
        self.r_cut = r_cut
        self.map_prm = map_prm
        self.ethresh = ethresh
        self.pme_order = pme_order
        self.colv_map = colv_map
        self.lmax = 0

    def generate_get_energy(self):
        
        def get_energy(positions, box, pairs, charges, mscales):
            # kappa, K1, K2, K3 = setup_ewald_parameters(self.r_cut, self.ethresh, box)
            # print(kappa, K1, K2, K3)
            kappa = 3.458910000000000 / 10
            K1 = 33
            K2 = 33
            K3 = 33
            self.r_cut = 1.0

            atomCharges = charges[self.map_prm[np.arange(positions.shape[0])]]
            atomChargesT = atomCharges.reshape(-1, 1)
            # # print(mscales)

            bufscales = pair_buffer_scales(pairs)

            # mscales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=jnp.float32) # debug
            colv_pair = self.colv_map[pairs[:, 0], pairs[:, 1]]
            mscale_pair = mscales[colv_pair-1]


            
            dr_vec = positions[pairs[:, 0]] - positions[pairs[:, 1]]
            dr_vec = v_pbc_shift(dr_vec, box, jnp.linalg.inv(box))
            
            dr_norm = jnp.linalg.norm(dr_vec, axis=1)

            
            # msk = dr_norm <= self.r_cut
            
            # dr_norm = dr_norm[msk]
            # erfc_dr = erfc(dr_norm * kappa)

            chgprod = atomCharges[pairs[:, 0]] * atomCharges[pairs[:, 1]]
            erf_dr = erf(dr_norm * kappa * 10)
            ene = chgprod * (mscale_pair - erf_dr) / dr_norm * bufscales * ONE_4PI_EPS0
            print(jnp.sum(ene))

            # chgprod = chgprod[msk]
            # mscale_pair = mscale_pair[msk]
            # ene_vec = chgprod * erfc_dr / dr_norm * ONE_4PI_EPS0 * mscale_pair
            # short_range_ene = jnp.sum(ene_vec)
            # print("Short Range:", short_range_ene)
            # self_corr_ene = -ONE_4PI_EPS0 * kappa / jnp.sqrt(jnp.pi) * jnp.sum(atomCharges ** 2)
            # # 1-2, 1-3 excl
            # excl_msk = jnp.logical_and(
            #     jnp.logical_or(
            #         jnp.logical_or(colv_pair == 1, colv_pair == 2),
            #         colv_pair == 3
            #     ),
            #     bufscales
            # )
            # excl_ene_vec = -chgprod * (1 - erfc_dr) / dr_norm * ONE_4PI_EPS0
            # excl_ene = jnp.sum(excl_ene_vec[excl_msk])

            # self_corr_ene += excl_ene
            # print("Self Corr:", self_corr_ene)
            # # print(jnp.mean(erfc_dr))

            # # # dr_shift = erfc(kappa * self.r_cut) / self.r_cut
            # # # ewald_shift = jnp.sum(chgprod * dr_shift * ONE_4PI_EPS0 * mscale_pair)
            # # # print("Shift:", ewald_shift)
            # # # short_range_ene -= ewald_shift
            # dipole = jnp.sum(atomChargesT * positions, axis=0)
            # # # ene_dipole = jnp.dot(dipole, dipole.T) * ONE_4PI_EPS0
            # # # print(ene_dipole)
            # print(-jnp.sum(dipole ** 2) * ONE_4PI_EPS0 * 3 * jnp.pi / 2 / jnp.linalg.det(box))
            # # # print(atomCharges)

            pme_recip_fn = generate_pme_recip(
                Ck_fn=Ck_1,
                kappa=kappa,
                gamma=False,
                pme_order=self.pme_order,
                K1=K1,
                K2=K2,
                K3=K3,
                lmax=self.lmax,
            )
            # ene_recip = pme_recip_fn(positions, box, atomChargesT) * 0.1
            # print("Coul. recip:", ene_recip)

            # return short_range_ene + self_corr_ene + ene_recip


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
                self.colv_map,
                None,
                pme_recip_fn,
                kappa,
                K1,
                K2,
                K3,
                self.lmax,
                False,
            )

        return get_energy
