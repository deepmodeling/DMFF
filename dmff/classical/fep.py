from typing import Callable, Union, Optional, Iterable, Tuple

import numpy as np
import jax.numpy as jnp
from jax.scipy.special import erf

from ..common.constants import DIELECTRIC
from ..admp.spatial import v_pbc_shift
from ..admp.recip import generate_pme_recip, Ck_1
from ..utils import regularize_pairs, pair_buffer_scales


ONE_4PI_EPS0 = DIELECTRIC * 0.1


def mk_softcore_fn(
    sc_alpha: float,
    sc_sigma: float,
    sc_power: int = 1,
    sc_r_power: int = 6,
    if_state_A: bool = True,
) -> Callable:
    """
    Make softcore function
    """
    assert sc_r_power == 6, f"sc_r_power must be 6"
    assert sc_power == 1 or sc_power == 2, f"sc_power must be 1 or 2"
    sig_pow = jnp.power(sc_sigma, sc_r_power)
    
    def softcore_fn(distances, fep_lambda: float):
        dist_pow = jnp.power(distances, sc_r_power)
        lmd = fep_lambda if if_state_A else 1 - fep_lambda
        lmd_pow = jnp.power(lmd, sc_power)
        shift_dist = jnp.power(sc_alpha * sig_pow * lmd_pow + dist_pow, 1 / 6)
        return shift_dist
    
    return softcore_fn


class LennardJonesFreeEnergyForce:
    def __init__(
        self,
        r_switch,
        r_cut,
        map_prm,
        map_nbfix,
        isSwitch: bool = False,
        isPBC: bool = True,
        isNoCut: bool = False,
        feLambda: float = 0.0,
        ifStateA: bool = True,
        coupleMask: Optional[Iterable[bool]] = None,
        useSoftCore: bool = False,
        sc_alpha: float = 0.0,
        sc_sigma: float = 0.0,
        sc_power: int = 1,
        sc_r_power: int = 6
    ) -> None:
        self.isSwitch = isSwitch
        self.r_switch = r_switch
        self.r_cut = r_cut

        self.map_prm = map_prm
        self.map_nbfix = map_nbfix
        self.ifPBC = isPBC
        self.ifNoCut = isNoCut
        assert not isNoCut, f"NoCut is not supported for free energy calculations"

        # free energy
        self.feLambda = feLambda
        self.ifStateA = ifStateA
        self.coupleMask = coupleMask
        
        self.useSoftCore = useSoftCore
        if self.useSoftCore:
            self.scFuncStateA = mk_softcore_fn(sc_alpha, sc_sigma, sc_power, sc_r_power, True)
            self.scFuncStateB = mk_softcore_fn(sc_alpha, sc_sigma, sc_power, sc_r_power, False)

    def generate_get_energy(self):

        def get_energy(positions, box, pairs, epsilon, sigma, epsfix, sigfix, mscales, lambda_):

            pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))
            bufScales = pair_buffer_scales(pairs[:, :2])

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

            colv_pair = pairs[:, 2]
            mscale_pair = mscales[colv_pair-1] # in mscale vector, the 0th item is 1-2 scale, the 1st item is 1-3 scale, etc...

            dr_vec = positions[pairs[:, 0]] - positions[pairs[:, 1]]
            prm_pair0 = self.map_prm[pairs[:, 0]]
            prm_pair1 = self.map_prm[pairs[:, 1]]
            eps = eps_mat[prm_pair0, prm_pair1]
            sig = sig_mat[prm_pair0, prm_pair1]

            eps_scale = eps * mscale_pair

            if self.ifPBC:
                dr_vec = v_pbc_shift(dr_vec, box, jnp.linalg.inv(box + jnp.eye(3) * 1e-36))
            
            dr_norm = jnp.linalg.norm(dr_vec, axis=1)

            def _calc_energy(dr_norm):
                dr_inv = 1.0 / dr_norm
                sig_dr = sig * dr_inv
                sig_dr6 = jnp.power(sig_dr, 6)
                sig_dr12 = jnp.power(sig_dr6, 2)
                E = 4.0 * eps_scale * (sig_dr12 - sig_dr6)

                if self.isSwitch:
                    x = (dr_norm - self.r_switch) / (self.r_cut - self.r_switch)
                    S = 1 - 6. * x ** 5 + 15. * x ** 4 - 10. * x ** 3
                    jnp.where(dr_norm > self.r_switch, E, E * S)
                return E
            
            if self.coupleMask is not None:
                # Couple mode, designed for solvation free energy calculations
                # StateA is coupled & StateB is decoupled
                # V(l) = (1-l) V(stateA) + l * V(stateB)
                
                # If two atoms in pairs[x] is belonged to different coupled state,
                # then mask[x] = False
                mask = jnp.logical_not(jnp.logical_xor(
                    self.coupleMask[pairs[:, 0]],
                    self.coupleMask[pairs[:, 1]]
                ))
                
                if self.useSoftCore:
                    # Only pairs with zero-endstate are soft-cored
                    dr_norm_couple = self.scFuncStateA(dr_norm, lambda_)
                    dr_norm_couple = dr_norm * mask + dr_norm_couple * jnp.logical_not(mask)
                    dr_norm_decouple = self.scFuncStateB(dr_norm, lambda_)
                    dr_norm_decouple = dr_norm * mask + dr_norm_couple * jnp.logical_not(mask)
                    E_couple = _calc_energy(dr_norm_couple)
                    E_decouple = _calc_energy(dr_norm_decouple) * mask
                else:
                    E_couple = _calc_energy(dr_norm)
                    E_decouple = E_couple * mask

                E_sum = (1 - lambda_) * E_couple + lambda_ * E_decouple
                return jnp.sum(E_sum * bufScales)
            else:
                # TODO : Implement this
                raise NotImplementedError()
                # # Designed for Free Energy Perturbation
                # if self.useSoftCore:
                #     scMask = np.logical_and(eps > 0, sig > 0)[rcut_msk]
                #     if self.ifStateA:
                #         dr_norm_sc = self.scFuncStateA(dr_norm, self.feLambda)
                #     else:
                #         dr_norm_sc = self.scFuncStateB(dr_norm, self.feLambda)
                #     dr_norm = dr_norm * scMask + dr_norm_sc * np.logical_not(scMask)
                # E_sum = _calc_energy(dr_norm)
                # if self.ifStateA:
                #     return jnp.sum(E_sum * bufScales) * (1 - self.feLambda)
                # else:
                #     return jnp.sum(E_sum * bufScales) * self.feLambda
    
        return get_energy


class LennardJonesLongRangeFreeEnergyForce:
    def __init__(
        self,
        r_cut: float,
        map_prm: Iterable[int],
        map_nbfix: Iterable[int],
        countMat: np.ndarray,
        feLambda: float,
        ifStateA: bool = True,
        coupleMask: Optional[Iterable[bool]] = None,
    ):
        self.r_cut = r_cut
        self.map_prm = map_prm
        self.map_nbfix = map_nbfix
        self.countMat = countMat
        self.feLambda = feLambda
        self.ifStateA = ifStateA
        self.coupleMask = coupleMask
        self.numParticles = len(map_prm)

        if self.coupleMask is not None:
            countMatDecouple = self.countMat.copy()
            typesA, countsA = np.unique(self.map_prm[self.coupleMask], return_counts=True)
            typesB, countsB = np.unique(self.map_prm[np.logical_not(self.coupleMask)], return_counts=True)
            for i in range(len(typesA)):
                for j in range(len(typesB)):
                    cnt = countsA[i] * countsB[j]
                    typ1, typ2 = min(typesA[i], typesB[j]), max(typesA[i], typesB[j])
                    countMatDecouple[typ1, typ2] -= cnt
            self.countMatDecouple = countMatDecouple
    
    def generate_get_energy(self):

        def get_energy(box, epsilon, sigma, epsfix, sigfix, lambda_):

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
            c6Mat = 4 * eps_mat * jnp.power(sig_mat, 6)

            volume = jnp.linalg.det(box)

            def _calc_energy(countMat, avgMat):
                c6 = jnp.sum(c6Mat * countMat) / jnp.sum(avgMat)
                dispCorrEnergy = -2 / 3 * jnp.pi * self.numParticles * self.numParticles / volume * c6 / jnp.power(self.r_cut, 3)
                return dispCorrEnergy
            
            if self.coupleMask is not None:
                dispCouple = _calc_energy(self.countMat, self.countMat)
                dispDecouple = _calc_energy(self.countMatDecouple, self.countMat)
                dispEnergy = (1 - lambda_) * dispCouple + lambda_ * dispDecouple
                return dispEnergy
            else:
                raise NotImplementedError()
                # if self.ifStateA:
                #     return (1 - self.feLambda) * _calc_energy(self.countMat, self.countMat)
                # else:
                #     return self.feLambda * _calc_energy(self.countMat, self.countMat)

        return get_energy


class CoulombPMEFreeEnergyForce:
    def __init__(
        self,
        r_cut: float,
        map_prm: Iterable[int],
        kappa: float,
        K: Tuple[int, int, int],
        feLambda: float,
        ifStateA: bool = True,
        coupleMask: Optional[Iterable[bool]] = None,
        useSoftCore: bool = False,
        sc_alpha: float = 0.0,
        sc_sigma: float = 0.0,
        sc_power: int = 1,
        sc_r_power: int = 6,
        pme_order: int = 6,
    ):
        self.r_cut = r_cut
        self.map_prm = map_prm
        self.kappa = kappa # in nm -1
        self.K1, self.K2, self.K3 = K[0], K[1], K[2]
        # free energy
        self.feLambda = feLambda
        self.ifStateA = ifStateA
        self.coupleMask = coupleMask
        
        self.useSoftCore = useSoftCore
        assert not self.useSoftCore, "Not implemented yet" 
        if self.useSoftCore:
            self.scFuncStateA = mk_softcore_fn(sc_alpha, sc_sigma, sc_power, sc_r_power, True)
            self.scFuncStateB = mk_softcore_fn(sc_alpha, sc_sigma, sc_power, sc_r_power, False)
        
        self.lmax = 0
        self.pme_order = pme_order
    
    def generate_get_energy(self):
        
        def get_energy(positions, box, pairs, charges, mscales, lambda_): 
            pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))
            bufScales = pair_buffer_scales(pairs[:, :2])
            dr_vec = positions[pairs[:, 0]] - positions[pairs[:, 1]]
            dr_vec = v_pbc_shift(dr_vec, box, jnp.linalg.inv(box + jnp.eye(3) * 1e-36))
            dr_norm = jnp.linalg.norm(dr_vec, axis=1)

            atomCharges = charges[self.map_prm[np.arange(positions.shape[0])]]
            chgprod = atomCharges[pairs[:, 0]] * atomCharges[pairs[:, 1]]

            colv_pair = pairs[:, 2]
            mscale_pair = mscales[colv_pair-1]

            pme_recip_fn = generate_pme_recip(
                Ck_fn=Ck_1,
                kappa=self.kappa / 10,
                gamma=False,
                pme_order=self.pme_order,
                K1=self.K1,
                K2=self.K2,
                K3=self.K3,
                lmax=self.lmax,
            )
            ene_couple_recip = pme_recip_fn(positions * 10, box * 10, jnp.reshape(atomCharges, (-1, 1)))

            ene_self = jnp.sum(-ONE_4PI_EPS0 * self.kappa / jnp.sqrt(jnp.pi) * jnp.sum(atomCharges ** 2))

            if self.coupleMask is not None:
                # Couple mode, designed for solvation free energy calculations
                # StateA is coupled & StateB is decoupled
                # V(l) = (1-l) V(stateA) + l * V(stateB)
                
                # If two atoms in pairs[x] is belonged to different coupled state,
                # then fe_mask[x] = False and sc_mask[x] = True
                sc_mask = jnp.logical_xor(
                    self.coupleMask[pairs[:, 0]],
                    self.coupleMask[pairs[:, 1]]
                )
                fe_mask = jnp.logical_not(sc_mask)

                ene_decoup_recip = pme_recip_fn(
                    positions * 10,
                    box * 10,
                    jnp.reshape(atomCharges * jnp.logical_not(self.coupleMask), (-1, 1))
                )
                ene_corr_recip = pme_recip_fn(
                    positions * 10,
                    box * 10,
                    jnp.reshape(atomCharges * self.coupleMask, (-1, 1))
                )

                ene_recip = (1 - lambda_) * ene_couple_recip + lambda_ * (ene_decoup_recip + ene_corr_recip)

                if not self.useSoftCore:      
                    erf_dr = erf(self.kappa * dr_norm)
                    tmp = chgprod / dr_norm * ONE_4PI_EPS0
                    ene_couple_vec = tmp * (mscale_pair - erf_dr)
                    ene_decoup_vec = ene_couple_vec * fe_mask
                else:
                    dr_norm_couple = self.scFuncStateA(dr_norm, lambda_) * sc_mask + dr_norm * fe_mask
                    dr_norm_decoup = self.scFuncStateB(dr_norm, lambda_) * sc_mask + dr_norm * fe_mask
                    erf_dr_couple = erf(self.kappa * dr_norm_couple)
                    erf_dr_decoup = erf(self.kappa * dr_norm_decoup)
                    ene_couple_vec = chgprod / dr_norm_couple * ONE_4PI_EPS0 * (mscale_pair - erf_dr_couple)
                    ene_decoup_vec = chgprod / dr_norm_decoup * ONE_4PI_EPS0 * (mscale_pair - erf_dr_decoup) * fe_mask
                
                ene_short_vec = (1 - lambda_) * ene_couple_vec + lambda_ * ene_decoup_vec
                ene_short = jnp.sum(ene_short_vec * bufScales)
                return ene_short + ene_self + ene_recip
            else:
                # TODO : Implement this
                raise NotImplementedError()
                # sc_mask = jnp.abs(atomCharges) < 1e-5
                # if self.ifStateA:
                #     dr_norm = self.scFuncStateA(dr_norm, self.feLambda) * sc_mask + dr_norm * jnp.logical_not(sc_mask)
                # else:
                #     dr_norm = self.scFuncStateB(dr_norm, self.feLambda) * sc_mask + dr_norm * jnp.logical_not(sc_mask)

        return get_energy






