from typing import Iterable, Tuple, Optional
import jax
import jax.numpy as jnp
import numpy as np

from ..utils import pair_buffer_scales, regularize_pairs
from ..admp.pme import energy_pme
from ..admp.recip import generate_pme_recip
from ..admp.spatial import v_pbc_shift
from ..admp.recip import generate_pme_recip, Ck_1
from ..admp.pme import DIELECTRIC 


ONE_4PI_EPS0 = DIELECTRIC * 0.1


class LennardJonesForce:
    def __init__(
        self,
        r_switch,
        r_cut,
        map_prm,
        map_nbfix,
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

    def generate_get_energy(self):
        def get_LJ_energy(dr_vec, sig, eps, box):
            if self.ifPBC:
                dr_vec = v_pbc_shift(dr_vec, box, jnp.linalg.inv(box + jnp.eye(3) * 1e-36))
            dr_norm = jnp.linalg.norm(dr_vec, axis=1)

            dr_inv = 1.0 / dr_norm
            sig_dr = sig * dr_inv
            sig_dr6 = jnp.power(sig_dr, 6)
            sig_dr12 = jnp.power(sig_dr6, 2)
            E = 4.0 * eps * (sig_dr12 - sig_dr6)

            if self.isSwitch:
                x = (dr_norm - self.r_switch) / (self.r_cut - self.r_switch)
                S = 1 - 6. * x ** 5 + 15. * x ** 4 - 10. * x ** 3
                E = jnp.where(dr_norm > self.r_switch, E * S, E)
                E = jnp.where(dr_norm >= self.r_cut, 0., E)
            
            return E

        def get_energy(positions, box, pairs, epsilon, sigma, epsfix, sigfix, mscales, aux=None):
            
            pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))
            mask = pair_buffer_scales(pairs[:, :2])
            map_prm = self.map_prm

            eps_m1 = jnp.repeat(epsilon.reshape((-1, 1)), epsilon.shape[0], axis=1)
            eps_m2 = eps_m1.T
            eps_mat = jnp.sqrt(eps_m1 * eps_m2 + 1e-32)
            sig_m1 = jnp.repeat(sigma.reshape((-1, 1)), sigma.shape[0], axis=1)
            sig_m2 = sig_m1.T
            sig_mat = (sig_m1 + sig_m2) * 0.5
            
            for _map in self.map_nbfix:
                eps_mat = eps_mat.at[_map[0],_map[1]].set(epsfix[_map[2]])
                eps_mat = eps_mat.at[_map[1],_map[0]].set(epsfix[_map[2]])
                sig_mat = sig_mat.at[_map[0],_map[1]].set(sigfix[_map[2]])
                sig_mat = sig_mat.at[_map[1],_map[0]].set(sigfix[_map[2]])

            colv_pair = pairs[:, 2]
            mscale_pair = mscales[colv_pair-1] # in mscale vector, the 0th item is 1-2 scale, the 1st item is 1-3 scale, etc...

            dr_vec = positions[pairs[:, 0]] - positions[pairs[:, 1]]
            prm_pair0 = map_prm[pairs[:, 0]]
            prm_pair1 = map_prm[pairs[:, 1]]
            eps = eps_mat[prm_pair0, prm_pair1]
            sig = sig_mat[prm_pair0, prm_pair1]

            eps_scale = eps * mscale_pair

            E_inter = get_LJ_energy(dr_vec, sig, eps_scale, box)
            if aux is None:
                return jnp.sum(E_inter * mask)
            else:
                return jnp.sum(E_inter * mask), aux

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

            eps_mat = eps_mat.at[self.map_nbfix[:, 0], self.map_nbfix[:, 1]].set(epsfix[self.map_nbfix[:, 2]])
            sig_mat = sig_mat.at[self.map_nbfix[:, 0], self.map_nbfix[:, 1]].set(sigfix[self.map_nbfix[:, 2]])

            volume = jnp.linalg.det(box)

            c6Mat = 4 * eps_mat * jnp.power(sig_mat, 6)
            c6 = jnp.sum(c6Mat * self.countMat) / jnp.sum(self.countMat)
            dispCorrEnergy = -2 / 3 * jnp.pi * self.numParticles * self.numParticles / volume * c6 / jnp.power(self.r_cut, 3)
            return dispCorrEnergy
        
        return get_energy
    

class CoulNoCutoffForce:
    # E=\frac{{q}_{1}{q}_{2}}{4\pi\epsilon_0\epsilon_1 r}

    def __init__(self, init_charges, epsilon_1=1.0, topology_matrix=None) -> None:

        self.init_charges = init_charges
        self.eps_1 = epsilon_1
        self.top_mat = topology_matrix

    def generate_get_energy(self):
        def get_coul_energy(dr_vec, chrgprod, box):
            dr_norm = jnp.linalg.norm(dr_vec, axis=1)

            dr_inv = 1.0 / dr_norm
            E = chrgprod * ONE_4PI_EPS0 / self.eps_1 * dr_inv

            return E

        def get_energy_kernel(positions, box, pairs, charges, mscales):
            pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))
            mask = pair_buffer_scales(pairs[:, :2])
            cov_pair = pairs[:, 2]
            mscale_pair = mscales[cov_pair-1]

            charge0 = charges[pairs[:, 0]]
            charge1 = charges[pairs[:, 1]]
            chrgprod = charge0 * charge1
            chrgprod_scale = chrgprod * mscale_pair
            dr_vec = positions[pairs[:, 0]] - positions[pairs[:, 1]]

            E_inter = get_coul_energy(dr_vec, chrgprod_scale, box)

            return jnp.sum(E_inter * mask)
        
        if self.top_mat is None:
            def get_energy(positions, box, pairs, mscales):
                return get_energy_kernel(positions, box, pairs, self.init_charges, mscales)
        else:
            def get_energy(positions, box, pairs, bcc, mscales):
                charges = self.init_charges + jnp.dot(self.top_mat, bcc).flatten()
                return get_energy_kernel(positions, box, pairs, charges, mscales)
        return get_energy
        
    def generate_esp(self):

        def esp_kernel(position, grid, charge):
            dist = jnp.linalg.norm(position - grid + 1e-16)
            oneR = 1. / dist
            return ONE_4PI_EPS0 * charge * oneR
        
        esp_grid_kernel = jax.vmap(esp_kernel, in_axes=(0, None, 0))

        def esp_grid(positions, grid, charges):
            return jnp.sum(esp_grid_kernel(positions, grid, charges))
        
        esp_all = jax.vmap(esp_grid, in_axes=(None, 0, None))
        
        if self.top_mat is None:
            def get_esp(positions, grids):
                charges = self.init_charges
                return esp_all(positions, grids, charges).ravel()
        else:
            def get_esp(positions, grids, bcc):
                charges = self.init_charges + jnp.dot(self.top_mat, bcc).flatten()
                return esp_all(positions, grids, charges).ravel()
        
        return get_esp
            
            


class CoulReactionFieldForce:
    # E=\frac{{q}_{1}{q}_{2}}{4\pi\epsilon_0\epsilon_1}\left(\frac{1}{r}+{k}_{\mathit{rf}}{r}^{2}-{c}_{\mathit{rf}}\right)
    def __init__(
        self,
        r_cut,
        init_charges,
        epsilon_1=1.0,
        epsilon_solv=78.5,
        isPBC=True,
        topology_matrix=None
    ) -> None:

        self.init_charges = init_charges
        self.r_cut = r_cut
        self.krf = (1.0 / r_cut ** 3) * (epsilon_solv - 1) / (2.0 * epsilon_solv + 1)
        self.crf = (1.0 / r_cut) * 3.0 * epsilon_solv / (2.0 * epsilon_solv + 1)
        self.exp_solv = epsilon_solv
        self.eps_1 = epsilon_1
        self.ifPBC = isPBC
        self.top_mat = topology_matrix

    def generate_get_energy(self):
        def get_rf_energy(dr_vec, chrgprod, box):
            if self.ifPBC:
                dr_vec = v_pbc_shift(dr_vec, box, jnp.linalg.inv(box + jnp.eye(3) * 1e-36))
            dr_norm = jnp.linalg.norm(dr_vec, axis=1)

            dr_inv = 1.0 / dr_norm

            E = (
                chrgprod
                * ONE_4PI_EPS0
                / self.eps_1
                * (dr_inv + self.krf * dr_norm * dr_norm - self.crf)
            )

            return E

        def get_energy_kernel(positions, box, pairs, charges, mscales):
            pairs = pairs.at[:, :2].set(regularize_pairs(pairs[:, :2]))
            mask = pair_buffer_scales(pairs[:, :2])

            colv_pair = pairs[:, 2]
            mscale_pair = mscales[colv_pair-1]

            charge0 = charges[pairs[:, 0]]
            charge1 = charges[pairs[:, 1]]
            chrgprod = charge0 * charge1
            chrgprod_scale = chrgprod * mscale_pair
            dr_vec = positions[pairs[:, 0]] - positions[pairs[:, 1]]

            E_inter = get_rf_energy(dr_vec, chrgprod_scale, box)

            return jnp.sum(E_inter * mask)
        
        if self.top_mat is None:
            def get_energy(positions, box, pairs, mscales):
                return get_energy_kernel(positions, box, pairs, self.init_charges, mscales)
        else:
            def get_energy(positions, box, pairs, bcc, mscales):
                charges = self.init_charges + jnp.dot(self.top_mat, bcc).flatten()
                return get_energy_kernel(positions, box, pairs, charges, mscales)
        return get_energy


class CoulombPMEForce:

    def __init__(
        self,
        r_cut: float,
        init_charges, 
        kappa: float,
        K: Tuple[int, int, int],
        pme_order: int = 6,
        topology_matrix: Optional[jnp.array] = None,
    ):
        self.r_cut = r_cut
        self.init_charges = init_charges
        self.lmax = 0
        self.kappa = kappa
        self.K1, self.K2, self.K3 = K[0], K[1], K[2]
        self.pme_order = pme_order
        self.top_mat = topology_matrix
        assert pme_order == 6, "PME order other than 6 is not supported"

    def generate_get_energy(self):

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
        
        def get_energy_kernel(positions, box, pairs, charges, mscales):

            atomCharges = charges
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
                self.kappa / 10,
                self.K1,
                self.K2,
                self.K3,
                self.lmax,
                False,
            )

        if self.top_mat is None:
            def get_energy(positions, box, pairs, mscales):
                return get_energy_kernel(positions, box, pairs, self.init_charges, mscales)
        else:
            def get_energy(positions, box, pairs, bcc, mscales):
                charges = self.init_charges + jnp.dot(self.top_mat, bcc).flatten()
                return get_energy_kernel(positions, box, pairs, charges, mscales)
        return get_energy


class CustomGBForce:
    def __init__(
            self,
            map_charge,
            map_radius,
            map_scale,
            epsilon_1=1.0,
            epsilon_solv=78.3,
            alpha=1,
            beta=0.8,
            gamma=4.85,
    ) -> None:
        self.map_charge = map_charge
        self.map_radius = map_radius
        self.map_scale = map_scale
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.exp_solv = epsilon_solv
        self.eps_1 = epsilon_1

    def generate_get_energy(self):
        @jax.jit
        def get_energy(positions, box, pairs, Ipairs, charges, radius, scales):
            def calI(posList, radMap, scalMap, rhoMap, pairMap):
                I = jnp.array([])

                for i in range(len(radMap)):
                    posj = posList[Ipairs[i]]
                    rhoj = rhoMap[Ipairs[i]]
                    scalj = scalMap[Ipairs[i]]
                    posi = posList[i]
                    rhoi = rhoMap[i]

                    r = jnp.sqrt(jnp.sum(jnp.power(posi-posj,2),axis=1))
                    sr2 = rhoj * scalj
                    D = jnp.abs(r - sr2)
                    L = jnp.maximum(D, rhoi)
                    C = 2 * (1 / rhoi - 1 / L) * jnp.heaviside(sr2 - r - rhoi, 1)
                    U = r + sr2
                    I = jnp.append(I, jnp.sum(0.5 * jnp.heaviside(r + sr2 - rhoi, 1) * (
                            1 / L - 1 / U + 0.25 * (1 / U ** 2 - 1 / L ** 2) * (
                            r - sr2 ** 2 / r) + 0.5 * jnp.log(L / U) / r + C)))

                return I

            chargeMap = charges[self.map_charge]
            radiusMap = radius[self.map_radius]
            scalesMap = scales[self.map_scale]
            rhoMap = radiusMap - 0.009

            # effective radius
            IList = calI(positions, radiusMap, scalesMap, rhoMap, Ipairs)
            psi = IList*rhoMap
            rEff = 1/(1/rhoMap-jnp.tanh(self.alpha*psi-self.beta*jnp.power(psi, 2)+self.gamma*jnp.power(psi, 3))/radiusMap)
            Ese = jnp.sum(28.3919551*(radiusMap+0.14)**2*jnp.power(radiusMap/rEff, 6)-0.5*138.935456*(1/self.eps_1-1/self.exp_solv)*chargeMap**2/rEff)
            dr_norm = jnp.linalg.norm(positions[pairs[:,0]] - positions[pairs[:,1]], axis=1)
            chargepro = chargeMap[pairs[:, 0]] * chargeMap[pairs[:, 1]]
            rEffpro = rEff[pairs[:, 0]] * rEff[pairs[:, 1]]
            Egb = jnp.sum(-138.935456*(1/self.eps_1-1/self.exp_solv)*chargepro/jnp.sqrt(jnp.power(dr_norm, 2)+rEffpro*jnp.exp(-jnp.power(dr_norm,2)/(4*rEffpro))))
            return Ese + Egb
        return get_energy
