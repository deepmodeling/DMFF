from mmap import MAP_EXECUTABLE
import sys
import jax.numpy as jnp
from dmff.admp.pme import energy_pme, setup_ewald_parameters
from dmff.admp.recip import generate_pme_recip
from dmff.admp.spatial import v_pbc_shift
import numpy as np
import jax.numpy as jnp
from jax import grad
from dmff.admp.recip import generate_pme_recip, Ck_1


class LennardJonesForce:
    def __init__(self,
                 r_switch,
                 r_cut,
                 map_prm,
                 map_nbfix,
                 map_exclusion,
                 scale_exclusion,
                 isSwitch=False,
                 ifPBC=True) -> None:

        self.isSwitch = isSwitch
        self.r_switch = r_switch
        self.r_cut = r_cut

        self.map_prm = map_prm
        self.map_nbfix = map_nbfix
        self.map_exclusion = map_exclusion
        self.scale_exclusion = scale_exclusion
        self.ifPBC = ifPBC

    def generate_get_energy(self):
        def get_LJ_energy(dr_vec, sig, eps, box):
            if self.ifPBC:
                dr_vec = v_pbc_shift(dr_vec, box, jnp.linalg.inv(box))
            dr_norm = jnp.linalg.norm(dr_vec, axis=1)
            dr_norm = dr_norm[dr_norm <= self.r_cut]

            dr_inv = 1.0 / dr_norm
            sig_dr = sig * dr_inv
            sig_dr12 = jnp.power(sig_dr, 12)
            sig_dr6 = jnp.power(sig_dr, 6)
            E = 4 * eps * (sig_dr12 - sig_dr6)

            if self.isSwitch:

                x = (dr_norm - self.r_switch) / (self.r_cut - self.r_switch)
                S = 1 - 6 * x**5 + 15 * x**4 - 10 * x**3
                jnp.where(dr_norm > self.r_switch, E, E * S)

            return E

        def get_energy(positions, box, pairs, epsilon, sigma, epsfix, sigfix):
            eps_m1 = jnp.repeat(epsilon.reshape((-1, 1)),
                                epsilon.shape[0],
                                axis=1)
            eps_m2 = eps_m1.T
            eps_mat = jnp.sqrt(eps_m1 * eps_m2)
            sig_m1 = jnp.repeat(sigma.reshape((-1, 1)), sigma.shape[0], axis=1)
            sig_m2 = sig_m1.T
            sig_mat = (sig_m1 + sig_m2) * 0.5

            eps_mat = eps_mat.at[self.map_nbfix[:, 0],
                                 self.map_nbfix[:, 1]].set(epsfix)
            eps_mat = eps_mat.at[self.map_nbfix[:, 1],
                                 self.map_nbfix[:, 0]].set(epsfix)
            sig_mat = sig_mat.at[self.map_nbfix[:, 0],
                                 self.map_nbfix[:, 1]].set(sigfix)
            sig_mat = sig_mat.at[self.map_nbfix[:, 1],
                                 self.map_nbfix[:, 0]].set(sigfix)

            dr_vec = positions[pairs[:, 0]] - positions[pairs[:, 1]]
            prm_pair0 = self.map_prm[pairs[:, 0]]
            prm_pair1 = self.map_prm[pairs[:, 1]]
            eps = eps_mat[prm_pair0, prm_pair1]
            sig = sig_mat[prm_pair0, prm_pair1]

            E_inter = get_LJ_energy(dr_vec, sig, eps, box)

            # exclusion
            dr_excl_vec = positions[self.map_exclusion[:, 0]] - positions[
                self.map_exclusion[:, 1]]
            excl_map0 = self.map_prm[self.map_exclusion[:, 0]]
            excl_map1 = self.map_prm[self.map_exclusion[:, 1]]
            eps_excl = eps_mat[excl_map0, excl_map1]
            sig_excl = sig_mat[excl_map0, excl_map1]

            E_excl = get_LJ_energy(dr_excl_vec, sig_excl, eps_excl, box)
            E_excl = self.scale_exclusion * E_excl

            return jnp.sum(E_inter) - jnp.sum(E_excl)

        return get_energy


class CoulombForce:
    def __init__(self, box, rc, ethresh):

        self.kappa, self.K1, self.K2, self.K3 = setup_ewald_parameters(
            rc, ethresh, box)

    def generate_get_energy(self):
        def get_energy(positions, box, pairs, Q, mScales):
            return energy_pme(positions, box, pairs, Q, None, None, None,
                              mScales, None, None, self.covalent_map, None,
                              self.pme_recip, self.kappa, self.K1, self.K2,
                              self.K3, self.lmax, False)

        return get_energy

    def refresh_calculator(self):

        self.construct_local_frames = None
        lmax = 0
        self.pme_recip = generate_pme_recip(Ck_1, self.kappa, False,
                                            self.pme_order, self.K1, self.K2,
                                            self.K3, lmax)

        self.get_energy = self.genreate_get_energy()

        return


if __name__ == '__main__':

    # atoms: 0, 1, 2, 3
    # exclusion: 0 - 1, 2 - 3
    # nbfix: 0 - 3 (3., 0.3)
    positions = jnp.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 1]],
                          dtype=float)
    p_ref = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 1]], dtype=float)

    box = jnp.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])

    pairs = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])
    pairs_ref = np.array([[0, 2], [0, 3], [1, 2], [1, 3]])

    epsilon = jnp.array([1., 2.])
    sigma = jnp.array([0.5, 0.6])

    map_prm = np.array([0, 0, 1, 1])
    map_nbfix = np.array([[0, 3]])
    epsfix = jnp.array([3.])
    sigfix = jnp.array([0.8])
    map_exclusion = np.array([[0, 1], [2, 3]])
    scale_exclusion = jnp.array([1.0, 1.0])
    map_14 = np.array([[]])

    lj = LennardJonesForce(0, 3, map_prm, map_nbfix, map_exclusion, scale_exclusion)
    get_energy = lj.generate_get_energy()

    E = get_energy(positions, box, pairs, epsilon, sigma, epsfix, sigfix)

    # Eref
    eps0 = epsilon[map_prm[pairs_ref[:, 0]]]
    eps1 = epsilon[map_prm[pairs_ref[:, 1]]]
    sig0 = sigma[map_prm[pairs_ref[:, 0]]]
    sig1 = sigma[map_prm[pairs_ref[:, 1]]]

    eps = np.sqrt(eps0 * eps1)
    sig = (sig0 + sig1) / 2.
    p0 = p_ref[pairs_ref[:, 0]]
    p1 = p_ref[pairs_ref[:, 1]]
    dr_vec = p1 - p0
    dr = np.sqrt(np.power(dr_vec, 2).sum(axis=1))
    sig_dr = sig / dr
    Eref = 4. * eps * (np.power(sig_dr, 12) - np.power(sig_dr, 6))
    Eref = Eref.sum()
    dr_fix = p_ref[0] - p_ref[3]
    dr_fix = np.sqrt((dr_fix * dr_fix).sum())
    Eref += 4. * 3. * ((0.3 / dr_fix)**12 - (0.3 / dr_fix)**6)

    print(E, "vs", Eref)
    F = grad(get_energy)(positions, box, pairs, epsilon, sigma, epsfix, sigfix)
    print(F)

    # test PME
    rc = 4
    ethresh = 1e-4
    mScales = np.array([1., 1., 1.])
    Q = np.array([0.1, 0.1])
    pme = CoulombForce(box, rc, ethresh)
    get_energy = pme.generate_get_energy()
    E = get_energy(positions, box, pairs, Q, mScales)
