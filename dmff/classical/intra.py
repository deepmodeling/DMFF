import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, vmap
from ..admp.spatial import v_pbc_shift


def distance(p1v, p2v):
    return jnp.sqrt(jnp.sum(jnp.power(p1v - p2v, 2), axis=1))


def angle(p1v, p2v, p3v):
    v1 = (p2v - p1v) / jnp.reshape(distance(p1v, p2v), (-1, 1))
    v2 = (p2v - p3v) / jnp.reshape(distance(p2v, p3v), (-1, 1))
    vxx = v1[:, 0] * v2[:, 0]
    vyy = v1[:, 1] * v2[:, 1]
    vzz = v1[:, 2] * v2[:, 2]
    return jnp.arccos(vxx + vyy + vzz)

@jax.vmap
def angle_v(v1, v2):
    # compute the angle between v1 and v2
    v1n = v1 / jnp.linalg.norm(v1)
    v2n = v2 / jnp.linalg.norm(v2)
    return jnp.arccos(jnp.dot(v1n, v2n))


def dihedral(i, j, k, l):
    b1, b2, b3 = j - i, k - j, l - k

    c1 = vmap(jnp.cross, (0, 0))(b2, b3)
    c2 = vmap(jnp.cross, (0, 0))(b1, b2)

    p1 = (b1 * c1).sum(-1)
    p1 = p1 * jnp.sqrt((b2 * b2).sum(-1))
    p2 = (c1 * c2).sum(-1)

    r = vmap(jnp.arctan2, (0, 0))(p1, p2)
    return r


class HarmonicBondJaxForce:
    def __init__(self, p1idx, p2idx, prmidx):
        self.p1idx = p1idx
        self.p2idx = p2idx
        self.prmidx = prmidx
        self.refresh_calculators()

    def generate_get_energy(self):
        def get_energy(positions, box, pairs, k, length):
            p1 = positions[self.p1idx,:]
            p2 = positions[self.p2idx,:]
            kprm = k[self.prmidx]
            b0prm = length[self.prmidx]
            dist = distance(p1, p2)
            return jnp.sum(0.5 * kprm * jnp.power(dist - b0prm, 2))

        return get_energy

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
        self.get_energy = self.generate_get_energy()
        self.get_forces = value_and_grad(self.get_energy)


class HarmonicAngleJaxForce:
    def __init__(self, p1idx, p2idx, p3idx, prmidx):
        self.p1idx = p1idx
        self.p2idx = p2idx
        self.p3idx = p3idx
        self.prmidx = prmidx
        self.refresh_calculators()

    def generate_get_energy(self):
        def get_energy(positions, box, pairs, k, theta0):
            box_inv = jnp.linalg.inv(box + jnp.eye(3) * 1e-36)
            p1 = positions[self.p1idx,:]
            p2 = positions[self.p2idx,:]
            p3 = positions[self.p3idx,:]
            v1 = v_pbc_shift(p1 - p2, box, box_inv)
            v2 = v_pbc_shift(p3 - p2, box, box_inv)
            kprm = k[self.prmidx]
            theta0prm = theta0[self.prmidx]
            ang = angle_v(v1, v2)
            return jnp.sum(0.5 * kprm * jnp.power(ang - theta0prm, 2))

        return get_energy

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
        self.get_energy = self.generate_get_energy()
        self.get_forces = value_and_grad(self.get_energy)


class PeriodicTorsionJaxForce:
    def __init__(self, p1idx, p2idx, p3idx, p4idx, prmidx, order):
        self.p1idx = p1idx
        self.p2idx = p2idx
        self.p3idx = p3idx
        self.p4idx = p4idx
        self.prmidx = prmidx
        self.order = order
        self.refresh_calculators()

    def generate_get_energy(self):
        if len(self.p1idx) == 0:
            return lambda positions, box, pairs, k, psi: 0.0
        def get_energy(positions, box, pairs, k, psi):
            p1 = positions[self.p1idx,:]
            p2 = positions[self.p2idx,:]
            p3 = positions[self.p3idx,:]
            p4 = positions[self.p4idx,:]
            kp = k[self.prmidx]
            psip = psi[self.prmidx]
            dih = dihedral(p1, p2, p3, p4)
            ener = kp * (1 + jnp.cos(self.order * dih - psip))
            return jnp.sum(ener)

        return get_energy

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
        self.get_energy = self.generate_get_energy()
        self.get_forces = value_and_grad(self.get_energy)


class Custom1_5BondJaxForce:
    def __init__(self, p1idx, p2idx, prmidx):
        self.p1idx = p1idx
        self.p2idx = p2idx
        self.prmidx = prmidx
        self.refresh_calculators()

    def generate_get_energy(self):
        def get_energy(positions, box, pairs, k, length):
            p1 = positions[self.p1idx,:]
            p2 = positions[self.p2idx,:]
            kprm = k[self.prmidx]
            b0prm = length[self.prmidx]
            dist = distance(p1, p2)
            return jnp.sum(0.5 * kprm * jnp.power(dist - b0prm, 2))

        return get_energy

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
        self.get_energy = self.generate_get_energy()
        self.get_forces = value_and_grad(self.get_energy)


class CustomTorsionJaxForce:
    def __init__(self, p1idx, p2idx, p3idx, p4idx, prmidx, order):
        self.p1idx = p1idx
        self.p2idx = p2idx
        self.p3idx = p3idx
        self.p4idx = p4idx
        self.prmidx = prmidx
        self.order = order
        self.refresh_calculators()

    def generate_get_energy(self):
        if len(self.p1idx) == 0:
            return lambda positions, box, pairs, k, psi, shift: 0.0
        def get_energy(positions, box, pairs, k, psi, shift):
            p1 = positions[self.p1idx, :]
            p2 = positions[self.p2idx, :]
            p3 = positions[self.p3idx, :]
            p4 = positions[self.p4idx, :]
            kp = k[self.prmidx]
            psip = psi[self.prmidx]
            shiftp = shift[self.prmidx]
            dih = dihedral(p1, p2, p3, p4)
            ener = kp * (jnp.cos(self.order * dih - psip)) + shiftp
            return jnp.sum(ener)

        return get_energy

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
        self.get_energy = self.generate_get_energy()
        self.get_forces = value_and_grad(self.get_energy)