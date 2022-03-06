import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, vmap, jit
from jax.scipy.special import erf

def distance(p1v, p2v):
    return jnp.sqrt(jnp.sum(jnp.power(p1v - p2v, 2), axis=1))
    
def angle(p1v, p2v, p3v):
    v1 = (p2v - p1v) / jnp.reshape(distance(p1v, p2v), (-1,1))
    v2 = (p2v - p3v) / jnp.reshape(distance(p2v, p3v), (-1,1))
    vxx = v1[:,0] * v2[:,0]
    vyy = v1[:,1] * v2[:,1]
    vzz = v1[:,2] * v2[:,2]
    return jnp.arccos(vxx + vyy + vzz)
    
def dihedral(i, j, k, l):
    f, g, h = i - j, j - k, l - k
    a = vmap(jnp.cross, (0, 0))(f, g)
    b = vmap(jnp.cross, (0, 0))(h, g)
    axb = vmap(jnp.cross, (0, 0))(a, b)
    cos = vmap(jnp.dot, (0, 0))(a, b)
    sin = vmap(jnp.dot, (0, 0))(axb, g) / jnp.linalg.norm(g)
    r = - vmap(jnp.arctan2, (0, 0))(sin, cos)
    return r

class HarmonicBondJaxForce:
    def __init__(self, p1idx, p2idx, prmidx):
        self.p1idx = p1idx
        self.p2idx = p2idx
        self.prmidx = prmidx
        self.refresh_calculators()

    def generate_get_energy(self):
        def get_energy(positions, box, pairs, k, length):
            p1 = positions[self.p1idx]
            p2 = positions[self.p2idx]
            kprm = k[self.prmidx]
            b0prm = length[self.prmidx]
            dist = distance(p1, p2)
            return jnp.sum(0.5 * kprm * jnp.power(dist - b0prm, 2))

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
            p1 = positions[self.p1idx]
            p2 = positions[self.p2idx]
            p3 = positions[self.p3idx]
            kprm = k[self.prmidx]
            theta0prm = theta0[self.prmidx]
            ang = angle(p1, p2, p3)
            return jnp.sum(0.5 * kprm * jnp.power(ang - theta0prm, 2))

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
        self.get_energy = self.generate_get_energy()
        self.get_forces = value_and_grad(self.get_energy)


class PeriodicTorsionJaxForce:
    def __init__(self, p1idx, p2idx, p3idx, p4idx, prmidx):
        self.p1idx = p1idx
        self.p2idx = p2idx
        self.p3idx = p3idx
        self.p4idx = p4idx
        self.prmidx = prmidx
        self.refresh_calculators()

    def generate_get_energy(self):
        def get_energy(positions, box, pairs, k1, psi1, k2, psi2, k3, psi3):
            p1 = positions[self.p1idx]
            p2 = positions[self.p2idx]
            p3 = positions[self.p3idx]
            p4 = positions[self.p4idx]
            k1p = k1[self.prmidx]
            psi1p = psi1[self.prmidx]
            k2p = k2[self.prmidx]
            psi2p = psi2[self.prmidx]
            k3p = k3[self.prmidx]
            psi3p = psi3[self.prmidx]
            dih = dihedral(p1, p2, p3, p4)
            return k1p * (1 + jnp.cos(dih - psi1p)) + k2p * (1 + jnp.cos(2. * dih - psi2p)) + k3p * (1 + jnp.cos(3. * dih - psi3p)) 

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
        self.get_energy = self.generate_get_energy()
        self.get_forces = value_and_grad(self.get_energy)
