import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, vmap, jit
from jax.scipy.special import erf

def distance(p1v, p2v):
    pass

def angle(p1v, p2v, p3v):
    pass

def dihedral(p1v, p2v, p3v, p4v):
    pass

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
            kprm = k[self.prmidx][0]
            b0prm = length[self.prmidx][1]
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
            kprm = k[self.prmidx][0]
            theta0prm = theta0[self.prmidx][1]
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
        def get_energy(positions, box, pairs, k, psi0):
            p1 = positions[self.p1idx]
            p2 = positions[self.p2idx]
            p3 = positions[self.p3idx]
            p4 = positions[self.p4idx]
            kprm = k[self.prmidx][0]
            psi0prm = psi0[self.prmidx][1]
            dih = dihedral(p1, p2, p3, p4)
            return jnp.sum(0.5 * k * jnp.power(dih - psi0, 2))

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
