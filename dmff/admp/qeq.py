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

from jax.scipy.special import erf, erfc

from dmff.utils import jit_condition, regularize_pairs, pair_buffer_scales


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
    e_sr_self = pre_self * q * q
    e_sr = jnp.sum(e_sr_pair) + jnp.sum(e_sr_self)
    return e_sr

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
    if pbc_flag is False:
        dr = pos1 - pos2
    else:
        box_inv = jnp.linalg.inv(box)
        dpos = pos1 - pos2
        dpos = dpos.dot(box_inv)
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
                         

