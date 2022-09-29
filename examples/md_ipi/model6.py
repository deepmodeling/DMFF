#!/usr/bin/env python
import sys
import numpy as np
import jax.numpy as jnp
from dmff.utils import jit_condition
from dmff.admp.parser import *
from dmff.admp.spatial import v_pbc_shift
import linecache
from dmff.settings import DO_JIT
from jax import jit
def get_line_context(file_path, line_number):
    return linecache.getline(file_path,line_number).strip()

def gen_trim_val_0(thresh):
    '''
    Trim the value at zero point to avoid singularity
    '''
    def trim_val_0(x):
        return jnp.piecewise(x, [x<thresh, x>=thresh], [lambda x: jnp.array(thresh), lambda x: x])
    if DO_JIT:
        return jit(trim_val_0)
    else:
        return trim_val_0

trim_val_0 = gen_trim_val_0(1e-8)

@jit_condition(static_argnums=())
def compute_leading_terms(positions,box):
    n_atoms = len(positions)
    c0 = jnp.zeros(n_atoms)
    c6_list = jnp.zeros(n_atoms)
    box_inv = jnp.linalg.inv(box)
    O = positions[::3]
    H1 = positions[1::3]
    H2 = positions[2::3]
    ROH1 = H1 - O
    ROH2 = H2 - O
    ROH1 = v_pbc_shift(ROH1, box, box_inv)
    ROH2 = v_pbc_shift(ROH2, box, box_inv)
    dROH1 = jnp.linalg.norm(ROH1, axis=1)
    dROH2 = jnp.linalg.norm(ROH2, axis=1)
    costh = jnp.sum(ROH1 * ROH2, axis=1) / (dROH1 * dROH2)
    angle = jnp.arccos(costh)*180/jnp.pi
    dipole1 = -0.016858755+0.002287251*angle + 0.239667591*dROH1 + (-0.070483437)*dROH2
    charge_H1 = dipole1/dROH1
    dipole2 = -0.016858755+0.002287251*angle + 0.239667591*dROH2 + (-0.070483437)*dROH1
    charge_H2 = dipole2/dROH2
    charge_O = -(charge_H1 + charge_H2)
    C6_H1 = (-2.36066199 + (-0.007049238)*angle + 1.949429648*dROH1+ 2.097120784*dROH2) * 0.529**6 * 2625.5
    C6_H2 = (-2.36066199 + (-0.007049238)*angle + 1.949429648*dROH2+ 2.097120784*dROH1) * 0.529**6 * 2625.5
    C6_O = (-8.641301261 + 0.093247893*angle + 11.90395358*(dROH1+ dROH2)) * 0.529**6 * 2625.5
    C6_H1 = trim_val_0(C6_H1)
    C6_H2 = trim_val_0(C6_H2)
    c0 = c0.at[::3].set(charge_O)
    c0 = c0.at[1::3].set(charge_H1)
    c0 = c0.at[2::3].set(charge_H2)
    c6_list = c6_list.at[::3].set(jnp.sqrt(C6_O))
    c6_list = c6_list.at[1::3].set(jnp.sqrt(C6_H1))
    c6_list = c6_list.at[2::3].set(jnp.sqrt(C6_H2))
    return c0, c6_list

