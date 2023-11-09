#!/usr/bin/env python
import sys
import numpy as np
import openmm
from openmm import *
from openmm.app import *
from openmm.unit import *
import jax
import jax_md
import jax.numpy as jnp
import dmff
from dmff.api import Hamiltonian
from dmff.common import nblist
import pickle
import time
from jax import value_and_grad, jit

if __name__ == '__main__':
    ff = 'peg.xml'
    pdb_AB = PDBFile('peg2.pdb')
    H_AB = Hamiltonian(ff)
    rc = 1.45
    # get potential functions
    pots_AB = H_AB.createPotential(pdb_AB.topology, nonbondedCutoff=rc*nanometer, nonbondedMethod=CutoffPeriodic, ethresh=1e-4)
    pot_pme_AB = pots_AB.dmff_potentials['ADMPPmeForce']
    pot_disp_AB = pots_AB.dmff_potentials['ADMPDispPmeForce']
    pot_ex_AB = pots_AB.dmff_potentials['SlaterExForce']
    pot_sr_es_AB = pots_AB.dmff_potentials['SlaterSrEsForce']
    pot_sr_pol_AB = pots_AB.dmff_potentials['SlaterSrPolForce']
    pot_sr_disp_AB = pots_AB.dmff_potentials['SlaterSrDispForce']
    pot_dhf_AB = pots_AB.dmff_potentials['SlaterDhfForce']
    pot_dmp_es_AB = pots_AB.dmff_potentials['QqTtDampingForce']
    pot_dmp_disp_AB = pots_AB.dmff_potentials['SlaterDampingForce']

    paramtree = H_AB.getParameters()

    # init positions used to set up neighbor list
    pos_AB0 = jnp.array(pdb_AB.positions._value)
    n_atoms = len(pos_AB0)
    box = jnp.array(pdb_AB.topology.getPeriodicBoxVectors()._value)

    # nn list initial allocation
    nbl_AB = nblist.NeighborList(box, rc, pots_AB.meta['cov_map'])
    nbl_AB.allocate(pos_AB0)
    pairs_AB = nbl_AB.pairs
    pairs_AB =  pairs_AB[pairs_AB[:, 0] < pairs_AB[:, 1]]

    pos_AB = jnp.array(pos_AB0)
    E_es = pot_pme_AB(pos_AB, box, pairs_AB, paramtree)
    E_disp = pot_disp_AB(pos_AB, box, pairs_AB, paramtree)
    E_ex_AB = pot_ex_AB(pos_AB, box, pairs_AB, paramtree)
    E_sr_es = pot_sr_es_AB(pos_AB, box, pairs_AB, paramtree) 
    E_sr_pol = pot_sr_pol_AB(pos_AB, box, pairs_AB, paramtree)
    E_sr_disp = pot_sr_disp_AB(pos_AB, box, pairs_AB, paramtree) 
    E_dhf = pot_dhf_AB(pos_AB, box, pairs_AB, paramtree)
    E_dmp_es = pot_dmp_es_AB(pos_AB, box, pairs_AB, paramtree) 
    E_dmp_disp = pot_dmp_disp_AB(pos_AB, box, pairs_AB, paramtree) 
    print(E_es, E_disp, E_ex_AB, E_sr_es, E_sr_pol, E_sr_disp, E_dhf, E_dmp_es, E_dmp_disp)
