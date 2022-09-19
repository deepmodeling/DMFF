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
# import optax


if __name__ == '__main__':
    ff = 'forcefield.xml'
    pdb_AB = PDBFile('peg2_dimer.pdb')
    pdb_A = PDBFile('peg2.pdb')
    pdb_B = PDBFile('peg2.pdb')
    param_file = 'params.1.pickle'
    H_AB = Hamiltonian(ff)
    H_A = Hamiltonian(ff)
    H_B = Hamiltonian(ff)

    rc = 15

    # get potential functions
    pots_AB = H_AB.createPotential(pdb_AB.topology, nonbondedCutoff=rc*angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4)
    pot_pme_AB = pots_AB.dmff_potentials['ADMPPmeForce']
    pot_disp_AB = pots_AB.dmff_potentials['ADMPDispPmeForce']
    pot_ex_AB = pots_AB.dmff_potentials['SlaterExForce']
    pot_sr_es_AB = pots_AB.dmff_potentials['SlaterSrEsForce']
    pot_sr_pol_AB = pots_AB.dmff_potentials['SlaterSrPolForce']
    pot_sr_disp_AB = pots_AB.dmff_potentials['SlaterSrDispForce']
    pot_dhf_AB = pots_AB.dmff_potentials['SlaterDhfForce']
    pot_dmp_es_AB = pots_AB.dmff_potentials['QqTtDampingForce']
    pot_dmp_disp_AB = pots_AB.dmff_potentials['SlaterDampingForce']
    pots_A = H_A.createPotential(pdb_A.topology, nonbondedCutoff=rc*angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4)
    pot_pme_A = pots_A.dmff_potentials['ADMPPmeForce']
    pot_disp_A = pots_A.dmff_potentials['ADMPDispPmeForce']
    pot_ex_A = pots_A.dmff_potentials['SlaterExForce']
    pot_sr_es_A = pots_A.dmff_potentials['SlaterSrEsForce']
    pot_sr_pol_A = pots_A.dmff_potentials['SlaterSrPolForce']
    pot_sr_disp_A = pots_A.dmff_potentials['SlaterSrDispForce']
    pot_dhf_A = pots_A.dmff_potentials['SlaterDhfForce']
    pot_dmp_es_A = pots_A.dmff_potentials['QqTtDampingForce']
    pot_dmp_disp_A = pots_A.dmff_potentials['SlaterDampingForce']
    pots_B = H_B.createPotential(pdb_B.topology, nonbondedCutoff=rc*angstrom, nonbondedMethod=CutoffPeriodic, ethresh=1e-4)
    pot_pme_B = pots_B.dmff_potentials['ADMPPmeForce']
    pot_disp_B = pots_B.dmff_potentials['ADMPDispPmeForce']
    pot_ex_B = pots_B.dmff_potentials['SlaterExForce']
    pot_sr_es_B = pots_B.dmff_potentials['SlaterSrEsForce']
    pot_sr_pol_B = pots_B.dmff_potentials['SlaterSrPolForce']
    pot_sr_disp_B = pots_B.dmff_potentials['SlaterSrDispForce']
    pot_dhf_B = pots_B.dmff_potentials['SlaterDhfForce']
    pot_dmp_es_B = pots_B.dmff_potentials['QqTtDampingForce']
    pot_dmp_disp_B = pots_B.dmff_potentials['SlaterDampingForce']

    pme_generator_AB = H_AB.getGenerators()[0]
    pme_generator_A = H_A.getGenerators()[0]
    pme_generator_B = H_B.getGenerators()[0]

    # init positions used to set up neighbor list
    pos_AB0 = jnp.array(pdb_AB.positions._value) * 10
    n_atoms = len(pos_AB0)
    n_atoms_A = n_atoms // 2
    n_atoms_B = n_atoms // 2
    pos_A0 = jnp.array(pdb_AB.positions._value[:n_atoms_A]) * 10
    pos_B0 = jnp.array(pdb_AB.positions._value[n_atoms_A:n_atoms]) * 10
    box = jnp.array(pdb_AB.topology.getPeriodicBoxVectors()._value) * 10

    # nn list initial allocation
    nbl_AB = nblist.NeighborList(box, rc)
    nbl_AB.allocate(pos_AB0)
    pairs_AB = nbl_AB.pairs
    nbl_A = nblist.NeighborList(box, rc)
    nbl_A.allocate(pos_A0)
    pairs_A = nbl_A.pairs
    nbl_B = nblist.NeighborList(box, rc)
    nbl_B.allocate(pos_B0)
    pairs_B = nbl_B.pairs

    pairs_AB =  pairs_AB[pairs_AB[:, 0] < pairs_AB[:, 1]]
    pairs_A =  pairs_A[pairs_A[:, 0] < pairs_A[:, 1]]
    pairs_B =  pairs_B[pairs_B[:, 0] < pairs_B[:, 1]]

    # construct total force field params
    comps = ['ex', 'es', 'pol', 'disp', 'dhf', 'tot']
    # load parameters
    with open(param_file, 'rb') as ifile:
        params = pickle.load(ifile)

    # setting up params for all calculators
    params_ex = {}
    params_sr_es = {}
    params_sr_pol = {}
    params_sr_disp = {}
    params_dhf = {}
    params_dmp_es = {}
    params_dmp_disp = {}
    for k in ['B', 'mScales']:
        params_ex[k] = params[k]
        params_sr_es[k] = params[k]
        params_sr_pol[k] = params[k]
        params_sr_disp[k] = params[k]
        params_dhf[k] = params[k]
        params_dmp_es[k] = params[k]
        params_dmp_disp[k] = params[k]
    params_ex['A'] = params['A_ex']
    params_sr_es['A'] = params['A_es']
    params_sr_pol['A'] = params['A_pol']
    params_sr_disp['A'] = params['A_disp']
    params_dhf['A'] = params['A_dhf']
    # damping parameters
    params_dmp_es['Q'] = params['Q']
    params_dmp_disp['C6'] = params['C6']
    params_dmp_disp['C8'] = params['C8']
    params_dmp_disp['C10'] = params['C10']
    # long range parameters
    params_espol = {}
    for k in ['mScales', 'pScales', 'dScales', 'Q_local', 'pol', 'tholes']:
        params_espol[k] = params[k]
    params_disp = {}
    for k in ['B', 'C6', 'C8', 'C10', 'mScales']:
        params_disp[k] = params[k]

    paramtree = {}
    paramtree['ADMPPmeForce'] = params_espol
    paramtree['ADMPDispPmeForce'] = params_disp
    paramtree['SlaterExForce'] = params_ex
    paramtree['SlaterSrEsForce'] = params_sr_es
    paramtree['SlaterSrPolForce'] = params_sr_pol
    paramtree['SlaterSrDispForce'] = params_sr_disp
    paramtree['SlaterDhfForce'] = params_dhf
    paramtree['QqTtDampingForce'] = params_dmp_es
    paramtree['SlaterDampingForce'] = params_dmp_disp

    # load data
    with open('data.pickle', 'rb') as ifile:
        data = pickle.load(ifile)
    with open('data_sr.pickle', 'rb') as ifile:
        data_sr = pickle.load(ifile)
    with open('data_lr.pickle', 'rb') as ifile:
        data_lr = pickle.load(ifile)
    sids = list(data.keys())
    sids.sort()

    ofile_tot = open('res_tot.xvg', 'w')
    ofile_es = open('res_es.xvg', 'w')
    ofile_ex = open('res_ex.xvg', 'w')
    ofile_pol = open('res_pol.xvg', 'w')
    ofile_disp = open('res_disp.xvg', 'w')
    ofile_dhf = open('res_dhf.xvg', 'w')
    # run test
    for sid in sids:
        print(sid)
    # for sid in ['000']:
        scan_res = data[sid]
        scan_res_sr = data_sr[sid]
        scan_res_lr = data_lr[sid]
        npts = len(scan_res['tot'])

        for ipt in range(npts):
            E_es_ref = scan_res['es'][ipt]
            E_pol_ref = scan_res['pol'][ipt]
            E_disp_ref = scan_res['disp'][ipt]
            E_ex_ref = scan_res['ex'][ipt]
            E_dhf_ref = scan_res['dhf'][ipt]
            E_tot_ref = scan_res['tot'][ipt]

            pos_A = jnp.array(scan_res['posA'][ipt])
            pos_B = jnp.array(scan_res['posB'][ipt])
            pos_AB = jnp.concatenate([pos_A, pos_B], axis=0)

            #####################
            # exchange repulsion
            #####################
            E_ex_AB = pot_ex_AB(pos_AB, box, pairs_AB, paramtree)
            E_ex_A = pot_ex_A(pos_A, box, pairs_A, paramtree)
            E_ex_B = pot_ex_B(pos_B, box, pairs_B, paramtree)
            E_ex = E_ex_AB - E_ex_A - E_ex_B

            # #######################
            # # electrostatic + pol
            # #######################
            E_AB = pot_pme_AB(pos_AB, box, pairs_AB, paramtree)
            E_A = pot_pme_A(pos_A, box, pairs_A, paramtree)
            E_B = pot_pme_B(pos_B, box, pairs_A, paramtree)
            E_espol = E_AB - E_A - E_B

            # use induced dipole of monomers to compute electrostatic interaction
            U_ind_AB = jnp.vstack((pme_generator_A.pme_force.U_ind, pme_generator_B.pme_force.U_ind))
            params = paramtree['ADMPPmeForce']
            map_atypes = pme_generator_AB.map_atomtype
            map_poltype = pme_generator_AB.map_poltype
            Q_local = params['Q_local'][map_atypes]
            pol = params['pol'][map_poltype]
            tholes = params['tholes'][map_poltype]
            pme_force = pme_generator_AB.pme_force
            E_AB_nonpol = pme_force.energy_fn(pos_AB, box, pairs_AB, Q_local, U_ind_AB, pol, tholes, params['mScales'], params['pScales'], params['dScales'])
            E_es = E_AB_nonpol - E_A - E_B
            E_dmp_es = pot_dmp_es_AB(pos_AB, box, pairs_AB, paramtree) \
                     - pot_dmp_es_A(pos_A, box, pairs_A, paramtree) \
                     - pot_dmp_es_B(pos_B, box, pairs_B, paramtree)
            E_sr_es = pot_sr_es_AB(pos_AB, box, pairs_AB, paramtree) \
                    - pot_sr_es_A(pos_A, box, pairs_A, paramtree) \
                    - pot_sr_es_B(pos_B, box, pairs_B, paramtree)


            ###################################
            # polarization (induction) energy
            ###################################
            E_pol = E_espol - E_es
            E_sr_pol = pot_sr_pol_AB(pos_AB, box, pairs_AB, paramtree) \
                     - pot_sr_pol_A(pos_A, box, pairs_A, paramtree) \
                     - pot_sr_pol_B(pos_B, box, pairs_B, paramtree)


            #############
            # dispersion
            #############
            E_AB_disp = pot_disp_AB(pos_AB, box, pairs_AB, paramtree)
            E_A_disp = pot_disp_A(pos_A, box, pairs_A, paramtree)
            E_B_disp = pot_disp_B(pos_B, box, pairs_B, paramtree)
            E_disp = E_AB_disp - E_A_disp - E_B_disp
            E_dmp_disp = pot_dmp_disp_AB(pos_AB, box, pairs_AB, paramtree) \
                       - pot_dmp_disp_A(pos_A, box, pairs_A, paramtree) \
                       - pot_dmp_disp_B(pos_B, box, pairs_B, paramtree)
            E_sr_disp = pot_sr_disp_AB(pos_AB, box, pairs_AB, paramtree) \
                      - pot_sr_disp_A(pos_A, box, pairs_A, paramtree) \
                      - pot_sr_disp_B(pos_B, box, pairs_B, paramtree)

            ###########
            # dhf
            ###########
            E_AB_dhf = pot_dhf_AB(pos_AB, box, pairs_AB, paramtree)
            E_A_dhf = pot_dhf_A(pos_A, box, pairs_A, paramtree)
            E_B_dhf = pot_dhf_B(pos_B, box, pairs_B, paramtree)
            E_dhf = E_AB_dhf - E_A_dhf - E_B_dhf

            # total energy
            E_tot = (E_es + E_sr_es + E_dmp_es) + (E_ex) + (E_pol + E_sr_pol) + (E_disp + E_dmp_disp + E_sr_disp) + (E_dhf)
            E_tot_sr = (E_sr_es + E_dmp_es) + (E_ex) + (E_sr_pol) + (E_sr_disp + E_dmp_disp) + (E_dhf)
            E_tot_lr = E_es + E_pol + E_disp

            # if E_tot < 30:
            #     e_es = (E_es + E_sr_es + E_dmp_es)
            #     e_ex = (E_ex)
            #     e_pol = (E_pol + E_sr_pol)
            #     e_disp = (E_disp + E_dmp_disp + E_sr_disp)
            #     e_dhf = (E_dhf)
            #     print(E_tot_ref, E_tot_ref, E_tot, file=ofile_tot)
            #     print(E_es_ref, E_es_ref, e_es, file=ofile_es)
            #     print(E_ex_ref, E_ex_ref, e_ex, file=ofile_ex)
            #     print(E_pol_ref, E_pol_ref, e_pol, file=ofile_pol)
            #     print(E_disp_ref, E_disp_ref, e_disp, file=ofile_disp)
            #     print(E_dhf_ref, E_dhf_ref, e_dhf, file=ofile_dhf)

            if E_tot < 30:
                print(E_tot_ref, E_tot_ref, E_tot)
