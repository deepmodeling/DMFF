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
import optax


if __name__ == '__main__':
    restart = 'params.0.pickle' # None
    # restart = None
    ff = 'forcefield.xml'
    pdb_AB = PDBFile('peg2_dimer.pdb')
    pdb_A = PDBFile('peg2.pdb')
    pdb_B = PDBFile('peg2.pdb')
    H_AB = Hamiltonian(ff)
    H_A = Hamiltonian(ff)
    H_B = Hamiltonian(ff)
    pme_generator_AB, \
            disp_generator_AB, \
            ex_generator_AB, \
            sr_es_generator_AB, \
            sr_pol_generator_AB, \
            sr_disp_generator_AB, \
            dhf_generator_AB, \
            dmp_es_generator_AB, \
            dmp_disp_generator_AB = H_AB.getGenerators()
    pme_generator_A, \
            disp_generator_A, \
            ex_generator_A, \
            sr_es_generator_A, \
            sr_pol_generator_A, \
            sr_disp_generator_A, \
            dhf_generator_A, \
            dmp_es_generator_A, \
            dmp_disp_generator_A = H_A.getGenerators()
    pme_generator_B, \
            disp_generator_B, \
            ex_generator_B, \
            sr_es_generator_B, \
            sr_pol_generator_B, \
            sr_disp_generator_B, \
            dhf_generator_B, \
            dmp_es_generator_B, \
            dmp_disp_generator_B = H_B.getGenerators()

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
    pots_A = H_A.createPotential(pdb_A.topology, nonbondedCutoff=rc*nanometer, nonbondedMethod=CutoffPeriodic, ethresh=1e-4)
    pot_pme_A = pots_A.dmff_potentials['ADMPPmeForce']
    pot_disp_A = pots_A.dmff_potentials['ADMPDispPmeForce']
    pot_ex_A = pots_A.dmff_potentials['SlaterExForce']
    pot_sr_es_A = pots_A.dmff_potentials['SlaterSrEsForce']
    pot_sr_pol_A = pots_A.dmff_potentials['SlaterSrPolForce']
    pot_sr_disp_A = pots_A.dmff_potentials['SlaterSrDispForce']
    pot_dhf_A = pots_A.dmff_potentials['SlaterDhfForce']
    pot_dmp_es_A = pots_A.dmff_potentials['QqTtDampingForce']
    pot_dmp_disp_A = pots_A.dmff_potentials['SlaterDampingForce']
    pots_B = H_B.createPotential(pdb_B.topology, nonbondedCutoff=rc*nanometer, nonbondedMethod=CutoffPeriodic, ethresh=1e-4)
    pot_pme_B = pots_B.dmff_potentials['ADMPPmeForce']
    pot_disp_B = pots_B.dmff_potentials['ADMPDispPmeForce']
    pot_ex_B = pots_B.dmff_potentials['SlaterExForce']
    pot_sr_es_B = pots_B.dmff_potentials['SlaterSrEsForce']
    pot_sr_pol_B = pots_B.dmff_potentials['SlaterSrPolForce']
    pot_sr_disp_B = pots_B.dmff_potentials['SlaterSrDispForce']
    pot_dhf_B = pots_B.dmff_potentials['SlaterDhfForce']
    pot_dmp_es_B = pots_B.dmff_potentials['QqTtDampingForce']
    pot_dmp_disp_B = pots_B.dmff_potentials['SlaterDampingForce']

    pos_AB0 = jnp.array(pdb_AB.positions._value)
    n_atoms = len(pos_AB0)
    n_atoms_A = n_atoms // 2
    n_atoms_B = n_atoms // 2
    pos_A0 = jnp.array(pdb_AB.positions._value[:n_atoms_A])
    pos_B0 = jnp.array(pdb_AB.positions._value[n_atoms_A:n_atoms])
    box = jnp.array(pdb_AB.topology.getPeriodicBoxVectors()._value)
    # nn list initial allocation
    nbl_AB = nblist.NeighborList(box, rc, pots_AB.meta["cov_map"])
    nbl_AB.allocate(pos_AB0)
    pairs_AB = nbl_AB.pairs
    nbl_A = nblist.NeighborList(box, rc, pots_A.meta["cov_map"])
    nbl_A.allocate(pos_A0)
    pairs_A = nbl_A.pairs
    nbl_B = nblist.NeighborList(box, rc, pots_B.meta["cov_map"])
    nbl_B.allocate(pos_B0)
    pairs_B = nbl_B.pairs

    pairs_AB =  pairs_AB[pairs_AB[:, 0] < pairs_AB[:, 1]]
    pairs_A =  pairs_A[pairs_A[:, 0] < pairs_A[:, 1]]
    pairs_B =  pairs_B[pairs_B[:, 0] < pairs_B[:, 1]]

    params0 = H_AB.getParameters()
    # construct total force field params
    comps = ['ex', 'es', 'pol', 'disp', 'dhf', 'tot']
    weights_comps = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 1.0])
    if restart is None:
        params = {}
        sr_forces = {
                'ex': 'SlaterExForce',
                'es': 'SlaterSrEsForce',
                'pol': 'SlaterSrPolForce',
                'disp': 'SlaterSrDispForce',
                'dhf': 'SlaterDhfForce',
                }
        for k in params0['ADMPPmeForce']:
            params[k] = params0['ADMPPmeForce'][k]
        for k in params0['ADMPDispPmeForce']:
            params[k] = params0['ADMPDispPmeForce'][k]
        for c in comps:
            if c == 'tot':
                continue
            force = sr_forces[c]
            for k in params0[sr_forces[c]]:
                if k == 'A':
                    params['A_'+c] = params0[sr_forces[c]][k]
                else:
                    params[k] = params0[sr_forces[c]][k]
        # a random initialization of A
        for c in comps:
            if c == 'tot':
                continue
            params['A_'+c] = jnp.array(np.random.random(params['A_'+c].shape))
        # specify charges for es damping
        params['Q'] = params0['QqTtDampingForce']['Q']
    else:
        with open(restart, 'rb') as ifile:
            params = pickle.load(ifile)


    @jit
    def MSELoss(params, scan_res):
        '''
        The weighted mean squared error loss function
        Conducted for each scan
        '''
        E_tot_full = scan_res['tot_full']
        kT = 2.494 # 300 K = 2.494 kJ/mol
        weights_pts = jnp.piecewise(E_tot_full, [E_tot_full<25, E_tot_full>=25], [lambda x: jnp.array(1.0), lambda x: jnp.exp(-(x-25)/kT)])
        npts = len(weights_pts)
        
        energies = {
                'ex': jnp.zeros(npts), 
                'es': jnp.zeros(npts), 
                'pol': jnp.zeros(npts),
                'disp': jnp.zeros(npts),
                'dhf': jnp.zeros(npts),
                'tot': jnp.zeros(npts)
                }

        # setting up params for all calculators
        params_ex = {}
        params_sr_es = {}
        params_sr_pol = {}
        params_sr_disp = {}
        params_dhf = {}
        params_dmp_es = {}  # electrostatic damping
        params_dmp_disp = {} # dispersion damping
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
        p = {}
        p['SlaterExForce'] = params_ex
        p['SlaterSrEsForce'] = params_sr_es
        p['SlaterSrPolForce'] = params_sr_pol
        p['SlaterSrDispForce'] = params_sr_disp
        p['SlaterDhfForce'] = params_dhf
        p['QqTtDampingForce'] = params_dmp_es
        p['SlaterDampingForce'] = params_dmp_disp

        # calculate each points, only the short range and damping components
        for ipt in range(npts):
            # get position array
            pos_A = jnp.array(scan_res['posA'][ipt]) / 10
            pos_B = jnp.array(scan_res['posB'][ipt]) / 10
            pos_AB = jnp.concatenate([pos_A, pos_B], axis=0)
     
            #####################
            # exchange repulsion
            #####################
            E_ex_AB = pot_ex_AB(pos_AB, box, pairs_AB, p)
            E_ex_A = pot_ex_A(pos_A, box, pairs_A, p)
            E_ex_B = pot_ex_B(pos_B, box, pairs_B, p)
            E_ex = E_ex_AB - E_ex_A - E_ex_B

            #######################
            # electrostatic + pol
            #######################
            E_dmp_es = pot_dmp_es_AB(pos_AB, box, pairs_AB, p) \
                     - pot_dmp_es_A(pos_A, box, pairs_A, p) \
                     - pot_dmp_es_B(pos_B, box, pairs_B, p)
            E_sr_es = pot_sr_es_AB(pos_AB, box, pairs_AB, p) \
                    - pot_sr_es_A(pos_A, box, pairs_A, p) \
                    - pot_sr_es_B(pos_B, box, pairs_B, p)

            ###################################
            # polarization (induction) energy
            ###################################
            E_sr_pol = pot_sr_pol_AB(pos_AB, box, pairs_AB, p) \
                     - pot_sr_pol_A(pos_A, box, pairs_A, p) \
                     - pot_sr_pol_B(pos_B, box, pairs_B, p)

            #############
            # dispersion
            #############
            E_dmp_disp = pot_dmp_disp_AB(pos_AB, box, pairs_AB, p) \
                       - pot_dmp_disp_A(pos_A, box, pairs_A, p) \
                       - pot_dmp_disp_B(pos_B, box, pairs_B, p)
            E_sr_disp = pot_sr_disp_AB(pos_AB, box, pairs_AB, p) \
                      - pot_sr_disp_A(pos_A, box, pairs_A, p) \
                      - pot_sr_disp_B(pos_B, box, pairs_B, p)

            ###########
            # dhf
            ###########
            E_AB_dhf = pot_dhf_AB(pos_AB, box, pairs_AB, p)
            E_A_dhf = pot_dhf_A(pos_A, box, pairs_A, p)
            E_B_dhf = pot_dhf_B(pos_B, box, pairs_B, p)
            E_dhf = E_AB_dhf - E_A_dhf - E_B_dhf

            energies['ex'] = energies['ex'].at[ipt].set(E_ex)
            energies['es'] = energies['es'].at[ipt].set(E_dmp_es + E_sr_es)
            energies['pol'] = energies['pol'].at[ipt].set(E_sr_pol)
            energies['disp'] = energies['disp'].at[ipt].set(E_dmp_disp + E_sr_disp)
            energies['dhf'] = energies['dhf'].at[ipt].set(E_dhf)
            energies['tot'] = energies['tot'].at[ipt].set(E_ex 
                                                        + E_dmp_es + E_sr_es
                                                        + E_sr_pol 
                                                        + E_dmp_disp + E_sr_disp 
                                                        + E_dhf)


        errs = jnp.zeros(len(comps))
        for ic, c in enumerate(comps):
            dE = energies[c] - scan_res[c]
            mse = dE**2 * weights_pts / jnp.sum(weights_pts)
            errs = errs.at[ic].set(jnp.sum(mse))

        return jnp.sum(weights_comps * errs)


    # load data
    with open('data_sr.pickle', 'rb') as ifile:
        data = pickle.load(ifile)

    MSELoss_grad = jit(value_and_grad(MSELoss, argnums=(0)))
    err, gradients = MSELoss_grad(params, data['000'])
    sids = np.array(list(data.keys()))


    # only optimize these parameters A/B
    def mask_fn(grads):
        for k in grads:
            if k.startswith('A_') or k == 'B':
                continue
            else:
                grads[k] = 0.0
        return grads

    # start to do optmization
    lr = 0.01
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    n_epochs = 2000
    for i_epoch in range(n_epochs):
        np.random.shuffle(sids)
        for sid in sids:
            loss, grads = MSELoss_grad(params, data[sid])
            grads = mask_fn(grads)
            print(loss)
            sys.stdout.flush()
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
        with open('params.pickle', 'wb') as ofile:
            pickle.dump(params, ofile)


