#!/usr/bin/env python
import sys
from pathlib import Path
admp_path = Path(__file__).parent.parent.parent
sys.path.append(str(admp_path))
import openmm.app as app
import openmm.unit as unit
import numpy as np
import jax.numpy as jnp
from collections import defaultdict
from dmff.api import Hamiltonian
from jax_md import space, partition
from jax import grad


if __name__ == '__main__':
    
    H = Hamiltonian('forcefield.xml')
    app.Topology.loadBondDefinitions("residues.xml")
    pdb = app.PDBFile("water1024.pdb")
    rc = 4.0
    # generator stores all force field parameters
    generator = H.getGenerators()
    disp_generator = generator[0]
    pme_generator = generator[1]
    pme_generator.ref_dip = 'dipole_1024'
    potentials = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom)
    # pot_fn is the actual energy calculator
    pot_disp = potentials[0]
    pot_pme = potentials[1]

    # construct inputs
    positions = jnp.array(pdb.positions._value) * 10
    a, b, c = pdb.topology.getPeriodicBoxVectors()
    box = jnp.array([a._value, b._value, c._value]) * 10
    # neighbor list
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
    nbr = neighbor_list_fn.allocate(positions)
    pairs = nbr.idx.T

    print(pot_disp(positions, box, pairs, disp_generator.params))
    param_grad = grad(pot_disp, argnums=3)(positions, box, pairs, generator[0].params)
    print(param_grad['mScales'])
    
    print(pot_pme(positions, box, pairs, pme_generator.params))
    param_grad = grad(pot_pme, argnums=3)(positions, box, pairs, generator[1].params)
    print(param_grad['mScales'])

    
