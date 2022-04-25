import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np
from jax_md import space, partition
import sys
sys.path.append('/home/lijichen/work/DMFF/')
from dmff.api import Hamiltonian
import jax.numpy as jnp
from jax import value_and_grad

if __name__ == '__main__':
    
    print('load forcefield and topo file')
    pdb = app.PDBFile('water_dimer.pdb')
    rc = 4.0
    H = Hamiltonian("forcefield.xml")
    system = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.angstrom)
    
    positions = jnp.array(pdb.positions._value) * 10
    a, b, c = pdb.topology.getPeriodicBoxVectors()
    box = jnp.array([a._value, b._value, c._value]) * 10
    
    # neighbor list
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
    nbr = neighbor_list_fn.allocate(positions)
    pairs = nbr.idx.T        


    bondE = H._potentials[0]
    print("Bond:", bondE(positions, box, pairs, H.getGenerators()[0].params))

    angleE = H._potentials[1]
    print("Angle:", angleE(positions, box, pairs, H.getGenerators()[1].params))

    nonBondE = H._potentials[2]
    print('NonBonded:', nonBondE(positions, box, pairs, H.getGenerators()[2].params))
    
    import optax
    # start to do optmization
    params = H.getGenerators()[0].params 
    lr = 0.01
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    n_epochs = 1000
    for i_epoch in range(n_epochs):
            loss, grads = value_and_grad(bondE, argnums=(3))(positions, box, pairs, params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            print(loss)