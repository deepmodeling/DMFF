#!/usr/bin/env python
import sys
from collections import OrderedDict
import jax
import jax.numpy as jnp
from jax import vmap, jit, value_and_grad
import numpy as np
from dmff.utils import jit_condition, regularize_pairs, pair_buffer_scales
from dmff.admp.pairwise import distribute_scalar, distribute_v3
from dmff.admp.spatial import pbc_shift
from functools import partial
import jax.nn.initializers
import pickle
# from jax.config import config
# config.update("jax_debug_nans", True)

# Make printing parameters a little more readable
def parameter_shapes(params):
    return jax.tree_util.tree_map(lambda p: p.shape, params)

@jit_condition(static_argnums=())
@partial(vmap, in_axes=(0, 0, None, None, None, None, None), out_axes=0)
def get_gto(i_atom, r, pairs, rc, rs, inta, species):
    gto_i = jnp.exp(-inta[species[pairs[i_atom][1]]] * (r - rs[species[pairs[i_atom][1]]])**2)
    gto_j = jnp.exp(-inta[species[pairs[i_atom][0]]] * (r - rs[species[pairs[i_atom][0]]])**2)
    return gto_i, gto_j

@jit_condition(static_argnums=())
@partial(vmap, in_axes=(0, None), out_axes=0)
def cutoff_cosine(distances, cutoff):
    return jnp.square(0.5 * jnp.cos(distances * (jnp.pi / cutoff)) + 0.5)

@jit_condition(static_argnums=())
@partial(vmap, in_axes=(0, 0, None), out_axes=0)
def distribute_pair_cij(i_elem, j_elem, cij):
    return cij[j_elem]
    
@jit_condition(static_argnums=())
@partial(vmap, in_axes=(0, None, None, None), out_axes=(0)) 
def reduce_atoms(i_atom, wfs, indices, buffer_scales):
    mask = (indices == i_atom)
    res = jnp.einsum('ijk,i,i', wfs, mask, buffer_scales)
    return res

def layer_norm(x, weight, bias, axis=-1, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    std = jnp.sqrt(var + eps)
    y = (x - mean) / std * weight + bias
    return y

# calculate neural network energy through features
# Linear, LayerNorm, Relu_Like, Linear, LayerNorm, Relu_Like, Linear
@jit_condition(static_argnums=())
@partial(vmap, in_axes=(0, 0, None), out_axes=(0))
def get_atomic_energies(features, elem_index, params):
    # 0 Linear
    features1 = features.dot(params['w.0'][elem_index].T) + params['b.0'][elem_index]
    # 1 LayerNorm
    features2 = layer_norm(features1, params['w.1'][elem_index], params['b.1'][elem_index])
    # 2 Relu_Like
    features3 = params['w.2'][elem_index] * jax.nn.silu(features2 * params['b.2'][elem_index])
    # 3 Linear 
    features4 = features3.dot(params['w.3'][elem_index].T) + params['b.3'][elem_index]
    # 4 LayerNorm
    features5 = layer_norm(features4, params['w.4'][elem_index], params['b.4'][elem_index])
    # 5 Relu_Like 
    features6 = params['w.5'][elem_index] * jax.nn.silu(features5 * params['b.5'][elem_index])
    # 6 Linear 
    features7 = features6.dot(params['w.6'][elem_index].T) + params['b.6'][elem_index]
    return features7


class EANNForce:

    def __init__(self, 
                 n_elem, 
                 elem_indices, 
                 n_gto=12, 
                 rc=4, 
                 nipsin=2, 
                 beta=0.2, 
                 sizes=(64, 64), 
                 seed=12345):
        """ Constructor

        Parameters
        ----------
        n_elem: int
            Number of elements in the model.
        elem_indices: array of ints
            Element type of each atom in the system.
        n_gto: int
            Number of GTOs used in EANN.
        rc: float
            Cutoff distances, used to determine initial rs and inta.
        nipsin: int, optional
            Largest L in angular channel. Default 2
        beta: float, optional
            beta used to determine initial \Delta rs. Default 0.2
        sizes: tupple, ints, optional
            Number of hidden neurons in the model, the length is number of layers.
            Default (64, 64)
        seed: int, optional
            Seed for random number generator, default 12345

        Examples
        ----------

        """
        self.n_elem = n_elem
        self.n_gto = n_gto
        self.rc = rc
        self.beta = beta
        self.sizes = sizes
        self.n_layers = len(sizes)
        self.nipsin = nipsin
        self.elem_indices = elem_indices
        self.n_atoms = len(elem_indices)

        # n_elements * n_features
        self.n_features = (nipsin+1) * n_gto
        cij = jnp.ones((n_elem, n_gto)) * 0.0
        rs, inta = self.get_init_rs(n_gto, beta, rc)
        initpot = jnp.ones(1) * 0.0

        # initialize NN params
        key = jax.random.PRNGKey(seed)
        initializer = jax.nn.initializers.he_uniform()
        weights = []
        bias = []

        dim_in = self.n_features
        W = []
        B = []
        # Linear, LayerNorm, Relu_Like, Linear, LayerNorm, Relu_Like, Linear
        for i_layer in range(self.n_layers):
            dim_out = sizes[i_layer]
            key, subkey = jax.random.split(key)
            W.append(initializer(subkey, (n_elem, dim_in, dim_out)))
            B.append(jnp.zeros((n_elem, dim_out)))
            # LayerNorm
            W.append(initializer(subkey, (n_elem, dim_out)))
            B.append(jnp.zeros((n_elem, dim_out)))
            # Relu_like 
            W.append(initializer(subkey, (n_elem, 1, dim_out)))
            B.append(jnp.zeros((n_elem, 1, dim_out)))            
            dim_in = dim_out
        key, subkey = jax.random.split(key)
        W.append(initializer(subkey, (n_elem, dim_in)))
        key, subkey = jax.random.split(key)
        B.append(jax.random.uniform(subkey, shape=(n_elem,)))

        # prepare input parameters
        # weights: weights[i_layer][n_elem, dim_in, dim_out]
        # bias: bias[i_layer][n_elem, dim_out]
        params = OrderedDict()
        params = {
                'w': W,
                'b': B,
                'density.params': cij,
                'density.rs': rs,
                'density.inta': inta,
                'nnmod.initpot': initpot
                }
        self.params = params
        # prepare angular channels
        npara = [1]
        for i in range(1,self.nipsin+1):
            npara.append(3**i)
        self.index_para = jnp.concatenate([jnp.ones((npara[i],), dtype=jnp.int32) * i for i in range(len(npara))])

        # generate get_energy
        self.get_energy = self.generate_get_energy()

        return

    def get_init_rs(self, n_gto, beta, rc):
        """
        Generate initial values for rs and inta (exponents)

        Parameters
        ----------
        n_gto: int
            number of radial GTOs used in EANN
        beta: float
            beta used to determine initial \Delta rs. Default 0.2
        rc: float
            cutoff distance

        Returns
        ----------
        rs: 
            (3, n_gto): list of rs (for different radial channels)
        inta:
            (3, n_gto): list of inta
        """
        drs = rc / (n_gto - 1 + 0.3333333333)
        a = beta / drs / drs
        # rs = jnp.arange(0, rc, drs)
        # inta = jnp.ones(n_gto) * a
        rs=jnp.stack([jnp.arange(0, rc, drs) for itype in range(self.n_elem)],axis=0)
        inta=jnp.stack([jnp.ones(n_gto) * a for itype in range(self.n_elem)],axis=0)
        return rs, inta

    def get_features(self, radial, dr, pairs, buffer_scales, orb_coeff):
        """ Get atomic features from pairwise gto arrays
        
        Parameters
        ----------
        gtos(radial): array, (2, n_pairs, nipsin+1, n_gtos)
            pairwise gto values, that is, 
            cij * exp(-inta * (r-rs)**2) * 0.25*(cos(r/rc*pi) + 1)**2
        dr: array
            dr_vec for each pair, pbc shifted
        pairs: int array
            Indices of interacting pairs
        buffer_scales: float (0 or 1)
            neighbor list buffer masks

        Returns
        ----------
        features: (n_atom, n_features) array
            Atomic features

        Examples
        ----------
        """

        dist_vec = jnp.concatenate((dr,-dr),axis=0)
        dr_norm = jnp.linalg.norm(dist_vec, axis=1)
        f_cut = cutoff_cosine(dr_norm, self.rc)
        neigh_list = jnp.concatenate((pairs,pairs[:,[1,0]]),axis=0)
        buffer_scales_ = jnp.concatenate((buffer_scales,buffer_scales),axis=0)
        totneighbour = len(neigh_list)
        prefacs = f_cut.reshape(1, -1)
        angular = prefacs
        for ipsin in range(1,self.nipsin+1):
            prefacs = jnp.einsum("ji,ki->jki", prefacs, dist_vec.T).reshape(-1, totneighbour)
            angular = jnp.vstack((angular, prefacs))
        orbital = jnp.einsum("ji,ik->ijk", angular, radial)
        expandpara = orb_coeff[neigh_list[:,1],:] 
        worbital = jnp.einsum("ijk,ik,i->ijk", orbital, expandpara, buffer_scales_) 
        sum_worbital = jnp.zeros((self.n_atoms, orbital.shape[1], self.rs.shape[1]), dtype=orbital.dtype) 
        sum_worbital = sum_worbital.at[neigh_list[:,0], :, :].add(worbital)
        features = jnp.zeros((self.n_atoms, self.nipsin+1, self.rs.shape[1]), dtype=orbital.dtype) 
        features = features.at[:,self.index_para,:].add(jnp.square(sum_worbital)) 
        features = features.reshape(self.n_atoms,-1)
        return features


    def generate_get_energy(self):

        @jit_condition(static_argnums=())
        def get_energy(positions, box, pairs, params):
            """ Get energy
            This function returns the EANN energy.

            Parameters
            ----------
            positions: (n_atom, 3) array
                The positions of all atoms, in cartesian
            box: (3, 3) array
                The box array, arranged in rows
            pairs: jax_md nbl index
                The neighbor list, in jax_md.partition.OrderedSparse format
            params: dict
                The parameter dictionary, including the following keys:
                c: ${c_{ij}} of all exponent prefactors, (n_elem, n_elem)
                rs: distance shifts of all radial gaussian functions, (n_gto,)
                inta: the exponents, (n_gto,)
                w: weights of NN, list of (n_elem, dim_in, dime_out) array, with a length of n_layer
                b: bias of NN, list of (n_elem, dim_out) array, with a length of n_layer
            
            Returns:
            ----------
            energy: float or double
                EANN energy

            Examples:
            ----------
            """
            pairs = pairs[:,:2]
            pairs = regularize_pairs(pairs)
            buffer_scales = pair_buffer_scales(pairs)

            # get distances
            box_inv = jnp.linalg.inv(box + jnp.eye(3) * 1e-36)
            ri = distribute_v3(positions, pairs[:, 0])
            rj = distribute_v3(positions, pairs[:, 1])
            dr = rj - ri
            dr = pbc_shift(dr, box, box_inv)

            dr_norm = jnp.linalg.norm(dr, axis=1)
            buffer_scales2 = jnp.piecewise(buffer_scales, (dr_norm <= self.rc, dr_norm > self.rc),
                              (lambda x: jnp.array(1), lambda x: jnp.array(0)))
            buffer_scales = buffer_scales2 * buffer_scales
            self.rs = params['density.rs']
            self.inta = params['density.inta']

            radial_i, radial_j = get_gto(jnp.arange(len(dr_norm)), dr_norm, pairs, self.rc, self.rs, self.inta, self.elem_indices)
            radial = jnp.concatenate((radial_i,radial_j), axis=0)
            orb_coeff = params['density.params'][self.elem_indices,:] # (48,16)

            features = self.get_features(radial, dr, pairs, buffer_scales, orb_coeff)
            atomic_energies = get_atomic_energies(features, self.elem_indices, params)
            return jnp.sum(atomic_energies + params['nnmod.initpot'][0])

        return get_energy


def validation():
    import openmm.app as app
    import openmm.unit as unit
    from dmff.api import Hamiltonian
    from dmff.common import nblist
    # neighbor list
    H = Hamiltonian('peg.xml')
    app.Topology.loadBondDefinitions("residues.xml")
    pdb = app.PDBFile("peg4.pdb")
    rc = 0.4
    # generator stores all force field parameters
    pots = H.createPotential(pdb.topology, nonbondedCutoff=rc*unit.nanometer, ethresh=5e-4)

    # construct inputs
    pos = jnp.array(pdb.positions._value) # nm
    box = jnp.array(pdb.topology.getPeriodicBoxVectors()._value) # nm

    # neighbor list
    nbl = nblist.NeighborList(box, rc, pots.meta['cov_map']) 
    nbl.allocate(pos)
    pairs = nbl.pairs

    atomtype = ['H','C','O']
    n_elem = len(atomtype)
    species = []
    mass = []
    # Loop over all atoms in the topology
    for atom in pdb.topology.atoms():
        # Get the element of the atom
        element = atom.element.symbol
        mass.append(atom.element.mass._value)
        species.append(atomtype.index(atom.element.symbol))
    elem_indices = jnp.array(species)
    
    eann_force = EANNForce(n_elem, elem_indices, n_gto=16, nipsin=2, rc=4)
    # params = eann_force.params
    # E = eann_force.get_energy(pos*10, box*10, pairs, params)
    # print(E)
    with open('eann_model.pickle', 'rb') as f:
        params = pickle.load(f)
    E = eann_force.get_energy(pos*10, box*10, pairs, params)
    print(E)

if __name__ == '__main__':
    validation()
