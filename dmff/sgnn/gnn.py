import pickle
import re
import sys
from collections import OrderedDict
from functools import partial

import jax.lax as lax
import jax.nn.initializers
import jax.numpy as jnp
import numpy as np
from .graph import TopGraph, from_pdb
from .graph import MAX_VALENCE, ATYPE_INDEX, FSCALE_BOND, FSCALE_ANGLE
from ..utils import jit_condition
from jax import value_and_grad, vmap


def prm_transform_f2i(params, n_layers):
    p = {}
    for k in params:
        p[k] = jnp.array(params[k])
    for i_nn in [0, 1]:
        nn_name = 'fc%d' % i_nn
        p['%s.weight' % nn_name] = []
        p['%s.bias' % nn_name] = []
        for i_layer in range(n_layers[i_nn]):
            k_w = '%s.%d.weight' % (nn_name, i_layer)
            k_b = '%s.%d.bias' % (nn_name, i_layer)
            p['%s.weight' % nn_name].append(p.pop(k_w, None))
            p['%s.bias' % nn_name].append(p.pop(k_b, None))
    return p


def prm_transform_i2f(params, n_layers):
    # transform format
    p = {}
    p['w'] = params['w']
    p['fc_final.weight'] = params['fc_final.weight']
    p['fc_final.bias'] = params['fc_final.bias']
    for i_nn in range(2):
        nn_name = 'fc%d' % i_nn
        for i_layer in range(n_layers[i_nn]):
            p[nn_name + '.%d.weight' %
                   i_layer] = params[nn_name + '.weight'][i_layer]
            p[nn_name +
                   '.%d.bias' % i_layer] = params[nn_name +
                                                  '.bias'][i_layer]
    return p


class MolGNNForce:

    def __init__(self,
                 G,
                 n_layers=(3, 2),
                 sizes=[(40, 20, 20), (20, 10)],
                 nn=1,
                 sigma=162.13039087945623,
                 mu=117.41975505778706,
                 seed=12345,
                 max_valence=MAX_VALENCE,
                 atype_index=ATYPE_INDEX,
                 fscale_bond=FSCALE_BOND,
                 fscale_angle=FSCALE_ANGLE
                 ):
        """ Constructor for MolGNNForce

        Parameters
        ----------
        G: TopGraph object
            The topological graph object, created using dmff.sgnn.graph.TopGraph
        n_layers: int tuple, optional
            Number of hidden layers before and after message passing
            default = (3, 2)
        sizes: [tuple, tuple], optional
            sizes (numbers of hidden neurons) of the network before and after message passing
            default = [(40, 20, 20), (20, 10)]
        nn: int, optional
            size of the subgraphs, i.e., how many neighbors to include around the central bond
            default = 1
        sigma: float, optional
            final scaling factor of the energy.
            default = 162.13039087945623
        mu: float, optional
            a constant shift
            the final total energy would be ${(E_{NN} + \mu) * \sigma}
        seed: int, optional
            random seed used in network initialization
            default = 12345
        max_valence: int, optional
            Maximal valence number for all atoms inside the graph, use the value in graph.py by default
        atype_index: dict, optional
            A dictionary that assign index to each relevant element: e.g., {'H': 0, 'C': 1, 'O': 2}, use the ATYPE_INDEX in graph.py by default
        fscale_bond: float, optional
            The scaling factor for bond features, use value in graph.py by default
        fscale_angle: float, optional
            The scaling factor for angle features, use value in graph.py by default
        """
        self.nn = nn
        self.G = G
        self.G.get_all_subgraphs(nn, typify=True)
        self.G.prepare_subgraph_feature_calc(max_valence=max_valence,
                atype_index=atype_index,
                fscale_bond=fscale_bond,
                fscale_angle=fscale_angle)
        params = OrderedDict()
        key = jax.random.PRNGKey(seed)
        params['w'] = jax.random.uniform(key)
        self.n_layers = n_layers
        self.sizes = sizes
        dim_in = G.n_features
        initializer = jax.nn.initializers.he_uniform()
        for i_nn, n_layers in enumerate(n_layers):
            nn_name = 'fc%d' % i_nn
            params[nn_name + '.weight'] = []
            params[nn_name + '.bias'] = []
            for i_layer in range(n_layers):
                layer_name = nn_name + '.' + '%d' % i_layer
                dim_out = sizes[i_nn][i_layer]
                # params[nn_name+'.weight'].append(jnp.array(np.random.random((dim_out, dim_in))))
                # params[nn_name+'.bias'].append(jnp.array(np.random.random(dim_out)))
                key, subkey = jax.random.split(key)
                params[nn_name + '.weight'].append(
                    initializer(subkey, (dim_out, dim_in)))
                params[nn_name + '.bias'].append(jnp.zeros(dim_out))
                dim_in = dim_out
        key, subkey = jax.random.split(key)
        params['fc_final.weight'] = jnp.array(initializer(subkey, (1, dim_in)))
        key, subkey = jax.random.split(key)
        params['fc_final.bias'] = jax.random.uniform(subkey)
        self.params = params
        self.sigma = sigma
        self.mu = mu

        # generate the forward functions
        @jit_condition(static_argnums=3)
        def forward(positions, box, params, nn):
            features = self.G.calc_subgraph_features(positions, box)

            @jit_condition(static_argnums=())
            @partial(vmap, in_axes=(0, None), out_axes=(0))
            @partial(vmap, in_axes=(0, None), out_axes=(0))
            def fc0(f_in, params):
                f = f_in
                for i in range(self.n_layers[0]):
                    f = jnp.tanh(params['fc0.weight'][i].dot(f) +
                                 params['fc0.bias'][i])
                return f

            @jit_condition(static_argnums=())
            @partial(vmap, in_axes=(0, None), out_axes=(0))
            def fc1(f_in, params):
                f = f_in
                for i in range(self.n_layers[1]):
                    f = jnp.tanh(params['fc1.weight'][i].dot(f) +
                                 params['fc1.bias'][i])
                return f

            @jit_condition(static_argnums=())
            @partial(vmap, in_axes=(0, None), out_axes=(0))
            def fc_final(f_in, params):
                return params['fc_final.weight'].dot(
                    f_in) + params['fc_final.bias']

            # @jit_condition(static_argnums=(3))
            @partial(vmap, in_axes=(0, 0, None, None), out_axes=(0))
            def message_pass(f_in, nb_connect, w, nn):
                if nn == 0:
                    return f_in[0]
                elif nn == 1:
                    nb_connect0 = nb_connect[0:max_valence - 1]
                    nb_connect1 = nb_connect[max_valence - 1:2 *
                                             (max_valence - 1)]
                    nb0 = jnp.sum(nb_connect0)
                    nb1 = jnp.sum(nb_connect1)
                    f = f_in[0] * (1 - jnp.heaviside(nb0, 0)*w - jnp.heaviside(nb1, 0)*w) + \
                        w * nb_connect0.dot(f_in[1:max_valence, :]) / jnp.piecewise(nb0, [nb0<1e-5, nb0>=1e-5], [lambda x: jnp.array(1e-5), lambda x: x]) + \
                        w * nb_connect1.dot(f_in[max_valence:2*max_valence-1, :])/ jnp.piecewise(nb1, [nb1<1e-5, nb1>=1e-5], [lambda x: jnp.array(1e-5), lambda x: x])
                    return f

            features = fc0(features, params)
            features = message_pass(features, self.G.nb_connect, params['w'],
                                    self.G.nn)
            features = fc1(features, params)
            energies = fc_final(features, params)

            return self.G.weights.dot(energies)[0] * self.sigma + self.mu

        self.forward = partial(forward, nn=self.G.nn)
        self.batch_forward = vmap(self.forward,
                                  in_axes=(0, 0, None),
                                  out_axes=(0))

        # provide the get_energy function, to be consistent with the other parts of DMFF
        self.get_energy = self.forward

        return


    def load_params(self, ifn):
        """ Load the network parameters from saved file

        Parameters
        ----------
        ifn: string
            the input file name

        """
        with open(ifn, 'rb') as ifile:
            params = pickle.load(ifile)
        for k in params.keys():
            params[k] = jnp.array(params[k])
        # transform format
        self.params = prm_transform_f2i(params, self.n_layers)
        return

    


    def save_params(self, ofn):
        """ Save the network parameters to a pickle file

        Parameters
        ----------
        ofn: string
            the output file name

        """
        # transform format
        params = prm_transform_i2f(self.params, self.n_layers)
        with open(ofn, 'wb') as ofile:
            pickle.dump(params, ofile)
        return

