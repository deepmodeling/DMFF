#!/usr/bin/env python
import sys
import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from jax import vmap, value_and_grad
from dmff.utils import jit_condition
from dmff.sgnn.graph import MAX_VALENCE
from dmff.sgnn.graph import TopGraph, from_pdb
import pickle
import re
from collections import OrderedDict
from functools import partial

class MolGNN:

    def __init__(self, G, n_layers=(3, 2), sizes=[(40, 20, 20), (20, 10)], nn=1, 
                 sigma=162.13039087945623, mu=117.41975505778706):
        self.nn = nn
        self.G = G
        self.G.get_all_subgraphs(nn, typify=True)
        self.G.prepare_subgraph_feature_calc()
        params = OrderedDict()
        params['w'] = jnp.array(np.random.random(1))
        self.n_layers = n_layers
        self.sizes = sizes
        dim_in = G.n_features
        for i_nn, n_layers in enumerate(n_layers):
            nn_name = 'fc%d'%i_nn
            params[nn_name + '.weight'] = []
            params[nn_name + '.bias'] = []
            for i_layer in range(n_layers):
                layer_name = nn_name + '.' + '%d'%i_layer
                dim_out = sizes[i_nn][i_layer]
                # params[layer_name+'.'+'weight'] = jnp.array(np.random.random((dim_out, dim_in)))
                # params[layer_name+'.'+'bias'] = jnp.array(np.random.random(dim_out))
                params[nn_name+'.weight'].append(jnp.array(np.random.random((dim_out, dim_in))))
                params[nn_name+'.bias'].append(jnp.array(np.random.random(dim_out)))
                dim_in = dim_out
        params['fc_final.weight'] = jnp.array(np.random.random((1, dim_in)))
        params['fc_final.bias'] = jnp.array(np.random.random(1))
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
                    f = jnp.tanh(params['fc0.weight'][i].dot(f) + params['fc0.bias'][i])
                return f

            @jit_condition(static_argnums=())
            @partial(vmap, in_axes=(0, None), out_axes=(0))
            def fc1(f_in, params):
                f = f_in
                for i in range(self.n_layers[1]):
                    f = jnp.tanh(params['fc1.weight'][i].dot(f) + params['fc1.bias'][i])
                return f

            @jit_condition(static_argnums=())
            @partial(vmap, in_axes=(0, None), out_axes=(0))
            def fc_final(f_in, params):
                return params['fc_final.weight'].dot(f_in) + params['fc_final.bias']

            # @jit_condition(static_argnums=(3))
            @partial(vmap, in_axes=(0, 0, None, None), out_axes=(0))
            def message_pass(f_in, nb_connect, w, nn):
                if nn == 0:
                    return f_in[0]
                elif nn == 1:
                    nb_connect0 = nb_connect[0:MAX_VALENCE-1]
                    nb_connect1 = nb_connect[MAX_VALENCE-1:2*(MAX_VALENCE-1)]
                    nb0 = jnp.sum(nb_connect0)
                    nb1 = jnp.sum(nb_connect1)
                    f = f_in[0] * (1 - jnp.heaviside(nb0, 0)*w - jnp.heaviside(nb1, 0)*w) + \
                        w * nb_connect0.dot(f_in[1:MAX_VALENCE, :]) / jnp.piecewise(nb0, [nb0<1e-5, nb0>=1e-5], [lambda x: jnp.array(1e-5), lambda x: x]) + \
                        w * nb_connect1.dot(f_in[MAX_VALENCE:2*MAX_VALENCE-1, :])/ jnp.piecewise(nb1, [nb1<1e-5, nb1>=1e-5], [lambda x: jnp.array(1e-5), lambda x: x])
                    return f

            features = fc0(features, params)
            features = message_pass(features, self.G.nb_connect, params['w'], self.G.nn)
            features = fc1(features, params)
            energies = fc_final(features, params)
            
            return self.G.weights.dot(energies)[0] * self.sigma + self.mu

        self.forward = partial(forward, nn=self.G.nn)
        self.batch_forward = vmap(self.forward, in_axes=(0, 0, None), out_axes=(0))

        return


    def load_params(self, ifn):
        with open(ifn, 'rb') as ifile:
            params = pickle.load(ifile)
        for k in params.keys():
            params[k] = jnp.array(params[k])
        # transform format
        keys = list(params.keys())
        for i_nn in [0, 1]:
            nn_name = 'fc%d'%i_nn
            keys_weight = []
            keys_bias = []
            for k in keys:
                if re.search(nn_name + '.[0-9]+.weight', k) is not None:
                    keys_weight.append(k)
                elif re.search(nn_name + '.[0-9]+.bias', k) is not None:
                    keys_bias.append(k)
            if len(keys_weight) != self.n_layers[i_nn] or len(keys_bias) != self.n_layers[i_nn]:
                sys.exit('Error while loading GNN params, inconsistent inputs with the GNN structure, check your input!')
            params['%s.weight'%nn_name] = []
            params['%s.bias'%nn_name] = []
            for i_layer in range(self.n_layers[i_nn]):
                k_w = '%s.%d.weight'%(nn_name, i_layer)
                k_b = '%s.%d.bias'%(nn_name, i_layer)
                params['%s.weight'%nn_name].append(params.pop(k_w, None))
                params['%s.bias'%nn_name].append(params.pop(k_b, None))
            # params[nn_name]
        self.params = params
        return 


    def save_params(self, ofn):
        # transform format
        params = {}
        params['w'] = self.params['w']
        params['fc_final.weight'] = self.params['fc_final.weight']
        params['fc_final.bias'] = self.params['fc_final.bias']
        for i_nn in range(2):
            nn_name = 'fc%d'%i_nn
            for i_layer in range(self.n_layers[i_nn]):
                params[nn_name+'.%d.weight'%i_layer] = self.params[nn_name+'.weight'][i_layer]
                params[nn_name+'.%d.bias'%i_layer] = self.params[nn_name+'.bias'][i_layer]
        with open(ofn, 'wb') as ofile:
            pickle.dump(params, ofile)
        return


def validation():
    # params = load_params('benchmark/model1.pickle')
    G = from_pdb('benchmark/peg4.pdb')
    model = MolGNN(G, nn=1)
    model.load_params('benchmark/model1.pickle')
    E = model.forward(G.positions, G.box, model.params)

    with open('benchmark/set009_remove_nb2.pickle', 'rb') as ifile:
        data = pickle.load(ifile)

    # pos = jnp.array(data['positions'][0:100])
    # box = jnp.tile(jnp.eye(3) * 50, (100, 1, 1))
    pos = jnp.array(data['positions'][0])
    box = jnp.eye(3) * 50

    # energies = model.batch_forward(pos, box, model.params)
    E, F = value_and_grad(model.forward, argnums=(0))(pos, box, model.params)
    F = -F
    print(E)
    print(F)


if __name__ == '__main__':
    validation()

