import jax
import jax.numpy as jnp
from jax import jit, value_and_grad, vmap, grad, vjp, tree_util, custom_jvp
from functools import partial
from .common.nblist import NeighborListFreud


class Loss_Generator:

    def __init__(self, f_nout, box, pos0, mass, dt, nsteps, nout, cov_map, rc, efunc):

        """ Constructor

        Parameters
        ----------
        f_nout: function
            Function of state user defined.
        box: jnp.ndarray
            Box of system, 3*3.
        pos0: jnp.ndarray
            Initial position used to allcate nblist.
        mass: jnp.ndarray
            Mass of each atom.
        dt: float
            Time step in simulation.
        nsteps: int
            Total steps in simulation.
        nout: int
            Get state in every nout steps.
        cov_map: jnp.ndarray
            Cov_map matrix.
        rc: float
            Cutoff distance in nblist.
        efunc: function
            Potential energy function.            

        Examples
        ----------

        """

        self.f_nout = f_nout
        self.box = box
        mass = jnp.tile(mass.reshape([len(mass), 1]), (1, 3))
        self.dt = dt
        self.nsteps = nsteps
        self.nout = nout
        nbl = NeighborListFreud(box, rc, cov_map)
        nbl.allocate(pos0)

        def return_pairs(pos):
            pos = jax.lax.stop_gradient(pos)
            nbl.update(pos)
            return nbl.pairs
        bonds = []
        for i in range(len(cov_map)):
            bonds.append(jnp.concatenate([jnp.array([i]), jnp.where(cov_map[i] > 0)[0]]))
        
        @jit
        def regularize_pos(pos):
            cpos = jnp.stack([jnp.sum(pos[bond], axis=0)/len(bond) for bond in bonds])
            box_inv = jnp.linalg.inv(box)
            spos = cpos.dot(box_inv)
            spos -= jnp.floor(spos)
            shift = spos.dot(box) - cpos
            return pos + shift
        
        self.states_axis = {'pos': 0, 'vel': 0}
        # use leap-frog Verlet integration method (v_0.5, x1) -> (v_1.5, x2)
        @jit
        def vv_step(state, params, pairs):
            x0 = state['pos']
            v0 = state['vel']
            f0 = -grad(efunc, argnums=(0))(x0, box, pairs, params)
            a0 = f0 / mass
            v1 = v0 + a0 * dt
            x1 = x0 + v1 * dt
            v1 = v1 - jnp.sum(v1*mass, axis=0)/jnp.sum(mass, axis=0)
            x1 = regularize_pos(x1)
            return {'pos': x1, 'vel':v1}   
        
        self.return_pairs = return_pairs
        self.regularize_pos = regularize_pos
        self.vv_step = vv_step

        return 
    

    def ode_fwd(self, state, params):
        """
        Run forward to get 'trajectory'

        Parameters
        ----------
        state: dict
            Initial state, {'pos': jnp.ndarray, 'vel': jnp.ndarray}
        params: dict
            Forcefield parameters

        Returns
        ----------
        state: dict
            Final state, {'pos': jnp.ndarray, 'vel': jnp.ndarray}
        traj: dict
            Save each 'state' in 'trajectory', {'time': jnp.ndarray, 'state': jnp.ndarray}
        """

        def fwd(state):
            for i in range(self.nout):
                pairs = jnp.stack([self.return_pairs(x) for x in state['pos']])
                state = vmap(self.vv_step, in_axes=(self.states_axis, None, 0), out_axes=(0))(state, params, pairs)
            return state
        traj = {}
        traj['time'] = jnp.zeros(self.nsteps//self.nout+1)
        traj0 = self.f_nout(state)
        traj['state'] = jnp.repeat(traj0[jnp.newaxis, ...], (self.nsteps//self.nout+1), axis=0)
        for i in range(self.nsteps//self.nout):
            state = fwd(state)
            traj['time'] = traj['time'].at[i+1].set(self.nout*self.dt*(i+1))
            traj['state'] = traj['state'].at[i+1].set(self.f_nout(state))
        return state, traj
  

    def _ode_bwd(self, state, params, gradient_traj):
        """
        Run backward to get final adjoint_state and gradient

        Parameters
        ----------
        state: dict
            Final state, {'pos': jnp.ndarray, 'vel': jnp.ndarray}
        params: dict
            Forcefield parameters
        gradient_traj: jnp.ndarray
            Derivatives of Loss with respect to 'state' in traj

        Returns
        ----------
        adjoint_state: dict
            Final adjoint state, {'pos': jnp.ndarray, 'vel': jnp.ndarray}
        gradient: dict
            Gradient of Loss with respect to params
        """
        def batch_vjp(state, params, pairs, adjoint_state):
            primals, vv_vjp = vjp(partial(self.vv_step, pairs=pairs), state, params)
            (grad_state, grad_params) = vv_vjp(adjoint_state)
            return grad_state, grad_params
        
        def bwd(state, adjoint_state, gradient):
            for i in range(self.nout):
                pairs = jnp.stack([self.return_pairs(x) for x in state['pos']])
                state = vmap(self.vv_step, in_axes=(self.states_axis, None, 0), out_axes=(0))(state, params, pairs)
                state['vel'] = - state['vel']
                state['pos'] = state['pos'] + state['vel']* self.dt
                state['pos'] = vmap(self.regularize_pos)(state['pos'])
                pairs = jnp.stack([self.return_pairs(x) for x in state['pos']])
                (grad_state, grad_params) = vmap(batch_vjp, in_axes=(self.states_axis, None, 0, self.states_axis))(state, params, pairs, adjoint_state)
                gradient = tree_util.tree_map(lambda p, u: p + jnp.sum(u, axis=0), gradient, grad_params)
                adjoint_state = grad_state  
                state['pos'] = state['pos'] - state['vel']*self.dt
                state['pos'] = vmap(self.regularize_pos)(state['pos'])
                state['vel'] = - state['vel']
            return state, adjoint_state, gradient
        primals, f_vjp = vjp(self.f_nout, state)
        adjoint_state = f_vjp(gradient_traj[-1])[0]
        gradient = tree_util.tree_map(jnp.zeros_like, params)
        # (v_1.5, x2) -> (-v_1.5, x1)
        state['pos'] = state['pos'] - state['vel']*self.dt
        state['pos'] = vmap(self.regularize_pos)(state['pos'])
        state['vel'] = - state['vel']
        for i in range(self.nsteps//self.nout):
            state, adjoint_state, gradient = bwd(state, adjoint_state, gradient)
            primals, f_vjp = vjp(self.f_nout, state)
            adjoint_state = {key: adjoint_state[key] + f_vjp(gradient_traj[-(i+2)])[0][key] for key in state}
        return adjoint_state, gradient
    
    def generate_Loss(self, L, has_aux=False, metadata=[]):
        """
        Generate Loss function

        Parameters
        ----------
        L:  function
            The 'Loss' function user defined, input: traj['state'], output: loss
        has_aux: bool
            If the L function returns auxiliary data
        metadata: []
            Record the traj and auxiliary data, {'traj':traj, 'aux_data':aux_data}
        
        Returns:
        ----------
        Loss: function
            Loss function

        Examples:
        ----------  
        """
        @custom_jvp
        def Loss(initial_state, params):
            """ 
            This function returns the loss.

            Parameters
            ----------
            initial_state: dict
                Initial state, {'pos': jnp.ndarray, 'vel': jnp.ndarray}
            params: dict
                The parameter dictionary.
            
            Returns:
            ----------
            loss: float 
                Loss

            Examples:
            ----------
            """
            final_state, traj = self.ode_fwd(initial_state, params)
            if has_aux == True:
                loss, aux_data = L(traj['state'])
                metadata.append({'traj':traj, 'aux_data':aux_data})
            else: 
                loss = L(traj['state'])
                metadata.append({'traj':traj})
            return loss
        
        @Loss.defjvp
        def _f_jvp(primals, tangents):
            x, y = primals
            x_dot, y_dot = tangents
            final_state, traj = self.ode_fwd(x, y)
            if has_aux == True:
                (primal_out, aux_data), gradient_traj = value_and_grad(L, has_aux=True)(traj['state'])
                metadata.append({'traj':traj, 'aux_data':aux_data})
            else:
                primal_out, gradient_traj = value_and_grad(L)(traj['state'])
                metadata.append({'traj':traj})
            adjoint_state, gradient = self._ode_bwd(final_state, y, gradient_traj)
            tangent_out = sum(tree_util.tree_leaves(tree_util.tree_map(lambda p, u: jnp.sum(p * u), adjoint_state, x_dot))) + sum(tree_util.tree_leaves(tree_util.tree_map(lambda p, u: jnp.sum(p * u), gradient, y_dot)))
            return primal_out, tangent_out
        
        return Loss
    
