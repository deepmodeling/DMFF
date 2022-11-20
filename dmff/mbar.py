import numpy as np
import mdtraj as md
from pymbar import MBAR
import dmff
dmff.update_jax_precision(dmff.PRECISION)
import jax
import jax.numpy as jnp
from jax import grad
from tqdm import tqdm, trange
import openmm as mm
import openmm.app as app 
import openmm.unit as unit


class TargetState:
    def __init__(self, temperature, energy_function):
        self._temperature = temperature
        self._efunc = energy_function

    def calc_energy(self, trajectory, parameters):
        beta = 1. / self._temperature / 8.314 * 1000.
        eners = self._efunc(trajectory, parameters)
        ulist = jnp.concatenate([beta * e.reshape((1, )) for e in eners])
        return ulist


class SampleState:
    def __init__(self, temperature, name):
        self._temperature = temperature
        self.name = name

    def calc_energy_frame(self, frame):
        return 0.0

    def calc_energy(self, trajectory):
        # return beta * u
        beta = 1. / self._temperature / 8.314 * 1000.
        eners = []
        for frame in tqdm(trajectory):
            e = self.calc_energy_frame(frame)
            eners.append(e * beta)
        return jnp.array(eners)


class OpenMMSampleState(SampleState):
    def __init__(self,
                 name,
                 parameter,
                 topology,
                 temperature=300.0,
                 pressure=0.0):
        super(OpenMMSampleState, self).__init__(temperature, name)
        self._pressure = pressure
        # create a context
        pdb = app.PDBFile(topology)
        ff = app.ForceField(parameter)
        system = ff.createSystem(pdb.topology,
                                 nonbondedMethod=app.PME,
                                 nonbondedCutoff=0.9 * unit.nanometer,
                                 constraints=None,
                                 rigidWater=False)

        for force in system.getForces():
            if isinstance(force, mm.NonbondedForce):
                force.setUseDispersionCorrection(False)
                force.setUseSwitchingFunction(False)

        integ = mm.LangevinIntegrator(0 * unit.kelvin, 5 / unit.picosecond,
                                      1.0 * unit.femtosecond)
        self.ctx = mm.Context(system, integ)

    def calc_energy_frame(self, frame):
        self.ctx.setPositions(frame.openmm_positions(0))
        self.ctx.setPeriodicBoxVectors(*frame.openmm_boxes(0))
        state = self.ctx.getState(getEnergy=True)
        vol = frame.unitcell_volumes[0]  # in nm^3
        ener = state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole) + 0.06023 * vol * self._pressure
        return ener


class Sample:
    def __init__(self, trajectory, from_state):
        self.trajectory = trajectory
        self.from_state = from_state
        self.energy_data = {}

    def generate_energy(self, state_list):
        for state in state_list:
            if state.name not in self.energy_data:
                self.energy_data[state.name] = np.array(
                    [state.calc_energy(self.trajectory)])


class MBAREstimator:
    def __init__(self):
        self.samples = []
        self.states = []
        self._mbar = None
        self._umat = None
        self._nk = None
        self._full_samples = None

    def add_sample(self, sample):
        self.samples.append(sample)

    def add_state(self, state):
        self.states.append(state)

    def remove_sample(self, name):
        init_num = len(self.samples)
        self.samples = [s for s in self.samples if s.from_state != name]
        final_num = len(self.samples)
        assert init_num > final_num

    def remove_state(self, name):
        init_num = len(self.states)
        self.states = [s for s in self.states if s.name != name]
        final_num = len(self.states)
        assert init_num > final_num
        self.remove_sample(name)

    def compute_energy_matrix(self):
        for sample in self.samples:
            sample.generate_energy(self.states)

    def _build_umat(self):
        nk_states = {state.name: 0 for state in self.states}
        for sample in self.samples:
            nk_states[sample.from_state] += sample.trajectory.n_frames
        nk_names = [k.name for k in self.states]
        nk = np.array([nk_states[k] for k in nk_states.keys()])
        umat = np.zeros((nk.shape[0], nk.sum()))
        istart = 0
        traj_merge = []
        for nk_name in nk_names:
            for sample in [s for s in self.samples if nk_name == s.from_state]:
                traj_merge.append(sample.trajectory)
                sample_frames = sample.trajectory.n_frames
                iend = istart + sample_frames
                for nnk, nk_name2 in enumerate(nk_names):
                    umat[nnk, istart:iend] = sample.energy_data[nk_name2]
                istart = iend
        return umat, nk, md.join(traj_merge)

    def optimize_mbar(self, initialize="BAR"):
        self.compute_energy_matrix()
        umat, nk, samples = self._build_umat()
        self._umat = umat
        self._nk = nk
        self._full_samples = samples

        self._mbar = MBAR(self._umat, self._nk, initialize=initialize)
        self._umat_jax = jax.numpy.array(self._umat)
        self._free_energy_jax = jax.numpy.array(self._mbar.f_k)
        self._nk_jax = jax.numpy.array(nk)

    def estimate_weight(self,
                        state,
                        parameters=None,
                        decompose=True,
                        return_energy=True):
        if isinstance(state, TargetState):
            unew = state.calc_energy(self._full_samples, parameters)
        else:
            unew = state.calc_energy(self._full_samples)
        unew_max = unew.max()
        du_1 = self._free_energy_jax.reshape((-1, 1)) - self._umat_jax
        delta_u = du_1 + unew.reshape((1, -1)) - unew_max - du_1.min()
        cm = 1. / (jax.numpy.exp(delta_u) * jax.numpy.array(self._nk).reshape(
            (-1, 1))).sum(axis=0)
        weight = cm / cm.sum()
        if return_energy:
            return weight, unew
        return weight

    def _estimate_weight_numpy(self, unew_npy, return_cn=False):
        unew_mean = unew_npy.mean()
        du_1 = self._mbar.f_k.reshape((-1, 1)) - self._umat
        delta_u = du_1 + unew_npy.reshape((1, -1)) - unew_mean - du_1.mean()
        cn = 1. / (np.exp(delta_u) * self._nk.reshape((-1, 1))).sum(axis=0)
        weight = cn / cn.sum()
        if return_cn:
            return weight, cn
        else:
            return weight

    def _computeCovar(self, W, N_k):
        K, N = W.shape
        Ndiag = np.diag(N_k)
        I = np.identity(K, dtype=np.float64)

        S2, V = np.linalg.eigh(W @ W.T)
        S2[np.where(S2 < 0.0)] = 0.0
        Sigma = np.diag(np.sqrt(S2))

        # Compute covariance
        Theta = (V @ Sigma @ np.linalg.pinv(
            I - Sigma @ V.T @ Ndiag @ V @ Sigma, rcond=1e-10) @ Sigma @ V.T)
        return Theta

    def estimate_effective_sample(self, unew, decompose=False):
        wnew, cn = self._estimate_weight_numpy(unew, return_cn=True)
        eff_samples = 1. / (wnew**2).sum()
        if decompose:
            state_effect = {}
            argsort = np.argsort(wnew)[::-1][:int(eff_samples)]
            for nstate in range(len(self.states)):
                istart = self._nk[:nstate].sum()
                iend = istart + self._nk[nstate]
                state_effect[self.states[nstate].name] = (
                    (argsort > istart) & (argsort < iend)).sum()
            state_effect["Total"] = eff_samples
            return state_effect
        return eff_samples

    def _estimate_free_energy(self, unew):
        a = self._free_energy_jax - self._umat_jax.T
        # log(sum(n_k*exp(a)))
        a_max = a.max(axis=1, keepdims=True)
        log_denominator_n = jnp.log((self._nk_jax.reshape(
            (1, -1)) * jnp.exp(a - a_max)).sum(axis=1)) + a_max.reshape((-1, ))
        a2 = -unew - log_denominator_n
        # log(sum(exp(a2)))
        a2_max = a2.max()
        f_new = -jnp.log(jnp.sum(jnp.exp(a2 - a2_max))) - a2_max
        return f_new

    def estimate_free_energy_difference(self,
                                        target_state,
                                        ref_state,
                                        target_parameters=None,
                                        ref_parameters=None,
                                        decompose=True,
                                        return_energy=False):
        # compute F_target - F_ref
        if isinstance(ref_state, TargetState):
            u_ref = ref_state.calc_energy(self._full_samples, ref_parameters)
        else:
            u_ref = ref_state.calc_energy(self._full_samples)
        if isinstance(target_state, TargetState):
            u_target = target_state.calc_energy(self._full_samples,
                                                target_parameters)
        else:
            u_target = target_state.calc_energy(self._full_samples)
        f_ref = self._estimate_free_energy(u_ref)
        f_target = self._estimate_free_energy(u_target)
        if return_energy:
            return f_target - f_ref, u_target, u_ref
        return f_target - f_ref