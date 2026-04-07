import numpy as np
import warnings
try:
    import mdtraj as md
except ImportError:
    warnings.warn("MDTraj not installed. MBAREstimator is not available.")

try:
    from pymbar import MBAR
except ImportError:
    MBAR = None
    warnings.warn("MBAR not installed, MBAREstimator for multiple states is not available.")

try:
    import base
    from base.inference.calculator import BaseCalculator
except ImportError:
    warnings.warn("base not installed, base-related functions are not available")

try:
    import ase
    from ase import Atoms
    from ase.io.lammpsdata import write_lammps_data
except ImportError:
    warnings.warn("ASE not installed, related functions are not available")
    
try:
    import torch
except ImportError:
    warnings.warn("torch not installed, related functions are not available")


from .settings import update_jax_precision, PRECISION
update_jax_precision(PRECISION)
import jax
import jax.numpy as jnp
from jax import grad
from tqdm import tqdm, trange
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from .common.nblist import NeighborListFreud
from .common.constants import EV2KJ, A2NM
from collections import defaultdict
import subprocess
from pathlib import Path
from .torch_tools import j2t_pytree


def buildTrajEnergyFunction(
    potential_func,
    cov_map,
    cutoff,
    usePBC=True,
    useFreud=True,
    ensemble="nvt",
    pressure=1.0,
):
    def energy_function(traj, parameters):
        pos_list, box_list, pairs_list, vol_list = [], [], [], []
        pair_full = []
        for na in range(traj.topology.n_atoms):
            for nb in range(na + 1, traj.topology.n_atoms):
                pair_full.append([na, nb, 0])
        pair_full = np.array(pair_full, dtype=int)
        pair_full[:, 2] = cov_map[pair_full[:, 0], pair_full[:, 1]]
        for frame in tqdm(traj):
            aa, bb, cc = frame.openmm_boxes(0).value_in_unit(unit.nanometer)
            box = jnp.array(
                [[aa[0], aa[1], aa[2]], [bb[0], bb[1], bb[2]], [cc[0], cc[1], cc[2]]]
            )
            positions = jnp.array(frame.xyz[0, :, :])
            if usePBC:
                nbobj = NeighborListFreud(box, cutoff, cov_map)
                nbobj.capacity_multiplier = 1
                pairs = nbobj.allocate(positions)
                pairs_list.append(pairs)
            else:
                pairs_list.append(pair_full)
        pos_list = jnp.array(traj.xyz)
        vol_list = jnp.array(traj.unitcell_volumes)
        box_list = jnp.array(traj.unitcell_vectors)

        pmax = max([p.shape[0] for p in pairs_list])
        pairs_jax = np.zeros((traj.n_frames, pmax, 3), dtype=int) + traj.n_atoms
        for nframe in range(traj.n_frames):
            pair = pairs_list[nframe]
            pairs_jax[nframe, : pair.shape[0], :] = pair[:, :]
        pairs_jax = jax.numpy.array(pairs_jax)
        if ensemble.upper() == "NVT":
            ensemble_cns = 0.0
        elif ensemble.upper() == "NPT":
            ensemble_cns = 1.0
        eners = [
            potential_func(pos_list[i], box_list[i], pairs_jax[i], parameters)
            + ensemble_cns * pressure * 0.06023 * vol_list[i]
            for i in trange(traj.n_frames)
        ]
        return eners

    return energy_function


def buildFrameEnergyFunction(
    potential_func,
    cov_map=None,
    cutoff=None,
    builtin_nbl: bool = False,
):
    """Wrap a DMFF potential into a frame-level energy function.

    This is the frame-based counterpart of :func:`buildTrajEnergyFunction`.

    Parameters
    ----------
    potential_func : callable
        Low-level DMFF potential with signature
        ``potential_func(positions, box, pairs, parameters)``.
    cov_map : array-like, optional
        Covalent map used to annotate neighbor-list pairs; required when
        ``builtin_nbl=Flase``.
    cutoff : float, optional
        Cutoff radius in nanometers for building the neighbor list; required
        when ``builtin_nbl=False``.
    builtin_nbl : bool, optional
        If ``False``, build a neighbor list using :class:`NeighborListFreud`
        from ``cov_map``, ``box``, and ``positions`` (typical for classical
        force fields). If ``False``, pass ``pairs=None`` to the potential,
        which is the usual case for ML potentials that manage neighbor lists
        internally.

    Returns
    -------
    energy_function : callable
        A function ``energy_function(frame, parameters)`` that takes a
        single mdtraj frame (``frame``) and a parameter dict, returning the
        potential energy in kJ/mol.
    """

    if not builtin_nbl:
        if cov_map is None or cutoff is None:
            raise ValueError(
                "cov_map and cutoff must be provided when builtin_nbl=True in buildFrameEnergyFunction."
            )

        def energy_function(frame, parameters):
            # frame is an mdtraj.Trajectory with a single frame
            box = jnp.array(frame.unitcell_vectors[0])  # (3, 3) in nm
            positions = jnp.array(frame.xyz[0, :, :])  # (n_atoms, 3) in nm
            nbobj = NeighborListFreud(box, cutoff, cov_map)
            nbobj.capacity_multiplier = 1
            pairs = nbobj.allocate(positions)
            return potential_func(positions, box, pairs, parameters)

    else:

        def energy_function(frame, parameters):
            box = jnp.array(frame.unitcell_vectors[0])
            positions = jnp.array(frame.xyz[0, :, :])
            pairs = None
            return potential_func(positions, box, pairs, parameters)

    return energy_function


class TargetState:
    def __init__(
        self,
        temperature,
        energy_function,
        pressure: float = 0.0,
        mu_dict=None,
        legacy: bool = True,
    ):
        """State describing a target ensemble for reweighting.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.
        energy_function : callable
            If ``legacy=True`` (default), this is a trajectory-level function
            ``energy_function(traj, parameters)`` that returns a list or array
            of potential energies (in kJ/mol) for each frame in a trajectory.
            This is the behavior relied on by existing code and tests.

            If ``legacy=False``, this is a frame-level function
            ``energy_function(frame, parameters)`` that operates on individual
            mdtraj frames. In that mode, PV and μ·N contributions are handled
            in :meth:`calc_energy`.
        pressure : float, optional
            Pressure in bar. Used only when ``legacy=False``.
        mu_dict : dict, optional
            Mapping atom names to chemical potentials. Used only when
            ``legacy=False``.
        legacy : bool, optional
            When True, preserve the original trajectory-level behavior. When
            False, use the frame-level API compatible with finetune/ft.py.
        """

        self._temperature = temperature
        self._efunc = energy_function
        self._pressure = pressure
        self._mu_dict = mu_dict if isinstance(mu_dict, dict) else defaultdict(float)
        self._legacy = legacy

    def calc_energy(self, trajectory, parameters):
        beta = 1.0 / self._temperature / 8.314 * 1000.0

        if self._legacy:
            # Original behavior: energy_function operates on the full
            # trajectory and returns per-frame energies. PV and μ·N are
            # assumed to be handled inside the energy_function.
            eners = self._efunc(trajectory, parameters)
            ulist = jnp.concatenate([beta * e.reshape((1,)) for e in eners])
            return ulist

        # New behavior: energy_function operates on individual frames.
        # trajectory is assumed to be an mdtraj.Trajectory.
        #
        # Note: self._efunc may return a scalar or a length-1 array for each
        # frame. We therefore flatten to a 1D array of length n_frames to
        # maintain compatibility with downstream code and tests.
        eners = jnp.array([
            self._efunc(trajectory[i : i + 1], parameters)
            for i in range(trajectory.n_frames)
        ]).reshape((trajectory.n_frames,))

        # PV term using unitcell_volumes (nm^3)
        if hasattr(trajectory, "unitcell_volumes"):
            eners = eners + 0.06023 * self._pressure * jnp.array(
                trajectory.unitcell_volumes
            )

        # μ·N term from topology atom names
        if hasattr(trajectory, "topology") and hasattr(trajectory.topology, "atoms"):
            mu_contrib = jnp.sum(jnp.array([
                self._mu_dict[a.name] for a in trajectory.topology.atoms
            ]))
            # Grand-canonical reduced potential: u = β(E + PV - μN).
            # Therefore the μN contribution enters with a minus sign.
            eners = eners - mu_contrib

        eners = eners * beta
        return eners


class SampleState:
    def __init__(
        self,
        temperature,
        name,
        pressure: float = 0.0,
        mu_dict=None,
        legacy: bool = True,
    ):
        """Base class for sampling states.

        Parameters
        ----------
        temperature : float
            Temperature in Kelvin.
        name : str
            Name identifying this state (used for MBAR bookkeeping).
        pressure : float, optional
            Pressure in bar. Used only when ``legacy=False``.
        mu_dict : dict, optional
            Mapping atom names to chemical potentials. Used only when
            ``legacy=False``.
        legacy : bool, optional
            When True, preserve the original behavior where PV and μ·N are
            handled in :meth:`calc_energy_frame` or elsewhere. When False, use
            a frame-level API that mirrors finetune/ft.py semantics.
        """

        self._temperature = temperature
        self.name = name
        self._pressure = pressure
        self._mu_dict = mu_dict if isinstance(mu_dict, dict) else defaultdict(float)
        self._legacy = legacy

    def calc_energy_frame(self, frame):
        return 0.0

    def calc_energy(self, trajectory):
        # return beta * u
        beta = 1.0 / self._temperature / 8.314 * 1000.0

        if self._legacy:
            # Original behavior: subclasses (e.g. OpenMMSampleState) are
            # expected to include PV contributions in calc_energy_frame.
            eners = []
            for frame in tqdm(trajectory):
                e = self.calc_energy_frame(frame)
                eners.append(e * beta)
            return jnp.array(eners)

        # New behavior: frame-level potential + PV and μ·N handled here.
        eners = jnp.array([
            self.calc_energy_frame(frame) for frame in tqdm(trajectory)
        ])

        if hasattr(trajectory, "unitcell_volumes"):
            eners = eners + 0.06023 * self._pressure * jnp.array(
                trajectory.unitcell_volumes
            )

        if hasattr(trajectory, "topology") and hasattr(trajectory.topology, "atoms"):
            mu_contrib = jnp.sum(jnp.array([
                self._mu_dict[a.name] for a in trajectory.topology.atoms
            ]))
            # Grand-canonical reduced potential: u = β(E + PV - μN).
            # Therefore the μN contribution enters with a minus sign.
            eners = eners - mu_contrib

        eners = eners * beta
        return eners

    def sample(self, *args, **kwargs):
        """Run MD and generate a :class:`Sample`.

        Subclasses representing concrete simulators (for example
        :class:`OpenMMSampleState`, :class:`ASENNPNPTSampleState`, or
        :class:`LammpsNNPNPTSampleState`) should override this method.
        """

        raise NotImplementedError

    def update_parameters(self, *args, **kwargs):
        """Update internal model parameters of the state.

        Concrete subclasses that wrap ML potentials are expected to override
        this (for example, to reload a checkpoint).
        """

        raise NotImplementedError


class OpenMMSampleState(SampleState):
    def __init__(
        self,
        name,
        parameter,
        topology,
        temperature=300.0,
        pressure=0.0,
        useDispersionCorrection=False,
        useSwitchingFunction=False,
        platform="CPU",
        properties={},
        legacy: bool = True,
        mu_dict=None,
        **args
    ):
        # ``legacy`` retains the original behavior used in the tests by
        # default. When ``legacy=False``, PV and μ·N are handled in
        # :meth:`SampleState.calc_energy` instead.
        super(OpenMMSampleState, self).__init__(
            temperature=temperature,
            name=name,
            pressure=pressure,
            mu_dict=mu_dict,
            legacy=legacy,
        )
        self._pressure = pressure
        # create a context
        pdb = app.PDBFile(topology)
        ff = app.ForceField(parameter)

        # default settings
        if "nonbondedMethod" not in args:
            args["nonbondedMethod"] = app.PME
        if "nonbondedCutoff" not in args:
            args["nonbondedCutoff"] = 0.9 * unit.nanometer
        if "constraints" not in args:
            args["constraints"] = None
        if "rigidWater" not in args:
            args["rigidWater"] = False
        system = ff.createSystem(pdb.topology, **args)

        platform = mm.Platform.getPlatformByName(platform)
        platform_properties = properties

        for force in system.getForces():
            if isinstance(force, mm.NonbondedForce):
                force.setUseDispersionCorrection(useDispersionCorrection)
                force.setUseSwitchingFunction(useSwitchingFunction)

        integ = mm.LangevinIntegrator(
            0 * unit.kelvin, 5 / unit.picosecond, 1.0 * unit.femtosecond
        )
        self.ctx = mm.Context(system, integ, platform, platform_properties)

    def calc_energy_frame(self, frame):
        self.ctx.setPositions(frame.openmm_positions(0))
        self.ctx.setPeriodicBoxVectors(*frame.openmm_boxes(0))
        state = self.ctx.getState(getEnergy=True)
        vol = frame.unitcell_volumes[0]  # in nm^3
        ener = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        # For legacy mode, include the PV contribution here to keep the
        # original behavior expected by existing tests. For non-legacy mode,
        # PV is added in :meth:`SampleState.calc_energy`.
        if self._legacy:
            ener = ener + 0.06023 * vol * self._pressure
        return ener


class ASENNPNPTSampleState(SampleState):
    """NPT sampling state driven by an ASE calculator wrapping a DMFF model.

    This class mirrors :class:`SampleState` and ``ASENNPNPTSampleState`` in
    ``finetune/ft.py``, but is integrated into the core MBAR module.
    """

    def __init__(
        self,
        temperature,
        name,
        init_model_path,
        config_path,
        ffname,
        e_eval_loader,
        cutoff=0.5,
        pressure=0.0,
    ):
        # Use non-legacy mode so that PV and μ·N are handled in
        # SampleState.calc_energy, matching finetune semantics.
        super().__init__(
            temperature=temperature,
            name=name,
            pressure=pressure,
            mu_dict=0.0,
            legacy=False,
        )
        self.ffname = ffname
        self.e_eval_loader = e_eval_loader
        self.config_path = config_path
        # cutoff is provided in Angstrom; convert to nm
        self.cutoff = cutoff / A2NM
        self.e_eval = self.e_eval_loader(init_model_path, self.config_path, self.cutoff)

    def calc_energy_frame(self, frame):
        from ase import Atoms

        atoms = Atoms(
            numbers=[a.element.atomic_number for a in frame.topology.atoms],
            cell=frame.unitcell_vectors.reshape(3, 3) / A2NM,
            positions=frame.xyz.reshape(-1, 3) / A2NM,
            pbc=[1, 1, 1],
        )
        atoms.calc = self.e_eval
        energy = atoms.get_potential_energy()
        return energy * EV2KJ

    def load_init_struct(self, init_atoms, top_file_name):
        self.init_atoms = init_atoms
        self.top_file_name = top_file_name

    def sample(self, nsteps, interval, skip, timestep, ttime, pfactor, file_name):
        """Run ASE NPT MD and return a :class:`Sample`.

        The interface matches ``ASENNPNPTSampleState.sample`` in
        ``finetune/ft.py``.
        """

        from copy import deepcopy
        from collections.abc import Sequence
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        from ase.md.npt import NPT
        from ase import units

        class TrajectoryObserver(Sequence):
            def __init__(self, atoms):
                self.atoms = atoms
                self.xyz = []
                self.unitcell_lengths = []
                self.unitcell_angles = []

            def __call__(self):
                cell = self.atoms.get_cell()
                self.xyz.append(self.atoms.get_positions())
                self.unitcell_lengths.append(cell.lengths())
                self.unitcell_angles.append(cell.angles())

            def __getitem__(self, item):
                return (
                    self.xyz[item],
                    self.unitcell_lengths[item],
                    self.unitcell_angles[item],
                )

            def __len__(self):
                return len(self.xyz)

            def save(self, top_file_name, filename, skip):
                traj = md.Trajectory(
                    xyz=self.xyz[skip:],
                    topology=md.load_topology(top_file_name),
                    unitcell_lengths=self.unitcell_lengths[skip:],
                    unitcell_angles=self.unitcell_angles[skip:],
                )
                traj.save_dcd(filename)

        init_atoms = deepcopy(self.init_atoms)
        init_atoms.calc = self.e_eval
        MaxwellBoltzmannDistribution(init_atoms, temperature_K=self._temperature)
        dyn = NPT(
            init_atoms,
            timestep=timestep * units.fs,
            temperature_K=self._temperature,
            externalstress=self._pressure * units.bar,
            ttime=ttime * units.fs,
            pfactor=pfactor * units.GPa * (units.fs**2),
            logfile=None,
            loginterval=1,
        )
        obs = TrajectoryObserver(init_atoms)
        dyn.attach(obs, interval=interval)
        dyn.run(nsteps)
        obs.save(self.top_file_name, file_name, skip=int(skip / interval))
        traj = md.load_dcd(file_name, top=self.top_file_name)
        traj.xyz = jnp.array(traj.xyz).astype(jnp.float32) * A2NM
        traj.unitcell_lengths = jnp.array(traj.unitcell_lengths).astype(jnp.float32) * A2NM
        traj.unitcell_angles = jnp.array(traj.unitcell_angles).astype(jnp.float32)
        return Sample(traj, from_state=self.name)

    def output_parameters(self, H, params, state_dict_name):
        for g in H.getGenerators():
            if g.getName() == self.ffname:
                generator = g
                break
        else:
            raise ValueError(f"Generator with name {self.ffname} not found in Hamiltonian.")
        generator.write_to(params=params, state_dict_file=state_dict_name)

    def update_parameters(self, ckpt_name):
        self.e_eval = self.e_eval_loader(ckpt_name, self.config_path, self.cutoff)


class LammpsNNPNPTSampleState(SampleState):
    """NPT sampling state driven by a LAMMPS runner and DMFF ML potential."""

    def __init__(
        self,
        temperature,
        name,
        ckpt_path,
        config_path,
        ffname,
        e_eval_loader,
        cutoff=0.5,
        pressure=0.0,
    ):
        # Use non-legacy mode for PV and μ·N handling.
        super().__init__(
            temperature=temperature,
            name=name,
            pressure=pressure,
            mu_dict=0.0,
            legacy=False,
        )
        self.ffname = ffname
        self.e_eval_loader = e_eval_loader
        self.config_path = config_path
        self.cutoff = cutoff / A2NM
        self.ckpt_path = ckpt_path
        self.e_eval = self.e_eval_loader(ckpt_path, self.config_path, self.cutoff)

    def calc_energy_frame(self, frame):
        from ase import Atoms

        atoms = Atoms(
            numbers=[a.element.atomic_number for a in frame.topology.atoms],
            cell=frame.unitcell_vectors.reshape(3, 3) / A2NM,
            positions=frame.xyz.reshape(-1, 3) / A2NM,
            pbc=[1, 1, 1],
        )
        atoms.calc = self.e_eval
        energy = atoms.get_potential_energy()
        return energy * EV2KJ

    def load_init_struct(self, init_atoms_lammps_data, ele_list, top_file_name):
        self.init_atoms_lammps_data = init_atoms_lammps_data
        self.ele_list = ele_list
        self.top_file_name = top_file_name

    def load_md_runner(self, run_md):
        """Attach a callable that runs LAMMPS given an input script path."""

        self.run_md = run_md

    def sample(self, nsteps, interval, skip, timestep, ttime, ptime, file_name):
        """Run LAMMPS NPT MD via an external runner and return a Sample."""

        from pathlib import Path
        import MDAnalysis as mda

        lines = [
            "units metal",
            "dimension 3",
            "boundary p p p",
            "atom_style atomic",
            "",
            f"read_data {self.init_atoms_lammps_data}",
            "",
            f"pair_style base {self.cutoff}",
            f"pair_coeff * * {self.ckpt_path} " + " ".join(self.ele_list),
            "",
            f"timestep {timestep * 1e-3}",  # metal uses ps
            f"velocity all create {self._temperature} 12345 loop all mom yes rot no dist gaussian",
            f"fix npt_fix all npt temp {self._temperature} {self._temperature} {ttime * 1e-3} tri {self._pressure} {self._pressure} {ptime * 1e-3}",
            "",
            "thermo 100",
            "thermo_style custom step temp pe ke etotal press vol",
            f"run {skip}",
            f"dump dcd_dump all dcd {interval} {file_name}",
            "dump_modify dcd_dump sort id",
            f"run {nsteps - skip}",
        ]
        in_lammps_dir = str(Path(self.ckpt_path).parent)
        in_lammps = in_lammps_dir + "/in.lammps"
        with open(in_lammps, "w") as f:
            for line in lines:
                f.write(line + "\n")

        # Delegate actual MD execution to the provided runner
        self.run_md(in_lammps)

        # Use MDAnalysis to read .dcd and build mdtraj.Trajectory to prevent
        # automatic unit changes when reading directly with mdtraj
        u = mda.Universe(self.top_file_name, file_name)
        xyz, unitcell_lengths, unitcell_angles = [], [], []
        for frame in u.trajectory:
            xyz.append(frame.positions)
            unitcell_lengths.append(frame.dimensions[:3])
            unitcell_angles.append(frame.dimensions[3:])
        traj = md.Trajectory(
            xyz=jnp.array(xyz).astype(jnp.float32),
            topology=md.load_topology(self.top_file_name),
            unitcell_lengths=jnp.array(unitcell_lengths).astype(jnp.float32),
            unitcell_angles=jnp.array(unitcell_angles).astype(jnp.float32),
        )
        traj.save_dcd(file_name)
        traj = md.load_dcd(file_name, top=self.top_file_name)
        traj.xyz = jnp.array(traj.xyz).astype(jnp.float32) * A2NM
        traj.unitcell_lengths = jnp.array(traj.unitcell_lengths).astype(jnp.float32) * A2NM
        traj.unitcell_angles = jnp.array(traj.unitcell_angles).astype(jnp.float32)
        return Sample(traj, from_state=self.name)

    def output_parameters(self, H, params, state_dict_name):
        for g in H.getGenerators():
            if g.getName() == self.ffname:
                generator = g
                break
        else:
            raise ValueError(f"Generator with name {self.ffname} not found in Hamiltonian.")
        generator.write_to(params=params, state_dict_file=state_dict_name)

    def update_parameters(self, ckpt_name):
        self.e_eval = self.e_eval_loader(ckpt_name, self.config_path, self.cutoff)



class LmpsBaseSampleState(SampleState):

    def __init__(
        self,
        temperature,
        name,
        ckpt_path,
        top_file,
        config_path=None,
        cutoff=0.5,
        pressure=0.0,
    ):
        super().__init__(
                temperature=temperature,
                name=name,
                pressure=pressure,
                mu_dict=0.0,
                legacy=False,
                )
        self.ffname = 'BASEForce'
        self.cutoff = cutoff / A2NM # self.cutoff is in A
        self.ckpt_path = ckpt_path
        self.config_path = config_path
        self.ase_calculator = BaseCalculator(self.ckpt_path)
        self.top_file = top_file
        self.init_frame = md.load(top_file)
        # initialize ASE atoms objects, used for frame energy calculation
        self.atoms = Atoms(
            numbers=[a.element.atomic_number for a in self.init_frame.topology.atoms],
            cell=self.init_frame.unitcell_vectors.reshape(3, 3) / A2NM,
            positions=self.init_frame.xyz.reshape(-1, 3) / A2NM,
            pbc=[1, 1, 1],
            )
        self.rundir = str(Path(self.ckpt_path).parent)
        self.atoms.calc = self.ase_calculator
        self.init_lmpsdata_file = self.rundir + '/init_struct.data'
        # write out the init lammps data file
        self.update_init_lmpsdata(self.init_frame)
        self.elements = sorted(set(self.atoms.get_chemical_symbols()))
        return

    def update_atoms_geom(self, frame):
        """ Update the geometry in the self.atoms ASE object """
        # trajectory should be in nm, convert to A to be compatible with ase
        self.atoms.positions = frame.xyz / A2NM
        self.atoms.set_cell(frame.unitcell_vectors[0] / A2NM)
        return 

    def update_init_lmpsdata(self, frame):
        """ Update the lammps initial data file """
        self.update_atoms_geom(frame)
        # ase.io.write(self.init_lmpsdata_file, self.atoms, format='lammps-data')
        write_lammps_data(self.init_lmpsdata_file, atoms=self.atoms, masses=True, atom_style='atomic', units='metal')
        return

    def calc_energy_frame(self, frame):
        self.update_atoms_geom(frame)
        energy = self.atoms.get_potential_energy()
        return energy * EV2KJ

    def sample(self, nsteps, interval, skip, timestep, ttime, ptime, traj_file):
        """Run LAMMPS NPT MD via an external runner and return a Sample."""

        from pathlib import Path
        import MDAnalysis as mda

        lines = [
            "units metal",
            "dimension 3",
            "boundary p p p",
            "atom_style atomic",
            "",
            f"read_data {self.init_lmpsdata_file}",
            "",
            f"pair_style base {self.cutoff}",
            f"pair_coeff * * {self.ckpt_path} " + " ".join(self.elements),
            "",
            f"timestep {timestep * 1e-3}",  # metal uses ps
            f"velocity all create {self._temperature} 12345 loop all mom yes rot no dist gaussian",
            f"fix npt_fix all npt temp {self._temperature} {self._temperature} {ttime * 1e-3} tri {self._pressure} {self._pressure} {ptime * 1e-3}",
            "",
            "thermo 100",
            "thermo_style custom step temp pe ke etotal press vol",
            f"run {skip}",
            f"dump dcd_dump all dcd {interval} {traj_file}",
            "dump_modify dcd_dump sort id",
            f"run {nsteps - skip}",
        ]
        in_lammps = self.rundir + "/in.lammps"
        with open(in_lammps, "w") as f:
            for line in lines:
                f.write(line + "\n")

        # run md
        subprocess.run([f'lmp_bamboo_v100 -k on g 1 -sf kk -in {in_lammps}'], shell=True)

        # collect trajectory, md_traj load everything in nm
        traj = md.load_dcd(traj_file, top=self.top_file)
        return Sample(traj, from_state=self.name)

    def update_parameters(self, params, ckpt_path=None):
        """ Update model parameters """
        # params is in jax, convert to torch
        if self.ffname in params:
            params_t = j2t_pytree(params[self.ffname])
        else:
            params_t = j2t_pytree(params)
        self.ase_calculator.model.load_state_dict(params_t, strict=False)
        
        # update the ckpt file used by lammps 
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path
        torch.jit.save(self.ase_calculator.model, self.ckpt_path)
        return

    def write_state_dict(self, params, state_dict_file):
        # update the state of the internal model
        if self.ffname in params:
            params_t = j2t_pytree(params[self.ffname])
        else:
            params_t = j2t_pytree(params)
        sd = self.ase_calculator.model.state_dict()
        for p in params_t.keys():
            sd[p] = params_t[p]
        torch.save(sd, state_dict_file)
        return

class Sample:
    def __init__(self, trajectory, from_state):
        self.trajectory = trajectory
        self.from_state = from_state
        self.energy_data = {}

    def generate_energy(self, state_list):
        for state in state_list:
            if state.name not in self.energy_data:
                self.energy_data[state.name] = np.array(
                    [state.calc_energy(self.trajectory)]
                )


class LegacyReweightEstimator:
    def __init__(
        self,
        ref_energies,
        base_energies=None,
        volume=None,
        temperature=300.0,
        pressure=1.0,
    ):
        self.beta = 1.0 / temperature / 8.314 * 1000.0
        self.ref_energies = jnp.array(ref_energies)
        if base_energies is None:
            self.base_energies = jnp.zeros(ref_energies.shape)
        else:
            self.base_energies = jnp.array(base_energies)
        if volume is not None:
            self.pv = jnp.array(volume * pressure * 0.06023)
        else:
            self.pv = jnp.zeros(ref_energies.shape)

    def estimate_weight(self, uinit):
        unew = (uinit + self.base_energies + self.pv) * self.beta
        uref = (self.ref_energies + self.pv) * self.beta
        deltaU = unew - uref
        deltaU = deltaU - deltaU.max()
        weight = jnp.exp(-deltaU)
        weight = weight / weight.mean()
        return weight


class ReweightEstimator:
    """Single-sample, single-state reweighting helper.

    This implementation mirrors :class:`ReweightEstimator` in
    ``finetune/ft.py``, with two intentional differences:

    - ``base_energies`` is provided at :meth:`estimate_weight` time instead of
      construction time;
    - ``base_energies`` is treated as an energy correction in kJ/mol applied
      to the *target* state before β-scaling (i.e. we add ``beta *
      base_energies`` to the reduced target energies).
    """

    def __init__(self):
        self.sample = None
        self.state = None

    def set_sample_and_state(self, sample, state):
        """Attach the reference sample and its generating state.

        Parameters
        ----------
        sample : Sample
            A :class:`Sample` instance holding the reference trajectory and
            energy cache.
        state : SampleState
            The sampling :class:`SampleState` that generated ``sample``.
        """
        # Sanity check: the sample must originate from the provided state.
        if getattr(sample, "from_state", None) != getattr(state, "name", None):
            raise ValueError(
                "Sample.from_state must match state.name in ReweightEstimator. "
                f"Got from_state={getattr(sample, 'from_state', None)!r}, "
                f"state.name={getattr(state, 'name', None)!r}."
            )
        self.sample = sample
        self.state = state
        # Ensure reference energies are available
        self.compute_energy_matrix()

    def remove_sample_and_state(self):
        """Detach any currently stored sample/state pair."""

        self.sample = None
        self.state = None

    def compute_energy_matrix(self):
        """Populate ``sample.energy_data`` for the current reference state."""

        if self.sample is None or self.state is None:
            raise ValueError("Sample and state must be set before computing energies.")
        self.sample.generate_energy([self.state])

    def estimate_weight(self, target_state, params, base_energies=0.0, calc_uref=False):
        """Estimate reweighting factors for a new target state.

        Parameters
        ----------
        target_state : TargetState
            Target ensemble description; its :meth:`calc_energy` method is
            expected to return reduced potentials ``u_new = beta * (E + PV - μN)``.
        params : PyTree
            Parameter tree passed through to ``target_state.calc_energy``.
        base_energies : float or array-like, optional
            Energy correction(s) in kJ/mol applied to the target state before
            β-scaling. Conceptually, if ``E_new`` is the uncorrected target
            energy, we are using ``E_new + base_energies`` for reweighting.

        Returns
        -------
        weight : jax.numpy.ndarray
            Normalized reweighting factors for each frame in the sample
            trajectory.
        """

        if self.sample is None or self.state is None:
            raise ValueError("Sample and state must be set before estimating weights.")

        # Sanity check: enforce identical temperatures between sampling and
        # target states to avoid mixing ensembles.
        if not hasattr(target_state, "_temperature") or not hasattr(
            self.state, "_temperature"
        ):
            raise ValueError(
                "Both target_state and sample state must expose '_temperature' "
                "for ReweightEstimator."
            )
        if abs(target_state._temperature - self.state._temperature) > 1e-6:
            raise ValueError(
                "TargetState temperature must match SampleState temperature: "
                f"target={target_state._temperature}, sample={self.state._temperature}."
            )

        # Ensure reference energies are available
        if calc_uref:
            self.compute_energy_matrix()

        # Target reduced potential u_new (shape: n_frames,)
        unew = target_state.calc_energy(self.sample.trajectory, params)

        # Recover beta from the target state's temperature and convert the
        # base_energies correction (kJ/mol) into reduced units.
        beta = 1.0 / target_state._temperature / 8.314 * 1000.0
        base = jnp.array(base_energies)
        unew = unew + beta * base

        # Reference reduced potential u_ref from the sampling state
        uref = self.sample.energy_data[self.state.name].flatten()

        deltaU = unew - uref
        deltaU = deltaU - deltaU.max()
        weight = jnp.exp(-deltaU)
        # weight = weight / weight.mean()
        weight = weight / jnp.sum(weight)
        kappa = jnp.exp(-jnp.sum(weight * jnp.log(weight))) / len(weight)
        return weight, kappa


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

    def estimate_weight(
        self, state, parameters=None, decompose=True, return_energy=True
    ):
        if isinstance(state, TargetState):
            unew = state.calc_energy(self._full_samples, parameters)
        else:
            unew = state.calc_energy(self._full_samples)
        unew_max = unew.max()
        du_1 = self._free_energy_jax.reshape((-1, 1)) - self._umat_jax
        delta_u = du_1 + unew.reshape((1, -1)) - unew_max - du_1.min()
        cm = 1.0 / (
            jax.numpy.exp(delta_u) * jax.numpy.array(self._nk).reshape((-1, 1))
        ).sum(axis=0)
        weight = cm / cm.sum()
        if return_energy:
            return weight, unew
        return weight

    def _estimate_weight_numpy(self, unew_npy, return_cn=False):
        unew_mean = unew_npy.mean()
        du_1 = self._mbar.f_k.reshape((-1, 1)) - self._umat
        delta_u = du_1 + unew_npy.reshape((1, -1)) - unew_mean - du_1.mean()
        cn = 1.0 / (np.exp(delta_u) * self._nk.reshape((-1, 1))).sum(axis=0)
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
        Theta = (
            V
            @ Sigma
            @ np.linalg.pinv(I - Sigma @ V.T @ Ndiag @ V @ Sigma, rcond=1e-10)
            @ Sigma
            @ V.T
        )
        return Theta

    def estimate_effective_sample(self, unew, decompose=False):
        wnew, cn = self._estimate_weight_numpy(unew, return_cn=True)
        eff_samples = 1.0 / (wnew**2).sum()
        if decompose:
            state_effect = {}
            argsort = np.argsort(wnew)[::-1][: int(eff_samples)]
            for nstate in range(len(self.states)):
                istart = self._nk[:nstate].sum()
                iend = istart + self._nk[nstate]
                state_effect[self.states[nstate].name] = (
                    (argsort > istart) & (argsort < iend)
                ).sum()
            state_effect["Total"] = eff_samples
            return state_effect
        return eff_samples

    def _estimate_free_energy(self, unew):
        a = self._free_energy_jax - self._umat_jax.T
        # log(sum(n_k*exp(a)))
        a_max = a.max(axis=1, keepdims=True)
        log_denominator_n = jnp.log(
            (self._nk_jax.reshape((1, -1)) * jnp.exp(a - a_max)).sum(axis=1)
        ) + a_max.reshape((-1,))
        a2 = -unew - log_denominator_n
        # log(sum(exp(a2)))
        a2_max = a2.max()
        f_new = -jnp.log(jnp.sum(jnp.exp(a2 - a2_max))) - a2_max
        return f_new

    def estimate_free_energy_difference(
        self,
        target_state,
        ref_state,
        target_parameters=None,
        ref_parameters=None,
        decompose=True,
        return_energy=False,
    ):
        # compute F_target - F_ref
        if isinstance(ref_state, TargetState):
            u_ref = ref_state.calc_energy(self._full_samples, ref_parameters)
        else:
            u_ref = ref_state.calc_energy(self._full_samples)
        if isinstance(target_state, TargetState):
            u_target = target_state.calc_energy(self._full_samples, target_parameters)
        else:
            u_target = target_state.calc_energy(self._full_samples)
        f_ref = self._estimate_free_energy(u_ref)
        f_target = self._estimate_free_energy(u_target)
        if return_energy:
            return f_target - f_ref, u_target, u_ref
        return f_target - f_ref
