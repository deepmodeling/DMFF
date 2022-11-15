import numpy as np
import dmff
import jax
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
import mdtraj as md


class State:
    def __init__(self,
                 positions=None,
                 velocities=None,
                 cell=None,
                 energy=None,
                 forces=None,
                 masses=None):
        self.positions = positions
        self.velocities = velocities
        self.cell = cell
        self.energy = energy
        self.forces = forces
        self.masses = masses

    def getPositions(self):
        return self.positions

    def getVelocities(self):
        return self.velocities

    def getCellVector(self):
        return self.cell

    def getPotentialEnergy(self):
        return self.energy

    def getKineticEnergy(self):
        return (self.masses * self.velocities * self.velocities / 2.).sum()

    def getTemperature(self):
        return self.getKineticEnergy(
        ) / 3 * 2 / self.positions.shape[0] / 8.314e-3

    def getTotalEnergy(self):
        return self.getPotentialEnergy() + self.getKineticEnergy()

    def getForces(self):
        return self.forces

    def getMasses(self):
        return self.masses


def init_state_from_PDB(omm_pdb, engrad=None, temperature=300.0):
    pos = omm_pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    a, b, c = omm_pdb.topology.getPeriodicBoxVectors()
    a = a.value_in_unit(unit.nanometer)
    b = b.value_in_unit(unit.nanometer)
    c = c.value_in_unit(unit.nanometer)
    box = jnp.array([[a[0], a[1], a[2]], [b[0], b[1], b[2]],
                     [c[0], c[1], c[2]]])
    mas = jnp.array([
        a.element.mass.value_in_unit(unit.dalton)
        for a in omm_pdb.topology.atoms()
    ]).reshape((-1, 1))
    gau = jnp.array(np.random.normal(size=(mas.shape[0], 3)))
    vscale = jnp.sqrt(0.00831446261815324 * temperature / mas)
    vel = np.random.normal(size=pos.shape) * vscale
    if engrad is not None:
        ene, grd = engrad(pos, box)
        frc = -grd
    else:
        ene = 0.0
        frc = jnp.zeros(pos.shape)
    state_new = State(positions=pos,
                      velocities=vel,
                      cell=box,
                      energy=ene,
                      forces=frc,
                      masses=mas)
    return state_new


class XYZWriter:
    def __init__(self, filename, topology):
        self.topo = topology
        self.file = open(filename, "w")

    def write(self, state):
        # generate text
        xx, yy, zz = state.getCellVector() * 10.
        text = f"{xx[0]} {xx[1]} {xx[2]} {yy[0]} {yy[1]} {yy[2]} {zz[0]} {zz[1]} {zz[2]}\n"
        # write no. of atoms
        pos = state.getPositions() * 10.  # nm 2 angstrom
        self.file.write(f"{pos.shape[0]}\n")
        self.file.write(text.strip())
        self.file.write("\n")
        for na, a in enumerate(self.topo.atoms()):
            self.file.write(
                f"{a.element.symbol:4s}{pos[na,0]:16.8f}{pos[na,1]:16.8f}{pos[na,2]:16.8f}\n"
            )
        self.file.flush()

    def close(self):
        self.file.close()


def xyz_to_traj(filename, topology):
    xyz, ulength, uangle = [], [], []
    with open(filename, "r") as f:
        text = f.readlines()
    p = 0
    while p < len(text):
        natom = int(text[p].strip())
        cell = [float(i) * 0.1 for i in text[p + 1].strip().split()]
        frame = text[p + 2:p + 2 + natom]
        p = p + natom + 2
        frame = [i.strip().split()[1:] for i in frame]
        frame = [[float(j) * 0.1 for j in i] for i in frame]
        if len(frame) == natom:
            xyz.append(frame)
            ulength.append([cell[0], cell[4], cell[8]])
            uangle.append([90., 90., 90.])

    xyz = np.array(xyz)
    ulength = np.array(ulength)
    uangle = np.array(uangle)
    return md.Trajectory(xyz,
                         topology,
                         unitcell_lengths=ulength,
                         unitcell_angles=uangle)


class BaseIntegrator:
    def __init__(self, timestep=1.0e-3):
        self.dt = timestep

    def step(self, state, engrad=None):
        pos = state.getPositions()
        vel = state.getVelocities()
        box = state.getCellVector()
        mas = state.getMasses()
        frc = state.getForces()
        ener = state.getPotentialEnergy()

        pos_1, vel_1, box_1, mas_1, ene_1, frc_1 = self.update_state(
            pos, vel, box, mas, ener, frc, engrad=engrad)

        state_new = State(positions=pos_1,
                          velocities=vel_1,
                          cell=box_1,
                          energy=ene_1,
                          forces=frc_1,
                          masses=mas)
        return state_new

    def update_state(self,
                     pos_0,
                     vel_0,
                     box_0,
                     mas,
                     ene_0,
                     frc_0,
                     engrad=None):
        ene_1, frc_1 = engrad(pos_0, box_0)
        return pos, vel, box, mas, ene_1, -frc_1


def createAnalyticalPotential(a=1.0):
    pdb = app.PDBFile("data/analytical_pot.pdb")

    def potential(positions, box):
        energy = jnp.power(
            positions[0, 0],
            4) + jnp.exp(-a * jnp.power(positions[0, 0], 2)) * (jnp.power(
                positions[0, 1], 2) + jnp.power(positions[0, 2], 2)) / 2
        return energy

    return pdb, jax.value_and_grad(potential, 0)


def create1DHarmonicOscillator():
    pdb = app.PDBFile("data/analytical_pot.pdb")

    def potential(positions, box):
        energy = 0.5 * 100. * positions[0, 0] * positions[0, 0]
        return energy

    return pdb, jax.value_and_grad(potential, 0)


def create2DHarmonicOscillator():
    pdb = app.PDBFile("data/analytical_pot.pdb")

    def potential(positions, box):
        energy = 0.5 * 100. * jnp.power(positions[0, :2], 2).sum()
        return energy

    return pdb, jax.value_and_grad(potential, 0)


def create3DHarmonicOscillator():
    pdb = app.PDBFile("data/analytical_pot.pdb")

    def potential(positions, box):
        energy = 0.5 * 100. * jnp.power(positions, 2).sum()
        return energy

    return pdb, jax.value_and_grad(potential, 0)


def createMultiHarmonicOscillator():
    pdb = app.PDBFile("data/multi_harmonic_osc.pdb")
    h = dmff.Hamiltonian("data/multi_harmonic_osc.xml")
    pot = h.createPotential(pdb.topology).getPotentialFunc()
    pairs = jnp.array([])

    def potential(positions, box):
        return pot(positions, box, pairs, h.paramtree)

    return pdb, jax.value_and_grad(potential, 0)


def createLJFluid():
    pdb = app.PDBFile("data/lj_fluid.pdb")
    h = dmff.Hamiltonian("data/lj_fluid.xml")
    pot = h.createPotential(pdb.topology,
                            nonbondedMethod=app.CutoffPeriodic,
                            nonbondedCutoff=1.0 *
                            unit.nanometer).getPotentialFunc()
    gen = h.getGenerators()[-1]
    a, b, c = pdb.topology.getPeriodicBoxVectors()
    a = a.value_in_unit(unit.nanometer)
    b = b.value_in_unit(unit.nanometer)
    c = c.value_in_unit(unit.nanometer)
    box = jnp.array([[a[0], a[1], a[2]], [b[0], b[1], b[2]],
                     [c[0], c[1], c[2]]])

    engrad = jax.value_and_grad(pot, 0)

    def potential_engrad(positions, box):
        nblist = dmff.NeighborListFreud(box, 1.0, gen.covalent_map)
        nblist.allocate(positions)
        pairs = nblist.pairs
        return engrad(positions, box, pairs, h.paramtree)

    return pdb, potential_engrad


def createAlaDipeptide():
    pdb = app.PDBFile("data/ala2_fixed.pdb")
    h = dmff.Hamiltonian("amber99sbildn.xml")
    pot = h.createPotential(pdb.topology,
                            nonbondedMethod=app.NoCutoff).getPotentialFunc()
    box = jnp.array([
        [99., 0.0, 0.0],
        [0.0, 99., 0.0],
        [0.0, 0.0, 99.],
    ])
    nbgen = [g for g in h.getGenerators()][-1]
    nblist = dmff.NeighborListFreud(box, 49.0, nbgen.covalent_map)
    pos = jnp.array(
        pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    nblist.allocate(pos)
    pairs = nblist.pairs
    engrad = jax.value_and_grad(pot, 0)

    def potential_engrad(positions, box):
        return engrad(positions, box, pairs, h.paramtree)

    return pdb, potential_engrad