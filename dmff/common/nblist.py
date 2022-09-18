from typing import Literal
import jax.numpy as jnp
from jax_md import space, partition
from dmff.admp.pairwise import distribute_v3
from dmff.utils import regularize_pairs
import numpy as np

try:
    import freud
except:
    pass


class NeighborList:
    def __init__(self, box, r_cutoff, covalent_map, dr_threshold=0, capacity_multiplier=1.25, format=Literal['dense', 'sparse', ]) -> None:
        """wrapper of jax_md.space_periodic_general and jax_md.partition.NeighborList

        Args:
            box (jnp.ndarray): A (spatial_dim, spatial_dim) affine transformation or [lx, ly, lz] vector
            rc (float): cutoff radius
        """
        self.box = box
        self.rc = self.r_cutoff = r_cutoff

        self.dr_threshold = dr_threshold
        self.capacity_multiplier = capacity_multiplier

        self.covalent_map = covalent_map
        self.displacement_fn, self.shift_fn = space.periodic_general(
            box, fractional_coordinates=False
        )
        self.neighborlist_fn = partition.neighbor_list(
            self.displacement_fn, box, r_cutoff, dr_threshold, format=partition.OrderedSparse
        )
        self.nblist = None

    def allocate(self, positions: jnp.ndarray):
        """A function to allocate a new neighbor list. This function cannot be compiled, since it uses the values of positions to infer the shapes.

        Args:
            positions (jnp.ndarray): particle positions

        Returns:
            jax_md.partition.NeighborList
        """
        if self.nblist is None:
            self.nblist = self.neighborlist_fn.allocate(positions)
        else:
            self.update(positions)
        return self.nblist

    def update(self, positions: jnp.ndarray):
        """A function to update a neighbor list given a new set of positions and a previously allocated neighbor list.

        Args:
            positions (jnp.ndarray): particle positions

        Returns:
            jax_md.partition.NeighborList
        """
        self.nblist = self.nblist.update(positions)
        return self.nblist

    @property
    def pairs(self):
        """get raw pair index

        Returns:
            jnp.ndarray: (nPairs, 2)
        """
        if self.nblist is None:
            raise RuntimeError("run nblist.allocate(positions) first")
        pairs = self.nblist.idx.T
        nbond = self.covalent_map[pairs[:, 0], pairs[:, 1]]
        return jnp.concatenate([pairs, nbond[:, None]], axis=1)

    @property
    def scaled_pairs(self):
        """get regularized pair index and mask

        Returns:
            (jnp.ndarray, jnp.ndarray): ((nParis, 2), (nPairs, ))
        """

        mask = jnp.sum(self.pairs[:, :2] == len(self.positions), axis=1)
        mask = jnp.logical_not(mask)
        pairs = regularize_pairs(self.pairs[:, :2])
        pairs = pairs[mask]
        nbond = self.covalent_map[pairs[:, 0], pairs[:, 1]]
        return jnp.concatenate([pairs, nbond[:, None]], axis=1)

    @property
    def positions(self):
        """get current positions in current neighborlist

        Returns:
            jnp.ndarray: (n, 3)
        """
        return self.nblist.reference_position

    @property
    def dr(self):
        """get pair distance vector in current neighborlist

        Returns:
            jnp.ndarray: (nPairs, 3)
        """
        pair = self.scaled_pairs
        r1 = distribute_v3(self.positions, pair[:, 0])
        r2 = distribute_v3(self.positions, pair[:, 1])
        return r1 - r2

    @property
    def distances(self):
        """get pair distance in current neighborlist

        Returns:
            jnp.ndarray: (nPairs, )

        """
        return jnp.linalg.norm(self.dr, axis=-1)

    @property
    def did_buffer_overflow(self) -> bool:
        """
        if the neighborlist buffer overflowed, return True

        Returns
        -------
        boolen
        """
        return self.nblist.did_buffer_overflow


class FreudNeighborList:

    def __init__(self, box, rc, covalent_map):

        self.box = box
        self.rc = rc
        self.covalent_map = covalent_map
        self.nblist = None

    def allocate(self, positions: np.ndarray):
        self._positions = positions
        nblist = freud.AABBQuery(self.box, positions).query(
            positions, {"r_max": self.rc, "exclude_ii": True}
        )
        self.nblist = nblist.toNeighborList()

    def update(self, positions: np.ndarray):
        self.allocate(positions)

    @property
    def pairs(self):
        if self.nblist is None:
            raise RuntimeError("run nblist.allocate(positions) first")
        pairs = np.concatenate(
            [
                self.nblist.point_indices[:, None],
                self.nblist.query_point_indices[:, None],
            ],
            axis=1,
        )
        pairs = pairs[pairs[:, 0] < pairs[:, 1]]
        nbond = self.covalent_map[pairs[:, 0], pairs[:, 1]]
        return np.concatenate([pairs, nbond[:, None]], axis=1)

    @property
    def scaled_pairs(self):
        return self.pairs

    @property
    def positions(self):
        return self._positions

    @property
    def dr(self):
        
        return self.positions[self.pairs[:, 0]] - self.positions[self.pairs[:, 1]]

    @property
    def distances(self):
        pairs = np.concatenate(
            [
                self.nblist.point_indices[:, None],
                self.nblist.query_point_indices[:, None],
            ],
            axis=1,
        )
        return self.nblist.distances[pairs[:, 0] < pairs[:, 1]]

    
