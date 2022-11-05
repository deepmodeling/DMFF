from typing import Optional, Literal

import numpy as np
import jax.numpy as jnp
from jax_md import space, partition
import freud

from dmff.utils import regularize_pairs


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
        
    def allocate(self, positions: jnp.ndarray, box: Optional[jnp.ndarray] = None):
        """ A function to allocate a new neighbor list. This function cannot be compiled, since it uses the values of positions to infer the shapes.

        Args:
            positions (jnp.ndarray): particle positions

        Returns:
            jax_md.partition.NeighborList
        """
        if self.nblist is None:
            self.nblist = self.neighborlist_fn.allocate(positions)
        else:
            self.update(positions, box)
        return self.nblist
    
    def update(self, positions: jnp.ndarray, box: Optional[jnp.ndarray] = None):
        """ A function to update a neighbor list given a new set of positions and a previously allocated neighbor list.

        Args:
            positions (jnp.ndarray): particle positions

        Returns:
            jax_md.partition.NeighborList
        """
        if box is None:
            self.nblist = self.nblist.update(positions)
        else:
            self.nblist = self.nblist.update(positions, box)
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
    def did_buffer_overflow(self) -> bool:
        """
        if the neighborlist buffer overflowed, return True

        Returns
        -------
        boolen
        """
        return self.nblist.did_buffer_overflow


class NeighborListFreud:
    def __init__(self, box, rcut, cov_map, padding=True):
        self.fbox = freud.box.Box.from_matrix(box)
        self.rcut = rcut
        self.capacity_multiplier = None
        self.padding = padding
        self.cov_map = cov_map
    
    def _do_cov_map(self, pairs):
        nbond = self.cov_map[pairs[:, 0], pairs[:, 1]]
        pairs = jnp.concatenate([pairs, nbond[:, None]], axis=1)
        return pairs

    def allocate(self, coords, box=None):
        self._positions = coords  # cache it
        fbox = freud.box.Box.from_matrix(box) if box is not None else self.fbox
        aq = freud.locality.AABBQuery(fbox, coords)
        res = aq.query(coords, dict(r_max=self.rcut, exclude_ii=True))
        nlist = res.toNeighborList()
        nlist = np.vstack((nlist[:, 0], nlist[:, 1])).T
        nlist = nlist.astype(np.int32)
        msk = (nlist[:, 0] - nlist[:, 1]) < 0
        nlist = nlist[msk]
        if self.capacity_multiplier is None:
            self.capacity_multiplier = int(nlist.shape[0] * 1.3)
        
        if not self.padding:
            self._pairs = self._do_cov_map(nlist)
            return self._pairs

        self.capacity_multiplier = max(self.capacity_multiplier, nlist.shape[0])
        padding_width = self.capacity_multiplier - nlist.shape[0]
        if padding_width == 0:
            self._pairs = self._do_cov_map(nlist)
            return self._pairs
        elif padding_width > 0:
            padding = np.ones((self.capacity_multiplier - nlist.shape[0], 2), dtype=np.int32) * coords.shape[0]
            nlist = np.vstack((nlist, padding))
            self._pairs = self._do_cov_map(nlist)
            return self._pairs
        else:
            raise ValueError("padding width < 0")

    def update(self, positions, box=None):
        self.allocate(positions, box)

    @property
    def pairs(self):
        return self._pairs

    @property
    def scaled_pairs(self):
        return self._pairs

    @property
    def positions(self):
        return self._positions
