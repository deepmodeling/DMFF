from typing import Optional
import numpy as np
import jax.numpy as jnp
from jax_md import space, partition
from dmff.utils import regularize_pairs
try:
    import freud
except ImportError:
    pass


class NeighborList:
    
    def __init__(self, box, rc, covalent_map) -> None:
        """ wrapper of jax_md.space_periodic_general and jax_md.partition.NeighborList

        Args:
            box (jnp.ndarray): A (spatial_dim, spatial_dim) affine transformation or [lx, ly, lz] vector
            rc (float): cutoff radius
        """
        self.box = box
        self.rc = rc
        self.covalent_map = covalent_map
        self.displacement_fn, self.shift_fn = space.periodic_general(box, fractional_coordinates=False)
        self.neighborlist_fn = partition.neighbor_list(self.displacement_fn, box, rc, 0, format=partition.OrderedSparse)
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
        """ get raw pair index

        Returns:
            jnp.ndarray: (nPairs, 2)
        """
        if self.nblist is None:
            raise RuntimeError('run nblist.allocate(positions) first')
        pairs = self.nblist.idx.T
        nbond = self.covalent_map[pairs[:, 0], pairs[:, 1]]
        return jnp.concatenate([pairs, nbond[:, None]], axis=1)
    
    @property
    def pair_mask(self):
        """ get regularized pair index and mask

        Returns:
            (jnp.ndarray, jnp.ndarray): ((nParis, 2), (nPairs, ))
        """

        mask = jnp.sum(self.pairs[:, :2] == len(self.positions), axis=1)
        mask = jnp.logical_not(mask)
        pair = regularize_pairs(self.pairs[:, :2])
        
        return pair, mask
    
    @property
    def positions(self):
        """ get current positions in current neighborlist

        Returns:
            jnp.ndarray: (n, 3)
        """
        return self.nblist.reference_position
    
    @property
    def dr(self):
        """ get pair distance vector in current neighborlist

        Returns:
            jnp.ndarray: (nPairs, 3)
        """
        pair, _ = self.pair_mask
        return self.positions[pair[:, 0]] - self.positions[pair[:, 1]]
        
    @property
    def distance(self):
        """ get pair distance in current neighborlist
        
        Returns:
            jnp.ndarray: (nPairs, )
        
        """
        return jnp.linalg.norm(self.dr, axis=1)

    @property
    def did_buffer_overflow(self)->bool:
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
        self.nmax = None
        self.padding = padding
        self.cov_map = cov_map
    
    def _do_cov_map(self, pairs):
        nbond = self.cov_map[pairs[:, 0], pairs[:, 1]]
        pairs = jnp.concatenate([pairs, nbond[:, None]], axis=1)
        return pairs

    def allocate(self, coords, box=None):
        fbox = freud.box.Box.from_matrix(box) if box is not None else self.fbox
        aq = freud.locality.AABBQuery(fbox, coords)
        res = aq.query(coords, dict(r_max=self.rcut, exclude_ii=True))
        nlist = res.toNeighborList()
        nlist = np.vstack((nlist[:, 0], nlist[:, 1])).T
        nlist = nlist.astype(np.int32)
        msk = (nlist[:, 0] - nlist[:, 1]) < 0
        nlist = nlist[msk]
        if self.nmax is None:
            self.nmax = int(nlist.shape[0] * 1.3)
        
        if not self.padding:
            return self._do_cov_map(nlist)

        self.nmax = max(self.nmax, nlist.shape[0])
        padding_width = self.nmax - nlist.shape[0]
        if padding_width == 0:
            return self._do_cov_map(nlist)
        elif padding_width > 0:
            padding = np.ones((self.nmax - nlist.shape[0], 2), dtype=np.int32) * coords.shape[0]
            nlist = np.vstack((nlist, padding))
            return self._do_cov_map(nlist)
        else:
            raise ValueError("padding width < 0")