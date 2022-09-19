import jax.numpy as jnp
from jax_md import space, partition
from dmff.utils import jit_condition
from dmff.utils import regularize_pairs
from jax import jit


class NeighborList:
    
    def __init__(self, box, rc) -> None:
        """ wrapper of jax_md.space_periodic_general and jax_md.partition.NeighborList

        Args:
            box (jnp.ndarray): A (spatial_dim, spatial_dim) affine transformation or [lx, ly, lz] vector
            rc (float): cutoff radius
        """
        self.box = box
        self.rc = rc
        self.displacement_fn, self.shift_fn = space.periodic_general(box, fractional_coordinates=False)
        self.neighborlist_fn = partition.neighbor_list(self.displacement_fn, box, rc, 0, format=partition.OrderedSparse)
        self.nblist = None
        
    def allocate(self, positions: jnp.ndarray):
        """ A function to allocate a new neighbor list. This function cannot be compiled, since it uses the values of positions to infer the shapes.

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
        """ A function to update a neighbor list given a new set of positions and a previously allocated neighbor list.

        Args:
            positions (jnp.ndarray): particle positions

        Returns:
            jax_md.partition.NeighborList
        """
        # jit_deco = jit_condition()
        # jit_deco(self.nblist.update)(positions)
        self.nblist = self.nblist.update(positions)
        # if self.nblist.did_buffer_overflow:
        #     self.nblist = self.neighborlist_fn.allocate(positions)
        return self.nblist
    
    @property
    def pairs(self):
        """ get raw pair index

        Returns:
            jnp.ndarray: (nPairs, 2)
        """
        return self.nblist.idx.T
    
    @property
    def pair_mask(self):
        """ get regularized pair index and mask

        Returns:
            (jnp.ndarray, jnp.ndarray): ((nParis, 2), (nPairs, ))
        """

        mask = jnp.sum(self.pairs == len(self.positions), axis=1)
        mask = jnp.logical_not(mask)
        pair = regularize_pairs(self.pairs)
        
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
