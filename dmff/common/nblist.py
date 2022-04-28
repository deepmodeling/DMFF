from jax_md import space, partition
import jax.numpy as jnp
from dmff.utils import regularize_pairs
import jax.numpy as jnp

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
        
    def allocate(self, positions: jnp.ndarray):
        """ A function to allocate a new neighbor list. This function cannot be compiled, since it uses the values of positions to infer the shapes.

        Args:
            positions (jnp.ndarray): particle positions

        Returns:
            jax_md.partition.NeighborList
        """
        self.nblist = self.neighborlist_fn.allocate(positions)
        return self.nblist
    
    def update(self, positions: jnp.ndarray):
        """ A function to update a neighbor list given a new set of positions and a previously allocated neighbor list.

        Args:
            positions (jnp.ndarray): particle positions

        Returns:
            jax_md.partition.NeighborList
        """
        self.nblist = self.neighborlist_fn.update(positions)
        
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
    def distance(self):
        """ get pair distance in current neighborlist

        Returns:
            jnp.ndarray: (nPairs, )
        """
        pair, _ = self.pair_mask
        return self.positions[pair[:, 0]] - self.positions[pair[:, 1]]
        
        
if __name__ == '__main__':
    positions = jnp.array([[12.434,   3.404,   1.540],
                            [13.030,   2.664,   1.322],
                            [12.312,   3.814,   0.660],
                            [14.216,   1.424,   1.103],
                            [14.246,   1.144,   2.054],
                            [15.155,   1.542,   0.910]])
    
    nbobj = NeighborList(jnp.array([31.289,   31.289,   31.289]), 4.0 )
    nbobj.allocate(positions)
    pairs = nbobj.pairs
    pair, mask = nbobj.pair_mask
    distance = nbobj.distance
    print(pairs*mask)