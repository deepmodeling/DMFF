import numpy as np
import jax.numpy as jnp
from itertools import permutations
try:
    import freud
    import freud.box
    import freud.locality
except ImportError:
    freud = None
    import warnings
    warnings.warn("WARNING: freud not installed, users need to create neighbor list by themselves.")
try:
   import dpnblist
except ImportError:
    dpnblist = None
    import warnings
    warnings.warn("WARNING: dpdpnblist not installed, users need to create neighbor list by themselves.")

class NeighborListDp:
    def __init__(self, alg_type, box, rcut, cov_map, padding=True):
        if dpnblist is None:
            raise ImportError("dpnblist not installed.")
        self.box = dpnblist.Box([box[0][0], box[1][1], box[2][2]], [90.0, 90.0, 90.0])
        self.nb = dpnblist.NeighborList(alg_type)
        self.flag = False
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
        dbox = dpnblist.Box([box[0][0], box[1][1], box[2][2]], [90.0, 90.0, 90.0]) if box is not None else self.box
        self.nb.build(dbox, coords, self.rcut)
        pair = self.nb.get_neighbor_pair()
        nlist = np.vstack((pair[:, 0], pair[:, 1])).T
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


class NeighborListFreud:
    def __init__(self, box, rcut, cov_map, padding=True):
        if freud is None:
            raise ImportError("Freud not installed.")
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

class NeighborList(NeighborListFreud):
    ...


class NoCutoffNeighborList:
    
    def __init__(self, cov_map, padding=True):
        self.capacity_multiplier = None
        self.padding = padding
        self.cov_map = cov_map
    
    def _do_cov_map(self, pairs):
        nbond = self.cov_map[pairs[:, 0], pairs[:, 1]]
        pairs = jnp.concatenate([pairs, nbond[:, None]], axis=1)
        return pairs

    def allocate(self, coords, box=None):
        self._positions = coords  # cache it
        natoms = coords.shape[0]
        dpnblist = np.fromiter(permutations(range(natoms), 2), dtype=np.dtype(int, 2))
        nlist = dpnblist[dpnblist[:, 0] < dpnblist[:, 1]]
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
        self.allocate(positions)

    @property
    def pairs(self):
        return self._pairs

    @property
    def scaled_pairs(self):
        return self._pairs

    @property
    def positions(self):
        return self._positions


class NoPeriodicNeighborList(NoCutoffNeighborList):
    
    def __init__(self, rcut, cov_map, padding=True):
        super().__init__(cov_map, padding)
        self.rcut = rcut

    def allocate(self, coords):
        self._positions = coords  # cache it
        natoms = coords.shape[0]
        dpnblist = np.fromiter(permutations(range(natoms), 2), dtype=np.dtype(int, 2))
        nlist = dpnblist[dpnblist[:, 0] < dpnblist[:, 1]]
        distances = np.linalg.norm(coords[nlist[:, 0]] - coords[nlist[:, 1]], axis=1)
        nlist = nlist[distances < self.rcut]
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
