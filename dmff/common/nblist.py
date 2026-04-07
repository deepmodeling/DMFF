import numpy as np
import jax.numpy as jnp
from itertools import permutations
import warnings
try:
    import freud
    import freud.box
    import freud.locality
except ImportError:
    freud = None
    warnings.warn("WARNING: freud not installed, users need to create neighbor list by themselves.")
try:
   import dpnblist
except ImportError:
    dpnblist = None
    warnings.warn("WARNING: dpdpnblist not installed, users need to create neighbor list by themselves.")
try:
    import torch
except ImportError:
    torch = None
try:
    from torch2jax import j2t, t2j
except ImportError:
    j2t = None
    t2j = None
try:
    from NNPOps.neighbors import getNeighborPairs as nnpops_get_neighbor_pairs
except ImportError:
    nnpops_get_neighbor_pairs = None

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


class NeighborListNNPOps:
    def __init__(self, box, rcut, cov_map, padding=True, capacity_multiplier=1.3, sort=False):
        if torch is None:
            raise ImportError("torch not installed.")
        if j2t is None or t2j is None:
            raise ImportError("torch2jax not installed.")
        if nnpops_get_neighbor_pairs is None:
            raise ImportError("NNPOps not installed.")
        self.box = box
        self.rcut = rcut
        self.capacity_multiplier = None
        self.capacity_multiplier_scale = capacity_multiplier
        self.padding = padding
        self.cov_map = cov_map
        self.sort = sort

    def _search_pairs(self, coords, box=None, max_num_pairs=1, check_errors=False):
        coords_t = coords if torch.is_tensor(coords) else j2t(jnp.asarray(coords))
        if box is None:
            box_t = self.box if torch.is_tensor(self.box) else j2t(jnp.asarray(self.box))
        else:
            box_t = box if torch.is_tensor(box) else j2t(jnp.asarray(box))
        pairs, _, _, _ = nnpops_get_neighbor_pairs(
            positions=coords_t,
            cutoff=float(self.rcut),
            max_num_pairs=int(max_num_pairs),
            box_vectors=box_t,
            check_errors=check_errors,
        )
        return pairs.transpose(0, 1)

    def _canonicalize_pairs(self, pairs, natoms):
        if pairs.numel() == 0:
            return torch.empty((0, 2), dtype=torch.int32)
        valid_mask = torch.logical_and(pairs[:, 0] >= 0, pairs[:, 1] >= 0)
        pairs = pairs[valid_mask]
        if pairs.numel() == 0:
            return torch.empty((0, 2), dtype=torch.int32, device=valid_mask.device)
        pairs = torch.stack(
            [torch.minimum(pairs[:, 0], pairs[:, 1]), torch.maximum(pairs[:, 0], pairs[:, 1])],
            dim=1,
        )
        if self.sort:
            keys = pairs[:, 0].to(torch.int64) * int(natoms) + pairs[:, 1].to(torch.int64)
            order = torch.argsort(keys)
            pairs = pairs[order]
        unique_mask = torch.ones(pairs.shape[0], dtype=torch.bool, device=pairs.device)
        unique_mask[1:] = torch.any(pairs[1:] != pairs[:-1], dim=1)
        return pairs[unique_mask].to(torch.int32)

    def _build_pairs(self, real_pairs, natoms):
        if not self.padding:
            pairs_jax = t2j(real_pairs)
            nbond = self.cov_map[pairs_jax[:, 0], pairs_jax[:, 1]]
            return jnp.concatenate([pairs_jax, nbond[:, None]], axis=1)

        if self.capacity_multiplier is None:
            raise RuntimeError("Neighbor list capacity is not initialized. Call allocate first.")
        if real_pairs.shape[0] > self.capacity_multiplier:
            raise ValueError(
                f"NeighborListNNPOps capacity exceeded: found {real_pairs.shape[0]} pairs, "
                f"but allocated capacity is {self.capacity_multiplier}."
            )

        pairs_jax = t2j(real_pairs)
        if real_pairs.shape[0] > 0:
            cov_jax = self.cov_map[pairs_jax[:, 0], pairs_jax[:, 1]]
            pairs_jax = jnp.concatenate([pairs_jax, cov_jax[:, None]], axis=1)
        else:
            pairs_jax = jnp.zeros((0, 3), dtype=jnp.int32)

        padding_width = self.capacity_multiplier - real_pairs.shape[0]
        if padding_width == 0:
            return pairs_jax

        padding = jnp.zeros((padding_width, 3), dtype=jnp.int32)
        padding = padding.at[:, 0].set(natoms)
        padding = padding.at[:, 1].set(natoms)
        return jnp.concatenate([pairs_jax, padding], axis=0)

    def _compute_capacity(self, npairs):
        capacity = int(np.ceil(npairs * self.capacity_multiplier_scale))
        if npairs > 0:
            return max(capacity, npairs)
        return max(capacity, 1)

    def allocate(self, coords, box=None):
        self._positions = coords
        natoms = int(coords.shape[0])
        search_box = box if box is not None else self.box
        total_possible_pairs = max(natoms * (natoms - 1) // 2, 1)
        probe_capacity = max(1, natoms)

        while True:
            try:
                raw_pairs = self._search_pairs(
                    coords,
                    box=search_box,
                    max_num_pairs=min(probe_capacity, total_possible_pairs),
                    check_errors=True,
                )
                break
            except RuntimeError as exc:
                if probe_capacity >= total_possible_pairs:
                    raise RuntimeError("Unable to allocate NNPOps neighbor list within the full pair bound.") from exc
                probe_capacity = min(probe_capacity * 2, total_possible_pairs)

        real_pairs = self._canonicalize_pairs(raw_pairs, natoms)
        if self.capacity_multiplier is None:
            self.capacity_multiplier = self._compute_capacity(real_pairs.shape[0])
        elif self.padding:
            self.capacity_multiplier = max(int(self.capacity_multiplier), real_pairs.shape[0])

        self._pairs = self._build_pairs(real_pairs, natoms)
        return self._pairs

    def update(self, positions, box=None):
        self._positions = positions
        natoms = int(positions.shape[0])
        search_box = box if box is not None else self.box

        if self.capacity_multiplier is None:
            return self.allocate(positions, box=search_box)

        raw_pairs = self._search_pairs(
            positions,
            box=search_box,
            max_num_pairs=self.capacity_multiplier,
            check_errors=True,
        )
        real_pairs = self._canonicalize_pairs(raw_pairs, natoms)
        self._pairs = self._build_pairs(real_pairs, natoms)
        return self._pairs

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
