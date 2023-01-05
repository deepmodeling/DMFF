from typing import Tuple, List, Dict, Callable, Optional
import numpy as np
import jax.numpy as jnp


class VirtualSite:
    """
    Class for manipulation of virtual sites
    """
    def __init__(self, matches_dict: Dict[Tuple[int], int]):
        """
        Initialize a virtual site object
        """
        self.num_vsites = len(matches_dict)
        self.matches, self.indices = [], []
        for key, value in matches_dict.items():
            self.matches.append(key)
            self.indices.append(value)

    def getAddVirtualSiteFunc(self) -> Callable:
        """
        Get fuction to compute virtual site coordinates
        """

        def add_vsite_position(pos: jnp.ndarray, vtypes: jnp.ndarray, vdist: jnp.ndarray) -> jnp.ndarray:
            newpos = jnp.zeros((pos.shape[0] + self.num_vsites, pos.shape[1]))
            newpos = newpos.at[:pos.shape[0]].set(pos)
            for i in range(self.num_vsites):
                match = self.matches[i]
                idx = self.indices[i]
                if vtypes[idx] == 1:
                    vec = newpos[match[0]] - newpos[match[1]]
                    nvec = vec / jnp.linalg.norm(vec, ord=2)
                    newpos = newpos.at[pos.shape[0] + i].set(newpos[match[0]] + vdist[idx] * nvec)
                elif vtypes[idx] == 2:
                    vec1 = newpos[match[0]] - newpos[match[1]]
                    vec2 = newpos[match[0]] - newpos[match[2]]
                    nvec1 = vec1 / jnp.linalg.norm(vec1, ord=2)
                    nvec2 = vec2 / jnp.linalg.norm(vec2, ord=2)
                    nvec = (nvec1 + nvec2) / jnp.linalg.norm(nvec1 + nvec2, ord=2)
                    newpos = newpos.at[pos.shape[0] + i].set(newpos[match[0]] + vdist[idx] * nvec)
            return newpos
        
        return add_vsite_position
    
    def addVirtualSiteToMol(self, rdmol, vtypes=None, vdist=None):
        """
        Add virtual site to rdkit.Chem.Mol object

        Parameters
        ----------
        rdmol: rdkit.Chem.Mol
            Mol object to which virtual sites are added
        vtypes: jnp.ndarray or None
            Virtual site types, can be obtained from `dmff.Hamiltonian.paramtree['vsite_types']`
        vdist: jnp.ndarray or None
            Virtual site distances params, can be obtained from `dmff.Hamiltonian.paramtree['vsite_distances']`
        
        Return
        ------
        newmol: rdkit.Chem.Mol
            Mol with virtual sites added 
        """
        if isinstance(vtypes, jnp.ndarray) and isinstance(vdist, jnp.ndarray):
            func = self.getAddVirtualSiteFunc()
            # convert between angstrom and nm
            pos = jnp.array(rdmol.GetConformer(0).GetPositions()) / 10
            addCoords = func(pos, vtypes, vdist) * 10
        else:
            addCoords = None

        newmol = self.add_dummy(rdmol, [m[0] for m in self.matches], addCoords)
        return newmol
    
    @staticmethod
    def add_dummy(mol, parentAtomIdx: List[int], addCoords: Optional[np.ndarray]):
        """
        Add dummy atom to rdkit.Chem.Mol object and make a dummy bond between
        the dummy atom and its parent atom

        Parameters
        ----------
        mol: rdkit.Chem.Mol
            Molecule to add dummy atom
        parentAtomIdx: int
            Index of the dummy atom's parent atom
        addCoords: numpy.ndarray or None
            Coordinates of the virtual sites. In unit of Angstrom
        """
        from rdkit import Chem
        ori_num_atoms = mol.GetNumAtoms()
        rwmol = Chem.RWMol(mol)
        
        duIdxs = []
        for pidx in parentAtomIdx:
            dummy = Chem.Atom(0)
            duIdx = rwmol.AddAtom(dummy)
            rwmol.AddBond(duIdx, pidx)
            duIdxs.append(duIdx)
        
        newmol = rwmol.GetMol()
        if addCoords is not None:
            assert len(addCoords) == len(parentAtomIdx) + ori_num_atoms, f"Number of atoms in coordinates doesn't match"
            conf = newmol.GetConformer()
            for i, duIdx in enumerate(duIdxs):
                conf.SetAtomPosition(duIdx, [float(x) for x in addCoords[duIdx]])
            
        return newmol

