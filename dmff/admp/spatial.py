
import jax.numpy as jnp
from jax import vmap, jit
import numpy as np
from functools import partial
from ..settings import DO_JIT
from ..utils import jit_condition

# This module deals with spatial geometric operations, mainly including:
# 1. PBC related operations
# 2. Local frame rotation operations

@jit_condition(static_argnums=())
def pbc_shift(drvecs, box, box_inv):
    '''
    Dealing with the pbc shifts of vectors

    Inputs:
        rvecs:
            N * 3, a list of real space vectors in Cartesian
        box:
            3 * 3, box matrix, with axes arranged in rows
        box_inv:
            3 * 3, inverse of box matrix

    Outputs:
        rvecs:
            N * 3, vectors that have been shifted, in Cartesian
    '''
    unshifted_dsvecs = drvecs.dot(box_inv)
    dsvecs = unshifted_dsvecs - jnp.floor(unshifted_dsvecs + 0.5)
    return dsvecs.dot(box)
   
v_pbc_shift = vmap(pbc_shift, in_axes=(0, None, None), out_axes=0)
    
def normalize(matrix, axis=1, ord=2):
    '''
    Normalise a matrix along one dimension
    '''
    normalised = matrix / jnp.linalg.norm(matrix + 1e-36, axis=axis, keepdims=True, ord=ord)
    return normalised


def generate_construct_local_frames(axis_types, axis_indices):
    """
    Generates the local frame constructor, common to the same physical system

    inputs:
        axis_types:
            N, a len(N) integer array, labels the types of localframe transformation rules for each atom.
        axis_indices:
            N * 3, indices of z,x,y atoms of the localframe of each atom.

    outputs:
        construct_local_frames:
            function type (positions, box) -> local_frames
    """
    ZThenX            = 0
    Bisector          = 1
    ZBisect           = 2
    ThreeFold         = 3
    Zonly             = 4
    NoAxisType        = 5
    LastAxisTypeIndex = 6
    
    z_atoms = jnp.array(axis_indices[:, 0])
    x_atoms = jnp.array(axis_indices[:, 1])
    y_atoms = jnp.array(axis_indices[:, 2])
    
    Zonly_filter = (axis_types == Zonly)
    not_Zonly_filter = jnp.logical_not(Zonly_filter)
    Bisector_filter = (axis_types == Bisector)
    ZBisect_filter = (axis_types == ZBisect)
    ThreeFold_filter = (axis_types == ThreeFold)
    NoAxisType_filter = (axis_types == NoAxisType)
    
    def construct_local_frames(positions, box):
        '''
        This function constructs the local frames for each site

        Inputs:
            positions:
                N * 3: the positions matrix
            box:
        Outputs:
            #jichen:
            #NOTE: It doesn't seem to return Q
            Q: 
                N*(lmax+1)^2, the multipole moments in global harmonics.
            local_frames:
                N*3*3, the local frames, axes arranged in rows
        '''

        positions = jnp.array(positions)
        n_sites = positions.shape[0]
        box_inv = jnp.linalg.inv(box + jnp.eye(3) * 1e-36)

        ### Process the x, y, z vectors according to local axis rules
        vec_z = pbc_shift(positions[z_atoms] - positions, box, box_inv)
        vec_z = normalize(vec_z)
        vec_x = jnp.zeros((n_sites, 3))
        vec_y = jnp.zeros((n_sites, 3))
        # Z-Only
        x_of_vec_z = jnp.round(jnp.abs(vec_z[:,0]))
        vec_x_Zonly = jnp.array([1.-x_of_vec_z, x_of_vec_z, jnp.zeros_like(x_of_vec_z)]).T[Zonly_filter]

        vec_x = vec_x.at[Zonly_filter].set(vec_x_Zonly)
        # for those that are not Z-Only, get normalized vecX
        vec_x_not_Zonly = positions[x_atoms[not_Zonly_filter]] - positions[not_Zonly_filter]
        vec_x_not_Zonly = pbc_shift(vec_x_not_Zonly, box, box_inv)

        vec_x = vec_x.at[not_Zonly_filter].set(normalize(vec_x_not_Zonly, axis=1))
        # Bisector
        if np.sum(Bisector_filter) > 0:
            vec_z_Bisector = vec_z[Bisector_filter] + vec_x[Bisector_filter]
            vec_z = vec_z.at[Bisector_filter].set(normalize(vec_z_Bisector, axis=1))
        # z-bisector
        if np.sum(ZBisect_filter) > 0:
            vec_y_ZBisect = positions[y_atoms[ZBisect_filter]] - positions[ZBisect_filter]
            vec_y_ZBisect = pbc_shift(vec_y_ZBisect, box, box_inv)
            vec_y_ZBisect = normalize(vec_y_ZBisect, axis=1)
            vec_x_ZBisect = vec_x[ZBisect_filter] + vec_y_ZBisect
            vec_x = vec_x.at[ZBisect_filter].set(normalize(vec_x_ZBisect, axis=1))
        # ThreeFold
        if np.sum(ThreeFold_filter) > 0:
            vec_x_threeFold = vec_x[ThreeFold_filter]
            vec_z_threeFold = vec_z[ThreeFold_filter]
            
            vec_y_threeFold = positions[y_atoms[ThreeFold_filter]] - positions[ThreeFold_filter]
            vec_y_threeFold = pbc_shift(vec_y_threeFold, box, box_inv)
            vec_y_threeFold = normalize(vec_y_threeFold, axis=1)
            vec_z_threeFold += (vec_x_threeFold + vec_y_threeFold)
            vec_z_threeFold = normalize(vec_z_threeFold)
            
            vec_y = vec_y.at[ThreeFold_filter].set(vec_y_threeFold)
            vec_z = vec_z.at[ThreeFold_filter].set(vec_z_threeFold)
        
        # up to this point, z-axis should already be set up and normalized
        xz_projection = jnp.sum(vec_x*vec_z, axis = 1, keepdims=True)
        vec_x = normalize(vec_x - vec_z * xz_projection, axis=1)
        # up to this point, x-axis should be ready
        vec_y = jnp.cross(vec_z, vec_x)
        
        # NoAxisType
        if np.sum(NoAxisType_filter) > 0:
            vec_y = vec_y.at[NoAxisType_filter].set(jnp.array([0,1,0]))
            vec_z = vec_z.at[NoAxisType_filter].set(jnp.array([0,0,1]))
            vec_x = vec_x.at[NoAxisType_filter].set(jnp.array([1,0,0]))


        return jnp.stack((vec_x, vec_y, vec_z), axis=1)

    if DO_JIT:
        return jit(construct_local_frames)
    else:
        return construct_local_frames

@partial(vmap, in_axes=(0, 0, 0, 0), out_axes=0)
@jit_condition(static_argnums=())
def build_quasi_internal(r1, r2, dr, norm_dr):
    '''
    Build the quasi-internal frame between a pair of sites
    In this frame, the z-axis is pointing from r2 to r1

    Input:
        r1:
            N * 3, positions of the first vector
        r2:
            N * 3, positions of the second vector
        dr:
            N * 3, vector pointing from r1 to r2
        norm_dr:
            (N,), distances between r1 and r2

    Output:
        local_frames:
            N * 3 * 3: local frames, three axes arranged in rows
    '''
    vectorZ = dr/norm_dr
    # vectorX = jnp.where(jnp.logical_or(r1[:, 1] != r2[:, 1], r1[:, 2]!=r2[:, 2]).reshape((-1, 1)), vectorZ+jnp.array([1., 0., 0.]), vectorZ + jnp.array([0., 1., 0.]))
    vectorX = jnp.where(jnp.logical_or(r1[1]!=r2[1], r1[2]!=r2[2]), vectorZ + jnp.array([1., 0., 0.]), vectorZ + jnp.array([0., 1., 0.]))
    # dot = jnp.sum(vectorZ * vectorX, axis=1)
    dot_xz = jnp.dot(vectorZ, vectorX)
    vectorX -= vectorZ * dot_xz
    vectorX = vectorX / jnp.linalg.norm(vectorX)
    vectorY = jnp.cross(vectorZ, vectorX)
    return jnp.stack([vectorX, vectorY, vectorZ])

