import sys
import jax.numpy as jnp
from jax import vmap
from dmff.utils import jit_condition
from functools import partial

# This module deals with the transformations and rotations of multipoles


# The important conversion matrices used in multipoles
rt3 = 1.73205080757
inv_rt3 = 1.0/rt3
# the dipole conversion matrices, cart2harm and harm2cart
C1_h2c = jnp.array([[0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0]])
C1_c2h = C1_h2c.T
# the quadrupole conversion matrices
C2_c2h = jnp.array([[      0,        0,     1,         0,         0,         0],
                    [      0,        0,     0,         0, 2*inv_rt3,         0],
                    [      0,        0,     0,         0,         0, 2*inv_rt3],
                    [inv_rt3, -inv_rt3,     0,         0,         0,         0],
                    [      0,        0,     0, 2*inv_rt3,         0,         0]])

C2_h2c = jnp.array([[-0.5,     0,     0,  rt3/2,     0],
                    [-0.5,     0,     0, -rt3/2,     0],
                    [   1,     0,     0,      0,     0],
                    [   0,     0,     0,      0, rt3/2],
                    [   0, rt3/2,     0,      0,     0],
                    [   0,     0, rt3/2,      0,     0]])


@partial(vmap, in_axes=(0, None), out_axes=0)
@jit_condition(static_argnums=(1))
def convert_cart2harm(Theta, lmax):
    '''
    Convert the multipole moments in cartesian repr to spherical harmonic repr
    Essentially, implementing the equations in Appendix E in Anthony's book

    Inputs:
        Theta:
            n * N_cart: stores the cartesian multipoles of each site
        lmax:
            integer, the maximum L, currently only supports up to quadrupole

    Outputs:
        Q:
            n * (l+1)^2, stores the spherical multipoles
    '''
    if lmax > 2:
        sys.exit('l > 2 (beyond quadrupole) not supported')

    # n_sites = Theta.shape[0]
    n_harm = (lmax + 1)**2
    # Q = jnp.zeros((n_sites, n_harm))
    Q_mono = Theta[0:1]
    
    # dipole
    if lmax >= 1:
        dipoles_cart = Theta[1:4].T
        Q_dip = C1_c2h.dot(dipoles_cart).T
    # quadrupole
    if lmax >= 2:
        quad_cart = Theta[4:10].T
        Q_quad = C2_c2h.dot(quad_cart).T

    if lmax == 0:
        Q = Q_mono
    elif lmax == 1:
        Q = jnp.hstack([Q_mono, Q_dip])
    else:
        Q = jnp.hstack([Q_mono, Q_dip, Q_quad])

    return Q


@partial(vmap, in_axes=(0, 0), out_axes=0)
@jit_condition(static_argnums=())
def rot_ind_global2local(U_g, localframes):
    '''
    A special rotation function for just dipoles, aim for applying on induced dipoles
    '''
    zxy = jnp.array([2,0,1])
    R1 = localframes[zxy][:,zxy]
    U_l = jnp.dot(R1, U_g)
    return U_l


@partial(vmap, in_axes=(0, 0, None), out_axes=0)
@jit_condition(static_argnums=(2))
def rot_global2local(Q_gh, localframes, lmax=2):
    '''
    This function rotates harmonic moments Q from global frame to local frame

    Input:
        Q_gh: 
            n * (l+1)^2, stores the global harmonic multipole moments of each site
        localframes: 
            n * 3 * 3, stores the Rotation matrix for each site, the R is defined as:
            [r1, r2, r3]^T, with r1, r2, r3 being the local frame axes
        lmax:
            integer, the maximum multipole order

    Output:
        Q_lh:
            n * (l+1)^2, stores the local harmonic multipole moments
    '''
    if lmax > 2:
        raise NotImplementedError('l > 2 (beyond quadrupole) not supported')

    # monopole
    Q_lh_0 = Q_gh[0:1]
    # for dipole
    if lmax >= 1:
        zxy = jnp.array([2,0,1])
        # the rotation matrix
        R1 = localframes[zxy][:,zxy]
        # rotate
        Q_lh_1 = jnp.dot(R1, Q_gh[1:4])
    if lmax >= 2:
        xx = localframes[0, 0]
        xy = localframes[0, 1]
        xz = localframes[0, 2]
        yx = localframes[1, 0]
        yy = localframes[1, 1]
        yz = localframes[1, 2]
        zx = localframes[2, 0]
        zy = localframes[2, 1]
        zz = localframes[2, 2]
        quadrupoles = Q_gh[4:9]
        # construct the local->global transformation matrix
        # this is copied directly from the convert_mom_to_xml.py code
        C2_gl_00 = (3*zz**2-1)/2
        C2_gl_01 = rt3*zx*zz
        C2_gl_02 = rt3*zy*zz
        C2_gl_03 = (rt3*(-2*zy**2-zz**2+1))/2
        C2_gl_04 = rt3*zx*zy
        C2_gl_10 = rt3*xz*zz
        C2_gl_11 = 2*xx*zz-yy
        C2_gl_12 = yx+2*xy*zz
        C2_gl_13 = -2*xy*zy-xz*zz
        C2_gl_14 = xx*zy+zx*xy
        C2_gl_20 = rt3*yz*zz
        C2_gl_21 = 2*yx*zz+xy
        C2_gl_22 = -xx+2*yy*zz
        C2_gl_23 = -2*yy*zy-yz*zz
        C2_gl_24 = yx*zy+zx*yy
        C2_gl_30 = rt3*(-2*yz**2-zz**2+1)/2
        C2_gl_31 = -2*yx*yz-zx*zz
        C2_gl_32 = -2*yy*yz-zy*zz
        C2_gl_33 = (4*yy**2+2*zy**2+2*yz**2+zz**2-3)/2
        C2_gl_34 = -2*yx*yy-zx*zy
        C2_gl_40 = rt3*xz*yz
        C2_gl_41 = xx*yz+yx*xz
        C2_gl_42 = xy*yz+yy*xz
        C2_gl_43 = -2*xy*yy-xz*yz
        C2_gl_44 = xx*yy+yx*xy
        # rotate
        C2_gl = jnp.array(
            [
                [C2_gl_00, C2_gl_10, C2_gl_20, C2_gl_30, C2_gl_40],
                [C2_gl_01, C2_gl_11, C2_gl_21, C2_gl_31, C2_gl_41],
                [C2_gl_02, C2_gl_12, C2_gl_22, C2_gl_32, C2_gl_42],
                [C2_gl_03, C2_gl_13, C2_gl_23, C2_gl_33, C2_gl_43],
                [C2_gl_04, C2_gl_14, C2_gl_24, C2_gl_34, C2_gl_44]
            ]
        ).swapaxes(0,1)
        Q_lh_2 = jnp.einsum('jk,k->j', C2_gl, quadrupoles)
    if lmax == 0:
        Q_lh = Q_lh_0
    elif lmax == 1:
        Q_lh = jnp.hstack([Q_lh_0, Q_lh_1])
    elif lmax == 2:
        Q_lh = jnp.hstack([Q_lh_0, Q_lh_1, Q_lh_2])

    return Q_lh


# @jit_condition(static_argnums=(2))
def rot_local2global(Q_lh, localframes, lmax=2):
    '''
    This function rotates harmonic moments Q from global frame to local frame
    Simply use the rot_global2local, and localframe^-1

    Input:
        Q_lh: 
            n * (l+1)^2, stores the local harmonic multipole moments of each site
        localframes: 
            n * 3 * 3, stores the Rotation matrix for each site, the R is defined as:
            [r1, r2, r3]^T, with r1, r2, r3 being the local frame axes
        lmax:
            integer, the maximum multipole order

    Output:
        Q_gh:
            n * (l+1)^2, stores the rotated global harmonic multipole moments
    '''
    return rot_global2local(Q_lh, jnp.swapaxes(localframes, -2, -1), lmax)

