from functools import partial

import jax.numpy as jnp
from ..utils import jit_condition
from jax import vmap
import jax

# This module deals with the transformations and rotations of multipoles


# The important conversion matrices used in multipoles
rt3 = 1.73205080757
inv_rt3 = 1.0/rt3
rt2 = 1.41421356237
inv_rt2 = 1.0/rt2
rt5 = 2.2360679775
inv_rt5 = 1.0/rt5
rt10 = 3.16227766017
inv_rt10 = 1.0/rt10
rt8 = 2.82842712475
rt12 = 3.46410161514
rt24 = 4.89897948557
inv_rt24 = 1.0/rt24


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
# the octupole conversion matrices
C3_c2h = jnp.array([[    0,      0,      0,      0,    0,      0,      0,      0,     0,     1],
                    [    0,      0,      0,      0,    0,      0,      0,   rt3/rt2,  0,     0],
                    [    0,      0,      0,      0,    0,      0,      0,      0,  rt3/rt2,  0],
                    [    0,      0,      0,      0, inv_rt5,   0,   -inv_rt5,  0,     0,     0],
                    [    0,      0,      0,      0,    0,  2*rt3/rt5,  0,      0,     0,     0],
                    [ inv_rt10,  0, -3*inv_rt10, 0,    0,      0,      0,      0,     0,     0],
                    [    0,  3*inv_rt10, 0, -inv_rt10, 0,      0,      0,      0,     0,     0]])


C3_h2c = jnp.array([[    0,  -rt3/rt8,   0,      0,      0,   rt5/rt8,      0],
                    [    0,      0,  -inv_rt24,  0,      0,      0,     rt5/rt8],
                    [    0,  -inv_rt24,  0,      0,      0,  -rt5/rt8,      0],
                    [    0,      0,  -rt3/rt8,   0,      0,      0,    -rt5/rt8],
                    [  -0.5,     0,      0,   rt5/rt12,  0,      0,         0],
                    [    0,      0,      0,      0,   rt5/rt12,  0,         0],
                    [  -0.5,     0,      0,  -rt5/rt12,  0,      0,         0],
                    [    0,   rt2/rt3,   0,      0,      0,      0,         0],
                    [    0,      0,   rt2/rt3,   0,      0,      0,         0],
                    [    1,      0,      0,      0,      0,      0,         0]])





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
    if lmax > 3:
        raise ValueError('l > 3 (beyond octupole) not supported')

    Q_mono = Theta[0:1]
    
    # dipole
    if lmax >= 1:
        dipoles_cart = Theta[1:4].T
        Q_dip = C1_c2h.dot(dipoles_cart).T
    # quadrupole
    if lmax >= 2:
        quad_cart = Theta[4:10].T
        Q_quad = C2_c2h.dot(quad_cart).T
    # octupole
    if lmax >= 3:
        octu_cart = Theta[10:20].T
        Q_octu = C3_c2h.dot(octu_cart).T
    
    if lmax == 0:
        Q = Q_mono
    elif lmax == 1:
        Q = jnp.hstack([Q_mono, Q_dip])
    elif lmax == 2:
        Q = jnp.hstack([Q_mono, Q_dip, Q_quad])
    else:
        Q = jnp.hstack([Q_mono, Q_dip, Q_quad, Q_octu])

    return Q


@partial(vmap, in_axes=(0, None), out_axes=0)
@jit_condition(static_argnums=(1))
def convert_harm2cart(Q, lmax):
    '''
    Convert the multipole moments in spherical representation to cartesian
    Basically the inverse operation of convert_cart2harm

    Inputs:
        Q:
            n * N_harm: stores the spherical harmonics moments of each site
        lmax:
            integer, the maximum L, currently only supports up to quadrupole

    Outputs:
        Theta:
            n * n_cart, stores the cartesian multipoles
    '''

    if lmax > 2:
        raise ValueError('l > 2 (beyond quadrupole) not supported')

    T_mono = Q[0:1]

    # dipole
    if lmax >= 1:
        T_dip = C1_h2c.dot(Q[1:4].T).T
    # quadrupole
    if lmax >= 2:
        T_quad = C2_h2c.dot(Q[4:9].T).T
    # octupole
    if lmax >= 3:
        T_octu = C3_h2c.dot(Q[9:16].T).T

    if lmax == 0:
        T = T_mono
    elif lmax == 1:
        T = jnp.hstack([T_mono, T_dip])
    elif lmax == 2:
        T = jnp.hstack([T_mono, T_dip, T_quad])
    else:
        T = jnp.hstack([T_mono, T_dip, T_quad, T_octu])

    return T


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
    if lmax > 3:
        raise NotImplementedError('l > 3 (beyond octupole) not supported')

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
    
    if lmax >= 3:
        octupoles = Q_gh[9:19]
        C3_gl_00 = zz*C2_gl_00 - 0.5773502691896258*(xz*C2_gl_10 + yz*C2_gl_20)
        C3_gl_01 = 0.25*(4.242640687119285*zz*C2_gl_01 - 2.449489742783178*(xz*C2_gl_11 + yz*C2_gl_21))
        C3_gl_02 = 0.25*(4.242640687119285*zz*C2_gl_02 - 2.449489742783178*(xz*C2_gl_12 + yz*C2_gl_22))
        C3_gl_03 = 0.4472135954999579*(3.0*zz*C2_gl_03 - 1.732050807568877*(xz*C2_gl_13 + yz*C2_gl_23))
        C3_gl_04 = 0.4472135954999579*(3.0*zz*C2_gl_04 - 1.732050807568877*(xz*C2_gl_14 + yz*C2_gl_24))
        C3_gl_05 = 0.3162277660168379*(1.732050807568877*(zx*C2_gl_03 - zy*C2_gl_04 - xx*C2_gl_13 + xy*C2_gl_14 - yx*C2_gl_23 + yy*C2_gl_24))
        C3_gl_06 = 0.5477225575051661*(zy*C2_gl_03 + zx*C2_gl_04) - 0.3162277660168379*(xy*C2_gl_13 + xx* C2_gl_14 + yy*C2_gl_23 + yx*C2_gl_24)
        C3_gl_10 = 0.2357022603955158*(3.464101615137755*xz*C2_gl_00 + 4.0*zz*C2_gl_10 - xz*C2_gl_30 - yz*C2_gl_40)
        C3_gl_11 = 0.25*(3.464101615137755*xz*C2_gl_01 + 4.0*zz*C2_gl_11 - xz*C2_gl_31 - yz*C2_gl_41)
        C3_gl_12 = 0.25*(3.464101615137755*xz*C2_gl_02 + 4.0*zz*C2_gl_12 - xz*C2_gl_32 - yz*C2_gl_42)
        C3_gl_13 = 0.3162277660168379*(3.464101615137755*xz*C2_gl_03 + 4.0*zz*C2_gl_13 - xz*C2_gl_33 - yz*C2_gl_43)
        C3_gl_14 = 0.3162277660168379*(3.464101615137755*xz*C2_gl_04 + 4.0*zz*C2_gl_14 - xz*C2_gl_34 - yz*C2_gl_44)
        C3_gl_15 = 0.07453559924999299*(-6.0*xy*C2_gl_04 + xx*(6.0*C2_gl_03 - 1.732050807568877*C2_gl_33) + 1.732050807568877*(4.0*zx*C2_gl_13 - 4.0*zy*C2_gl_14 + xy*C2_gl_34 - yy*C2_gl_43 + yx*C2_gl_44))
        C3_gl_16 = 0.07453559924999299*(6.0*(xy*C2_gl_03 + xx*C2_gl_04) + 6.928203230275509*(zy*C2_gl_13 + zx*C2_gl_14) - 1.732050807568877*(xy*C2_gl_33 + xx*C2_gl_34 + yy*C2_gl_43 + yx*C2_gl_44))
        C3_gl_20 = 0.2357022603955158*(4.0*zz*C2_gl_20 + yz*(3.464101615137755*C2_gl_00 +C2_gl_30) - xz*C2_gl_40)
        C3_gl_21 = 0.25*(4.*zz*C2_gl_21 + yz*(3.464101615137755*C2_gl_01 + C2_gl_31) - xz*C2_gl_41)
        C3_gl_22 = 0.25*(4.*zz*C2_gl_22 + yz*(3.464101615137755*C2_gl_02 + C2_gl_32) - xz*C2_gl_42)
        C3_gl_23 = 0.3162277660168379*(4.*zz*C2_gl_23 + yz*(3.464101615137755*C2_gl_03 + C2_gl_33) - xz*C2_gl_43)
        C3_gl_24 = 0.3162277660168379*(4.*zz*C2_gl_24 + yz*(3.464101615137755*C2_gl_04 + C2_gl_34) - xz*C2_gl_44)
        C3_gl_25 = 0.07453559924999299*(-6.*yy*C2_gl_04 + yx*(6.*C2_gl_03 + 1.732050807568877*C2_gl_33) + 1.732050807568877*(4.*zx*C2_gl_23 - 4.*zy*C2_gl_24 - yy*C2_gl_34 - xx*C2_gl_43 + xy*C2_gl_44))
        C3_gl_26 = 0.07453559924999299*(6.*yx*C2_gl_04 + yy*(6.*C2_gl_03 + 1.732050807568877*C2_gl_33) + 1.732050807568877*(4.*zy*C2_gl_23 + 4.*zx*C2_gl_24 + yx*C2_gl_34 - xy*C2_gl_43 - xx*C2_gl_44))
        C3_gl_30 = 0.7453559924999299*(xz*C2_gl_10 - yz*C2_gl_20 + zz*C2_gl_30)
        C3_gl_31 = 0.7905694150420948*(xz*C2_gl_11 - yz*C2_gl_21 + zz*C2_gl_31)
        C3_gl_32 = 0.7905694150420948*(xz*C2_gl_12 - yz*C2_gl_22 + zz*C2_gl_32)
        C3_gl_33 = xz*C2_gl_13 - yz*C2_gl_23 + zz*C2_gl_33
        C3_gl_34 = xz*C2_gl_14 - yz*C2_gl_24 + zz*C2_gl_34
        C3_gl_35 = 0.408248290463863*(xx*C2_gl_13 - xy*C2_gl_14 - yx*C2_gl_23 + yy*C2_gl_24 + zx*C2_gl_33 - zy*C2_gl_34)
        C3_gl_36 = 0.408248290463863*(xy*C2_gl_13 + xx*C2_gl_14 - yy*C2_gl_23 - yx*C2_gl_24 + zy*C2_gl_33 + zx*C2_gl_34)
        C3_gl_40 = 0.7453559924999299*(yz*C2_gl_10 + xz*C2_gl_20 + zz*C2_gl_40)
        C3_gl_41 = 0.7905694150420948*(yz*C2_gl_11 + xz*C2_gl_21 + zz*C2_gl_41)
        C3_gl_42 = 0.7905694150420948*(yz*C2_gl_12 + xz*C2_gl_22 + zz*C2_gl_42)
        C3_gl_43 = yz*C2_gl_13 + xz*C2_gl_23 + zz*C2_gl_43
        C3_gl_44 = yz*C2_gl_14 + xz*C2_gl_24 + zz*C2_gl_44
        C3_gl_45 = 0.408248290463863*(yx*C2_gl_13 - yy*C2_gl_14 + xx*C2_gl_23 - xy*C2_gl_24 + zx*C2_gl_43 - zy*C2_gl_44)
        C3_gl_46 = 0.408248290463863*(yy*C2_gl_13 + yx*C2_gl_14 + xy*C2_gl_23 + xx*C2_gl_24 + zy*C2_gl_43 + zx*C2_gl_44)
        C3_gl_50 = 0.9128709291752769*(xz*C2_gl_30 - yz*C2_gl_40)
        C3_gl_51 = 0.9682458365518542*(xz*C2_gl_31 - yz*C2_gl_41)
        C3_gl_52 = 0.9682458365518542*(xz*C2_gl_32 - yz*C2_gl_42)
        C3_gl_53 = 1.224744871391589*(xz*C2_gl_33 - yz*C2_gl_43)
        C3_gl_54 = 1.224744871391589*(xz*C2_gl_34 - yz*C2_gl_44)
        C3_gl_55 = 0.5*(xx*C2_gl_33 - xy*C2_gl_34 - yx*C2_gl_43 + yy*C2_gl_44)
        C3_gl_56 = 0.5*(xy*C2_gl_33 + xx*C2_gl_34 - yy*C2_gl_43 - yx*C2_gl_44)
        C3_gl_60 = 0.9128709291752769*(yz*C2_gl_30 + xz*C2_gl_40)
        C3_gl_61 = 0.9682458365518542*(yz*C2_gl_31 + xz*C2_gl_41)
        C3_gl_62 = 0.9682458365518542*(yz*C2_gl_32 + xz*C2_gl_42)
        C3_gl_63 = 1.224744871391589*(yz*C2_gl_33 + xz*C2_gl_43)
        C3_gl_64 = 1.224744871391589*(yz*C2_gl_34 + xz*C2_gl_44)
        C3_gl_65 = 0.5*(yx*C2_gl_33 - yy*C2_gl_34 + xx*C2_gl_43 - xy*C2_gl_44)
        C3_gl_66 = 0.5*(yy*C2_gl_33 + yx*C2_gl_34 + xy*C2_gl_43 + xx*C2_gl_44)
        
        # rotate
        C3_gl = jnp.array(
                [
                    [C3_gl_00, C3_gl_10, C3_gl_20, C3_gl_30, C3_gl_40, C3_gl_50, C3_gl_60],
                    [C3_gl_01, C3_gl_11, C3_gl_21, C3_gl_31, C3_gl_41, C3_gl_51, C3_gl_61],
                    [C3_gl_02, C3_gl_12, C3_gl_22, C3_gl_32, C3_gl_42, C3_gl_52, C3_gl_62],
                    [C3_gl_03, C3_gl_13, C3_gl_23, C3_gl_33, C3_gl_43, C3_gl_53, C3_gl_63],
                    [C3_gl_04, C3_gl_14, C3_gl_24, C3_gl_34, C3_gl_44, C3_gl_54, C3_gl_64],
                    [C3_gl_05, C3_gl_15, C3_gl_25, C3_gl_35, C3_gl_45, C3_gl_55, C3_gl_65],
                    [C3_gl_06, C3_gl_16, C3_gl_26, C3_gl_36, C3_gl_46, C3_gl_56, C3_gl_66]
                ]
        ).swapaxes(0,1)

        Q_lh_3 = jnp.einsum('jk,k->j', C3_gl, octupoles)
        
    if lmax == 0:
        Q_lh = Q_lh_0
    elif lmax == 1:
        Q_lh = jnp.hstack([Q_lh_0, Q_lh_1])
    elif lmax == 2:
        Q_lh = jnp.hstack([Q_lh_0, Q_lh_1, Q_lh_2])
    else:
        Q_lh = jnp.hstack([Q_lh_0, Q_lh_1, Q_lh_2, Q_lh_3])

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

