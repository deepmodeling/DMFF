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
rt6 = 2.44948974278
inv_rt6 = 1.0/rt6
rt10 = 3.16227766017
inv_rt10 = 1.0/rt10
rt8 = 2.82842712475
rt12 = 3.46410161514
rt15 = 3.87298334621
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
                    [    0,      0,      0,      0, rt3/rt5,   0,   -rt3/rt5,  0,     0,     0],
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
    elif lmax == 3:
        Q = jnp.hstack([Q_mono, Q_dip, Q_quad, Q_octu])
    else:
        raise ValueError('l > 3 (beyond octupole) not supported')

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

    if lmax > 3:
        raise ValueError('l > 3 (beyond octupole) not supported')

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
    elif lmax == 3:
        T = jnp.hstack([T_mono, T_dip, T_quad, T_octu])
    else:
        raise ValueError('l > 3 (beyond octupole) not supported')

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
        octupoles = Q_gh[9:16]
        C3_gl_00 = ( -8 * xx * yy + 8 * yx * xy + 5 * zz **3 + 5 * zz) / 2
        C3_gl_01 = (rt6*zx*(5.0*zz**2-1.0))/4.0
        C3_gl_02 = (rt6*zy*(5.0*zz**2-1.0))/4.0
        C3_gl_03 = (rt15*zz*(-2.0*zy**2-zz**2+1.0))/2.0
        C3_gl_04 = rt15*zx*zy*zz
        C3_gl_05 = (rt10*zx*(-4.0*zy**2-zz**2+1.0))/4.0
        C3_gl_06 = (rt10*zy*(-4.0*zy**2-3.0*zz**2+3.0))/4.0
        C3_gl_10 = (rt3*xz*(5.0*zz**2-1.0))/(2.0*rt2)
        C3_gl_11 = (-10.0*xx*yy**2+15.0*xx*zz**2-xx+10.0*yx*xy*yy)/4.0
        C3_gl_12 = (10.0*xy*yz**2+15.0*xy*zz**2-11.0*xy-10.0*yy*xz*yz)/4.0
        C3_gl_13 = (rt10*(4.0*xy*yy*yz-4.0*yy**2*xz-6.0*zy**2*xz-3.0*xz*zz**2+5.0*xz))/4.0
        C3_gl_14 = rt10*(-xx*yy*yz-yx*xy*yz+2.0*yx*yy*xz+3.0*zx*zy*xz)/2.0
        C3_gl_15 = (rt15*(-2.0*xx*yy**2-4.0*xx*zy**2-xx*zz**2+3.0*xx+2.0*yx*xy*yy))/4.0
        C3_gl_16 = (rt15*(-4.0*xy*zy**2-2.0*xy*yz**2-3.0*xy*zz**2+3.0*xy+2.0*yy*xz*yz))/4.0
        C3_gl_20 = (rt3*yz*(5.0*zz**2-1.0))/(2.0*rt2)
        C3_gl_21 = (10.0*yx*zy**2+15.0*yx*zz**2-11.0*yx-10.0*zx*yy*zy)/4.0
        C3_gl_22 = (5.0*yy*zz**2-yy+10.0*zy*yz*zz)/4.0
        C3_gl_23 = (rt10*(-4.0*yy*zy*zz-2.0*zy**2*yz-3.0*yz*zz**2+yz))/4.0
        C3_gl_24 = (rt10*(yx*zy*zz+zx*yy*zz+zx*zy*yz))/2.0
        C3_gl_25 = (rt15*(-2.0*yx*zy**2-yx*zz**2+yx-2.0*zx*yy*zy))/4.0
        C3_gl_26 = (rt15*(-4.0*yy*zy**2-yy*zz**2+yy-2.0*zy*yz*zz))/4.0
        C3_gl_30 = rt15*zz*(-2.0*yz**2-zz**2+1.0)/2.0
        C3_gl_31 = (rt10*(4.0*yx*yy*zy-4.0*zx*yy**2-6.0*zx*yz**2-3.0*zx*zz**2+5.0*zx))/4.0
        C3_gl_32 = (rt10*(-4.0*yy*yz*zz-2.0*zy*yz**2-3.0*zy*zz**2+zy))/4.0
        C3_gl_33 = (-4.0*xx*yy-4.0*yx*xy+12.0*yy**2*zz+6.0*zy**2*zz+6.0*yz**2*zz+3.0*zz**3-9.0*zz)/2.0
        C3_gl_34 = -6.0*yx*yy*zz-3.0*zx*zy*zz-4.0*xy*yy-2.0*xz*yz
        C3_gl_35 = (rt6*(4.0*yx*yy*zy+4.0*zx*yy**2+4.0*zx*zy**2+2.0*zx*yz**2+zx*zz**2-3.0*zx))/4.0
        C3_gl_36 = (rt6*(8.0*yy**2*zy+4.0*yy*yz*zz+4.0*zy**3+2.0*zy*yz**2+3.0*zy*zz**2-5.0*zy))/4.0
        C3_gl_40 = rt15*xz*yz*zz
        C3_gl_41 = (rt10*(-xx*yy*zy-yx*xy*zy+2.0*zx*xy*yy+3.0*zx*xz*yz))/2.0
        C3_gl_42 = (rt10*(xy*yz*zz+yy*xz*zz+zy*xz*yz))/2.0
        C3_gl_43 = -4.0*yx*yy-2.0*zx*zy-6.0*xy*yy*zz-3.0*xz*yz*zz
        C3_gl_44 = 3.0*xx*yy*zz+3.0*yx*xy*zz-4.0*yy**2-2.0*zy**2-2.0*yz**2-zz**2+3.0
        C3_gl_45 = (rt6*(-xx*yy*zy-yx*xy*zy-2.0*zx*xy*yy-zx*xz*yz))/2.0
        C3_gl_46 = (rt6*(-4.0*xy*yy*zy-xy*yz*zz-yy*xz*zz-zy*xz*yz))/2.0
        C3_gl_50 = (rt5*xz*(-4.0*yz**2-zz**2+1.0))/(2.0*rt2)
        C3_gl_51 = (rt15*(-2.0*xx*yy**2-4.0*xx*yz**2-xx*zz**2+3.0*xx+2.0*yx*xy*yy))/4.0
        C3_gl_52 = (rt15*(-2.0*xy*yz**2-xy*zz**2+xy-2.0*yy*xz*yz))/4.0
        C3_gl_53 = (rt6*(4.0*xy*yy*yz+4.0*yy**2*xz+2.0*zy**2*xz+4.0*xz*yz**2+xz*zz**2-3.0*xz))/4.0
        C3_gl_54 = (rt6*(-xx*yy*yz-yx*xy*yz-2.0*yx*yy*xz-zx*zy*xz))/2.0
        C3_gl_55 = (10.0*xx*yy**2+4.0*xx*zy**2+4.0*xx*yz**2+xx*zz**2-7.0*xx+6.0*yx*xy*yy)/4.0
        C3_gl_56 = (16.0*xy*yy**2+4.0*xy*zy**2+6.0*xy*yz**2+3.0*xy*zz**2-7.0*xy+6.0*yy*xz*yz)/4.0
        C3_gl_60 = (rt5*yz*(-4.0*yz**2-3.0*zz**2+3.0))/(2.0*rt2)
        C3_gl_61 = (rt15*(-2.0*yx*zy**2-4.0*yx*yz**2-3.0*yx*zz**2+3.0*yx+2.0*zx*yy*zy))/4.0
        C3_gl_62 = (rt15*(-4.0*yy*yz**2-yy*zz**2+yy-2.0*zy*yz*zz))/4.0
        C3_gl_63 = (rt6*(8.0*yy**2*yz+4.0*yy*zy*zz+2.0*zy**2*yz+4.0*yz**3+3.0*yz*zz**2-5.0*yz))/4.0
        C3_gl_64 = (rt6*(-4.0*yx*yy*yz-yx*zy*zz-zx*yy*zz-zx*zy*yz))/2.0
        C3_gl_65 = (16.0*yx*yy**2+6.0*yx*zy**2+4.0*yx*yz**2+3.0*yx*zz**2-7.0*yx+6.0*zx*yy*zy)/4.0
        C3_gl_66 = (16.0*yy**3+12.0*yy*zy**2+12.0*yy*yz**2+3.0*yy*zz**2-15.0*yy+6.0*zy*yz*zz)/4.0
        
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

