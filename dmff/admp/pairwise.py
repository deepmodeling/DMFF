import sys
from jax import vmap
import jax.numpy as jnp
from dmff.utils import jit_condition
from dmff.admp.spatial import v_pbc_shift
from functools import partial

# for debug
# from jax_md import partition, space
# from admp.parser import *
# from admp.multipole import *
# from jax import grad, value_and_grad
# from admp.pme import *

# jitted and vmapped parameter distributors
# all three look identical, but they assume different input shapes
# you should use different functions for different inputs, to avoid recompiling
@partial(vmap, in_axes=(None, 0), out_axes=(0))
@jit_condition(static_argnums=())
def distribute_scalar(params, index):
    return params[index]


@partial(vmap, in_axes=(None, 0), out_axes=(0))
@jit_condition(static_argnums=())
def distribute_v3(pos, index):
    return pos[index]


@partial(vmap, in_axes=(None, 0), out_axes=(0))
@jit_condition(static_argnums=())
def distribute_multipoles(multipoles, index):
    return multipoles[index]


@partial(vmap, in_axes=(None, 0), out_axes=(0))
@jit_condition(static_argnums=())
def distribute_dispcoeff(c_list, index):
    return c_list[index]


def generate_pairwise_interaction(pair_int_kernel, covalent_map, static_args):
    '''
    This is a calculator generator for pairwise interaction 

    Input:
        pair_int_kernel:
            function type (dr, m, p1i, p1j, p2i, p2j) -> energy : the vectorized kernel function, 
            dr is the distance, m is the topological scaling factor, p1i, p1j, p2i, p2j are pairwise parameters

        covalent_map:
            Na * Na, int: the covalent_map matrix that marks the topological distances between atoms

        static_args:
            dict: a dictionary that stores all static global parameters (such as lmax, kappa, etc)

    Output:
        pair_int:
            function type (positions, box, pairs, mScales, p1, p2, ...) -> energy
            The pair interaction calculator. p1, p2 ... involved atomic parameters, the order should be consistent
            with the order in kernel
    '''

    def pair_int(positions, box, pairs, mScales, *atomic_params):
        pairs =  pairs[pairs[:, 0] < pairs[:, 1]]
        ri = distribute_v3(positions, pairs[:, 0])
        rj = distribute_v3(positions, pairs[:, 1])
        # ri = positions[pairs[:, 0]]
        # rj = positions[pairs[:, 1]]
        nbonds = covalent_map[pairs[:, 0], pairs[:, 1]]
        mscales = distribute_scalar(mScales, nbonds-1)
        # mscales = mScales[nbonds-1]
        box_inv = jnp.linalg.inv(box)
        dr = ri - rj
        dr = v_pbc_shift(dr, box, box_inv)
        dr = jnp.linalg.norm(dr, axis=1)

        pair_params = []
        for i, param in enumerate(atomic_params):
            pair_params.append(distribute_scalar(param, pairs[:, 0]))
            pair_params.append(distribute_scalar(param, pairs[:, 1]))
            # pair_params.append(param[pairs[:, 0]])
            # pair_params.append(param[pairs[:, 1]])

        energy = jnp.sum(pair_int_kernel(dr, mscales, *pair_params))
        return energy

    return pair_int


@vmap
@jit_condition(static_argnums={})
def TT_damping_qq_c6_kernel(dr, m, ai, aj, bi, bj, qi, qj, ci, cj):
    a = jnp.sqrt(ai * aj)
    b = jnp.sqrt(bi * bj)
    c = ci * cj
    q = qi * qj
    r = dr * 1.889726878 # convert to bohr
    br = b * r
    br2 = br * br
    br3 = br2 * br
    br4 = br3 * br
    br5 = br4 * br
    br6 = br5 * br
    exp_br = jnp.exp(-br)
    f = 2625.5 * a * exp_br \
        + (-2625.5) * exp_br * (1+br) * q / r \
        + exp_br*(1+br+br2/2+br3/6+br4/24+br5/120+br6/720) * c / dr**6

    return f * m

@vmap
@jit_condition(static_argnums=())
def slater_disp_damping_kernel(dr, m, bi, bj, c6i, c6j, c8i, c8j, c10i, c10j):
    '''
    Slater-ISA type damping for dispersion:
    f(x) = -e^{-x} * \sum_{k} x^k/k!
    x = Br - \frac{2*(Br)^2 + 3Br}{(Br)^2 + 3*Br + 3}
    see jctc 12 3851
    '''
    b = jnp.sqrt(bi*bj)
    c6 = jnp.sqrt(c6i*c6j)
    c8 = jnp.sqrt(c8i*c8j)
    c10 = jnp.sqrt(c10i*c10j)
    br = b * dr
    br2 = br * br
    x = br - (2*br2 + 3*br) / (br2 + 3*br + 3)
    s6 = 1 + x + x**2/2 + x**3/6 + x**4/24 + x**5/120 + x**6/720
    s8 = s6 + x**7/5040 + x**8/40320
    s10 = s8 + x**9/362880 + x**10/3628800
    exp_x = jnp.exp(-x)
    f6 = exp_x * s6
    f8 = exp_x * s8
    f10 = exp_x * s10
    return (f6*c6/dr**6 + f8*c8/dr**8 + f10*c10/dr**10) * m


@vmap
@jit_condition(static_argnums=())
def slater_sr_kernel(dr, m, ai, aj, bi, bj):
    '''
    Slater-ISA type short range terms
    see jctc 12 3851
    '''
    b = jnp.sqrt(bi * bj)
    a = jnp.sqrt(ai * aj)
    br = b * dr
    br2 = br * br
    P = 1/3 * br2 + br + 1 
    return a * P * jnp.exp(-br) * m


def validation(pdb):
    xml = 'mpidwater.xml'
    pdbinfo = read_pdb(pdb)
    serials = pdbinfo['serials']
    names = pdbinfo['names']
    resNames = pdbinfo['resNames']
    resSeqs = pdbinfo['resSeqs']
    positions = pdbinfo['positions']
    box = pdbinfo['box'] # a, b, c, α, β, γ
    charges = pdbinfo['charges']
    positions = jnp.asarray(positions)
    lx, ly, lz, _, _, _ = box
    box = jnp.eye(3)*jnp.array([lx, ly, lz])

    mScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
    pScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])
    dScales = jnp.array([0.0, 0.0, 0.0, 1.0, 1.0])

    rc = 4  # in Angstrom
    ethresh = 1e-4

    n_atoms = len(serials)

    atomTemplate, residueTemplate = read_xml(xml)
    atomDicts, residueDicts = init_residues(serials, names, resNames, resSeqs, positions, charges, atomTemplate, residueTemplate)

    covalent_map = assemble_covalent(residueDicts, n_atoms)
    displacement_fn, shift_fn = space.periodic_general(box, fractional_coordinates=False)
    neighbor_list_fn = partition.neighbor_list(displacement_fn, box, rc, 0, format=partition.OrderedSparse)
    nbr = neighbor_list_fn.allocate(positions)
    pairs = nbr.idx.T

    pmax = 10
    kappa, K1, K2, K3 = setup_ewald_parameters(rc, ethresh, box)
    kappa = 0.657065221219616

    # construct the C list
    c_list = np.zeros((3, n_atoms))
    a_list = np.zeros(n_atoms)
    q_list = np.zeros(n_atoms)
    b_list = np.zeros(n_atoms)
    nmol=int(n_atoms/3)
    for i in range(nmol):
        a = i*3
        b = i*3+1
        c = i*3+2
        # dispersion coeff
        c_list[0][a]=37.199677405
        c_list[0][b]=7.6111103
        c_list[0][c]=7.6111103
        c_list[1][a]=85.26810658
        c_list[1][b]=11.90220148
        c_list[1][c]=11.90220148
        c_list[2][a]=134.44874488
        c_list[2][b]=15.05074749
        c_list[2][c]=15.05074749
        # q
        q_list[a] = -0.741706
        q_list[b] = 0.370853
        q_list[c] = 0.370853
        # b, Bohr^-1
        b_list[a] = 2.00095977
        b_list[b] = 1.999519942
        b_list[c] = 1.999519942
        # a, Hartree
        a_list[a] = 458.3777
        a_list[b] = 0.0317
        a_list[c] = 0.0317


    c_list = jnp.array(c_list)

#     @partial(vmap, in_axes=(0, 0, 0, 0), out_axes=(0))
#     @jit_condition(static_argnums=())
#     def disp6_pme_real_kernel(dr, m, ci, cj):
#         # unpack static arguments
#         kappa = static_args['kappa']
#         # calculate distance
#         dr2 = dr ** 2
#         dr6 = dr2 ** 3
#         # do calculation
#         x2 = kappa**2 * dr2
#         exp_x2 = jnp.exp(-x2)
#         x4 = x2 * x2
#         g = (1 + x2 + 0.5*x4) * exp_x2
#         return (m + g - 1) * ci * cj / dr6
   
#     static_args = {'kappa': kappa}
#     disp6_pme_real = generate_pairwise_interaction(disp6_pme_real_kernel, covalent_map, static_args)
#     print(disp6_pme_real(positions, box, pairs, mScales, c_list[0, :]))
    
    TT_damping_qq_c6 = generate_pairwise_interaction(TT_damping_qq_c6_kernel, covalent_map, static_args={})

    TT_damping_qq_c6(positions, box, pairs, mScales, a_list, b_list, q_list, c_list[0])
    print('ok')
    print(TT_damping_qq_c6(positions, box, pairs, mScales, a_list, b_list, q_list, c_list[0]))
    return 


# below is the validation code
if __name__ == '__main__':
    validation(sys.argv[1])
