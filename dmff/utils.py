import jax.numpy as jnp


class DMFFException(BaseException):
    pass


_standardResidues = ['ALA', 'ASN', 'CYS', 'GLU', 'HIS', 'LEU', 'MET', 'PRO', 'THR', 'TYR',
                     'ARG', 'ASP', 'GLN', 'GLY', 'ILE', 'LYS', 'PHE', 'SER', 'TRP', 'VAL',
                     'A', 'G', 'C', 'U', 'I', 'DA', 'DG', 'DC', 'DT', 'DI', 'HOH']


def dict_to_jnp(inp: dict):
    newdict = {}
    for key in inp.keys():
        if isinstance(inp[key], dict):
            newdict[key] = dict_to_jnp(inp[key])
        else:
            newdict[key] = jnp.array(inp[key])
    return newdict