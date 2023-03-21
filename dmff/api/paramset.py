import jax
import jax.numpy as jnp


class ParamSet:
    def __init__(self, data=None):
        if data is None:
            self.parameters = {}
        else:
            self.parameters = data

    def addField(self, field):
        self.parameters[field] = {}

    def addParameter(self, values, name, field=None):
        if field is not None:
            self.parameters[field][name] = values
        else:
            self.parameters[name] = values

    def to_jax(self):
        for key1 in self.parameters:
            if isinstance(self.parameters[key1], dict):
                for key2 in self.parameters[key1]:
                    self.parameters[key1][key2] = jnp.array(
                        self.parameters[key1][key2])
            else:
                self.parameters[key1] = jnp.array(self.parameters[key1])

    def __getitem__(self, key):
        return self.parameters[key]


def flatten_paramset(prmset):
    return prmset.parameters, None


def unflatten_paramset(aux_data, contents):
    ret = ParamSet(data=contents)
    return ret


jax.tree_util.register_pytree_node(ParamSet, flatten_paramset,
                                   unflatten_paramset)
