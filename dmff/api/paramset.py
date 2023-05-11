import jax
import jax.numpy as jnp
from typing import Dict, Union


class ParamSet:
    """
    A class to store and manipulate a set of parameters.

    Attributes:
    -----------
    parameters: Dict[str, Union[Dict[str, jnp.ndarray], jnp.ndarray]]
        A dictionary containing the parameters. If a parameter belongs to a field, it is stored in a nested dictionary.
    mask: Dict[str, Union[Dict[str, jnp.ndarray], jnp.ndarray]]
        A dictionary containing the masks for each parameter. If a parameter belongs to a field, it is stored in a nested dictionary.

    Methods:
    --------
    addField(field: str) -> None:
        Adds a new field to the parameters and mask dictionaries.
    addParameter(values: jnp.ndarray, name: str, field: str = None, mask: jnp.ndarray = None) -> None:
        Adds a new parameter to the parameters and mask dictionaries.
    to_jax() -> None:
        Converts all parameters to jax arrays.
    """

    def __init__(self, data: Dict[str, Union[Dict[str, jnp.ndarray], jnp.ndarray]] = None, mask: Dict[str, Union[Dict[str, jnp.ndarray], jnp.ndarray]] = None):
        """
        Initializes a new ParamSet object.

        Parameters:
        -----------
        data: Dict[str, Union[Dict[str, jnp.ndarray], jnp.ndarray]], optional
            A dictionary containing the parameters. If a parameter belongs to a field, it is stored in a nested dictionary.
        mask: Dict[str, Union[Dict[str, jnp.ndarray], jnp.ndarray]], optional
            A dictionary containing the masks for each parameter. If a parameter belongs to a field, it is stored in a nested dictionary.
        """
        self.parameters = data if data is not None else {}
        self.mask = mask if mask is not None else {}

    def addField(self, field: str) -> None:
        """
        Adds a new field to the parameters and mask dictionaries.

        Parameters:
        -----------
        field: str
            The name of the new field.
        """
        self.parameters[field] = {}
        self.mask[field] = {}

    def addParameter(self, values: jnp.ndarray, name: str, field: str = None, mask: jnp.ndarray = None) -> None:
        """
        Adds a new parameter to the parameters and mask dictionaries.

        Parameters:
        -----------
        values: jnp.ndarray
            The values of the new parameter.
        name: str
            The name of the new parameter.
        field: str, optional
            The name of the field to which the parameter belongs.
        mask: jnp.ndarray, optional
            The mask of the new parameter.
        """
        if field is not None:
            self.parameters[field][name] = values
            if mask is None:
                self.mask[field][name] = jnp.ones(values.shape)
            else:
                self.mask[field][name] = jnp.array(mask)
        else:
            self.parameters[name] = values
            if mask is None:
                self.mask[name] = jnp.ones(values.shape)
            else:
                self.mask[name] = jnp.array(mask)

    def to_jax(self) -> None:
        """
        Converts all parameters to jax arrays.
        """
        for key1 in self.parameters:
            if isinstance(self.parameters[key1], dict):
                for key2 in self.parameters[key1]:
                    self.parameters[key1][key2] = jnp.array(
                        self.parameters[key1][key2])
            else:
                self.parameters[key1] = jnp.array(self.parameters[key1])

    def __getitem__(self, key: str) -> Union[Dict[str, jnp.ndarray], jnp.ndarray]:
        """
        Returns the value of the parameter with the given key.

        Parameters:
        -----------
        key: str
            The name of the parameter.

        Returns:
        --------
        Union[Dict[str, jnp.ndarray], jnp.ndarray]
            The value of the parameter.
        """
        return self.parameters[key]


def flatten_paramset(prmset: ParamSet) -> tuple:
    """
    Flattens a ParamSet object.

    Parameters:
    -----------
    prmset: ParamSet
        The ParamSet object to be flattened.

    Returns:
    --------
    tuple
        A tuple containing the parameters and masks.
    """
    return [prmset.parameters], prmset.mask


def unflatten_paramset(aux_data: Dict, contents: tuple) -> ParamSet:
    """
    Unflattens a ParamSet object.

    Parameters:
    -----------
    aux_data: None
        Unused.
    contents: tuple
        A tuple containing the parameters and masks.

    Returns:
    --------
    ParamSet
        The unflattened ParamSet object.
    """
    return ParamSet(data=contents[0], mask=aux_data)


jax.tree_util.register_pytree_node(
    ParamSet, flatten_paramset, unflatten_paramset)
