# Automatic Differentiation for Charges and Virtual Site Weights

## Summary

This feature enables automatic differentiation of energy with respect to point charges and virtual site weights in the DMFF force field framework. This allows for parameter optimization of charges and virtual site positions.

## Motivation

Previously, charges and virtual site weights were hardcoded during potential creation and could not be optimized through gradient-based methods. This feature makes these parameters differentiable, enabling:

1. Optimization of point charges to fit QM energies or experimental data
2. Optimization of virtual site positions for better force field accuracy
3. Joint optimization of multiple force field parameters including charges and vsite weights

## Implementation Details

### Changes to Charge Handling

1. **Parameter Storage**: Charges are now stored in `ParamSet` during Hamiltonian initialization:
   - `CoulombGenerator` stores charges (per atom type) under `params["CoulombForce"]["charge"]` during `__init__`
   - `NonbondedGenerator` stores charges (per atom type) under `params["NonbondedForce"]["charge"]` during `__init__`
   - When `createPotential()` is called with a topology, these charges are updated to be per-atom (topology-dependent)
   
   **Note**: During initialization, charges are stored per atom TYPE based on the force field definition. After calling `createPotential()`, they are updated to be per ATOM based on the actual molecular topology.

2. **Force Class Updates**: All Coulomb force classes now accept charges as runtime parameters:
   - `CoulombNoCutoffForce`
   - `CoulReactionFieldForce`
   - `CoulombPMEForce`

3. **Energy Function**: The energy function now reads charges from the `params` dictionary passed at runtime, enabling differentiation.

### Changes to Virtual Site Weight Handling

1. **Parameter Storage**: Virtual site weights are stored in `ParamSet` during `buildVSiteUpdateFunction()`:
   - Type 2 (average2): `params["VirtualSite"]["vsite_w2_type_2"]`
   - Type 3 (average3): `params["VirtualSite"]["vsite_w2_type_3"]`, `params["VirtualSite"]["vsite_w3_type_3"]`
   - Type 2fd: `params["VirtualSite"]["vsite_dist_type_2fd"]`
   - Type 3fd: `params["VirtualSite"]["vsite_dist_type_3fd"]`

2. **Update Function**: The `update_pos` function now accepts optional `vsite_params` argument and uses weights from params if provided.

3. **Potential Function**: The `getPotentialFunc()` now extracts vsite parameters from params and passes them to the position update function.

## Usage Example

### Differentiating Energy with Respect to Charges

```python
import jax
import jax.numpy as jnp
from dmff import Hamiltonian
import openmm.app as app
import openmm.unit as unit

# Load force field and create potential
pdb = app.PDBFile('system.pdb')
ff = Hamiltonian('forcefield.xml')
potential = ff.createPotential(pdb.topology, nonbondedMethod=app.NoCutoff)

# Get parameters
paramset = ff.getParameters()

# Get energy function
efunc = jax.jit(potential.getPotentialFunc())

# Prepare inputs
positions = jnp.array(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
box = jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
pairs = jnp.zeros((0, 3), dtype=jnp.int32)  # For NoCutoff

# Define function to compute energy as a function of charges only
def energy_wrt_charges(charges):
    import copy
    params = copy.deepcopy(paramset.parameters)
    params["CoulombForce"]["charge"] = charges  # or "NonbondedForce" depending on your force field
    return efunc(positions, box, pairs, params)

# Get original charges
original_charges = paramset.parameters["CoulombForce"]["charge"]

# Compute gradients
grad_fn = jax.grad(energy_wrt_charges)
charge_gradients = grad_fn(original_charges)

print(f"Charge gradients: {charge_gradients}")
```

### Differentiating Energy with Respect to Virtual Site Weights

```python
import jax
import jax.numpy as jnp
from dmff import Hamiltonian

# Load force field with virtual sites
ff = Hamiltonian('forcefield_with_vsites.xml')
potential = ff.createPotential(pdb.topology, nonbondedMethod=app.NoCutoff)

# Get parameters
paramset = ff.getParameters()

# Define function to compute energy as a function of vsite weights
def energy_wrt_vsite_weights(weights):
    import copy
    params = copy.deepcopy(paramset.parameters)
    params["VirtualSite"]["vsite_w2_type_2"] = weights
    return efunc(positions, box, pairs, params)

# Get original weights
original_weights = paramset.parameters["VirtualSite"]["vsite_w2_type_2"]

# Compute gradients
grad_fn = jax.grad(energy_wrt_vsite_weights)
vsite_gradients = grad_fn(original_weights)

print(f"Virtual site weight gradients: {vsite_gradients}")
```

### Joint Optimization

You can also optimize charges and vsite weights simultaneously:

```python
def energy_wrt_all_params(charges, vsite_weights):
    import copy
    params = copy.deepcopy(paramset.parameters)
    params["CoulombForce"]["charge"] = charges
    params["VirtualSite"]["vsite_w2_type_2"] = vsite_weights
    return efunc(positions, box, pairs, params)

# Compute gradients with respect to both
grad_fn = jax.grad(energy_wrt_all_params, argnums=(0, 1))
charge_grads, vsite_grads = grad_fn(original_charges, original_weights)
```

## API Changes

### Modified Functions

1. **`Generator.createPotential()`**: Now accepts an optional `paramset: ParamSet` parameter
   - All generators (HarmonicBondGenerator, NonbondedGenerator, CoulombGenerator, etc.) have been updated

2. **`DMFFTopology.buildVSiteUpdateFunction()`**: Now accepts an optional `paramset: ParamSet` parameter
   - Stores virtual site weights in paramset when provided

3. **`Potential.getPotentialFunc()`**: Updated to extract and pass virtual site parameters
   - The returned energy function now properly handles vsite params from the parameters dictionary

### Force Class Changes

All Coulomb force classes have been updated to accept charges as runtime parameters:

- `CoulNoCutoffForce.generate_get_energy()`: Returns function that accepts charges
- `CoulReactionFieldForce.generate_get_energy()`: Returns function that accepts charges
- `CoulombPMEForce.generate_get_energy()`: Returns function that accepts charges

## Backward Compatibility

The changes are designed to be backward compatible:

1. The `paramset` parameter in `createPotential()` is optional (defaults to `None`)
2. The `vsite_params` parameter in `update_pos()` is optional (defaults to `None`)
3. When not provided, the functions fall back to using the original hardcoded values

## Testing

Tests have been added in `tests/test_frontend/test_charge_vsite_autodiff.py`:

- `test_charge_autodiff_simple()`: Validates that charges are stored in paramset
- `test_vsite_weight_autodiff_simple()`: Validates that vsite weights are stored in paramset
- `test_charge_gradient_computation()`: Validates that gradients can be computed

## Troubleshooting

### Charges not appearing in paramset

**Issue**: `paramset.parameters["CoulombForce"]["charge"]` or `paramset.parameters["NonbondedForce"]["charge"]` doesn't exist or throws KeyError.

**Solution**: Charges should now be available immediately after Hamiltonian initialization. They are stored per atom type. If you still don't see charges:
1. Verify your XML has charge definitions (either in `<NonbondedForce>` atoms or in `<Residues>` atoms)
2. Check the correct force field name (use "NonbondedForce" if your XML uses `<NonbondedForce>`, or "CoulombForce" if it uses `<CoulombForce>`)

Example:
```python
ff = Hamiltonian('forcefield.xml')
# Charges are NOW available immediately
paramset = ff.getParameters()

# For NonbondedForce XML:
charges = paramset.parameters["NonbondedForce"]["charge"]  # Per atom type
print(f"Initial charges (per type): {charges}")

# After createPotential, charges are updated to be per atom
potential = ff.createPotential(topology, nonbondedMethod=app.NoCutoff)
paramset = ff.getParameters()
charges = paramset.parameters["NonbondedForce"]["charge"]  # Per atom in topology
print(f"Updated charges (per atom): {charges}")
```

### Understanding per-type vs per-atom charges

- **After Hamiltonian initialization**: Charges are per atom TYPE (one charge value per unique atom type defined in the force field)
- **After createPotential()**: Charges are per ATOM (one charge value per atom in the actual molecular system)

For example, if your force field defines types "H" and "O", you'll have 2 charge values initially. After creating a potential for a water box with 300 water molecules, you'll have 900 charge values (300 O atoms + 600 H atoms).

## Future Enhancements

Potential future improvements:

1. Add support for more virtual site types (4fdn, 3out, 3fad)
2. Provide helper functions for parameter optimization workflows
3. Add examples of charge optimization from QM data
4. Support for masking specific parameters from optimization
