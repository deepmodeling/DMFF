#!/usr/bin/env python
"""
Example: Automatic Differentiation of Energy with Respect to Charges

This example demonstrates how to use DMFF to compute gradients of energy
with respect to point charges, enabling charge optimization.
"""

import jax
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
from dmff import Hamiltonian


def main():
    print("=" * 70)
    print("DMFF Charge Autodiff Example")
    print("=" * 70)
    
    # Load a simple water system
    print("\n1. Loading system...")
    pdb = app.PDBFile('tests/data/water_dimer.pdb')
    print(f"   ✓ Loaded {pdb.topology.getNumAtoms()} atoms")
    
    # Load force field
    print("\n2. Loading force field...")
    ff = Hamiltonian('tests/data/water4.xml')
    print("   ✓ Force field loaded")
    
    # Create potential
    print("\n3. Creating potential...")
    potential = ff.createPotential(pdb.topology, nonbondedMethod=app.NoCutoff)
    print("   ✓ Potential created")
    
    # Get parameters
    print("\n4. Extracting parameters...")
    paramset = ff.getParameters()
    
    # Find which force contains charges
    force_name = None
    if "CoulombForce" in paramset.parameters and "charge" in paramset.parameters["CoulombForce"]:
        force_name = "CoulombForce"
    elif "NonbondedForce" in paramset.parameters and "charge" in paramset.parameters["NonbondedForce"]:
        force_name = "NonbondedForce"
    
    if force_name is None:
        print("   ✗ No charges found in paramset")
        return
    
    charges = paramset.parameters[force_name]["charge"]
    print(f"   ✓ Found {len(charges)} charges in {force_name}")
    print(f"   Initial charges: {charges}")
    
    # Get energy function
    print("\n5. Creating energy function...")
    efunc = jax.jit(potential.getPotentialFunc())
    print("   ✓ Energy function created and JIT compiled")
    
    # Prepare system state
    print("\n6. Preparing system state...")
    positions = jnp.array(pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    box = jnp.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    pairs = jnp.zeros((0, 3), dtype=jnp.int32)  # Empty pairs for NoCutoff
    print(f"   ✓ System prepared: {len(positions)} atoms")
    
    # Define energy as a function of charges
    print("\n7. Computing energy and gradients...")
    
    def energy_wrt_charges(charge_params):
        """Energy as a function of charges only"""
        import copy
        params = copy.deepcopy(paramset.parameters)
        params[force_name]["charge"] = charge_params
        return efunc(positions, box, pairs, params)
    
    # Compute energy and gradient
    energy_and_grad = jax.value_and_grad(energy_wrt_charges)
    energy, grad_charges = energy_and_grad(charges)
    
    print(f"   ✓ Energy: {energy:.6f} kJ/mol")
    print(f"   ✓ Charge gradients computed")
    print(f"      Gradient shape: {grad_charges.shape}")
    print(f"      Gradient range: [{jnp.min(grad_charges):.6e}, {jnp.max(grad_charges):.6e}]")
    print(f"      Gradients: {grad_charges}")
    
    # Demonstrate that we can perturb charges and see energy change
    print("\n8. Validating gradients with finite differences...")
    epsilon = 1e-5
    numerical_grads = []
    
    for i in range(len(charges)):
        charges_plus = charges.at[i].add(epsilon)
        charges_minus = charges.at[i].add(-epsilon)
        
        energy_plus = energy_wrt_charges(charges_plus)
        energy_minus = energy_wrt_charges(charges_minus)
        
        numerical_grad = (energy_plus - energy_minus) / (2 * epsilon)
        numerical_grads.append(numerical_grad)
    
    numerical_grads = jnp.array(numerical_grads)
    
    # Compare analytical and numerical gradients
    max_diff = jnp.max(jnp.abs(grad_charges - numerical_grads))
    relative_diff = max_diff / (jnp.max(jnp.abs(grad_charges)) + 1e-10)
    
    print(f"   Analytical gradients: {grad_charges}")
    print(f"   Numerical gradients:  {numerical_grads}")
    print(f"   Max absolute diff: {max_diff:.6e}")
    print(f"   Relative diff: {relative_diff:.6e}")
    
    if relative_diff < 1e-4:
        print("   ✓ Gradients validated successfully!")
    else:
        print("   ⚠ Large gradient mismatch - check implementation")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print("\nNext steps:")
    print("  - Use these gradients for charge optimization")
    print("  - Fit charges to QM energies or experimental data")
    print("  - Combine with other parameter gradients for joint optimization")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
