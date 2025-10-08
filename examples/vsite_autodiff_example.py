#!/usr/bin/env python
"""
Example: Automatic Differentiation of Energy with Respect to Virtual Site Weights

This example demonstrates how to use DMFF to compute gradients of energy
with respect to virtual site weights, enabling virtual site optimization.
"""

import jax
import jax.numpy as jnp
import openmm.app as app
import openmm.unit as unit
from dmff import Hamiltonian


def main():
    print("=" * 70)
    print("DMFF Virtual Site Weight Autodiff Example")
    print("=" * 70)
    
    # Try to load a system with virtual sites
    print("\n1. Loading system with virtual sites...")
    try:
        pdb = app.PDBFile('tests/data/chloropyridine.pdb')
        ff_file = 'tests/data/chloropyridine_vsite.xml'
    except FileNotFoundError:
        print("   ⚠ Virtual site test files not found")
        print("   This example requires a system with virtual sites")
        return
    
    print(f"   ✓ Loaded {pdb.topology.getNumAtoms()} atoms")
    
    # Load force field
    print("\n2. Loading force field with virtual sites...")
    ff = Hamiltonian(ff_file)
    print("   ✓ Force field loaded")
    
    # Create potential
    print("\n3. Creating potential...")
    potential = ff.createPotential(pdb.topology, nonbondedMethod=app.NoCutoff)
    print("   ✓ Potential created")
    
    # Get parameters
    print("\n4. Extracting virtual site parameters...")
    paramset = ff.getParameters()
    
    # Check for virtual site parameters
    if "VirtualSite" not in paramset.parameters:
        print("   ⚠ No virtual site parameters found in this force field")
        print("   The system may not have virtual sites defined")
        return
    
    print(f"   ✓ Virtual site parameters found:")
    for key, value in paramset.parameters["VirtualSite"].items():
        print(f"      {key}: shape {value.shape}, values {value}")
    
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
    
    # Compute initial energy
    print("\n7. Computing initial energy...")
    initial_energy = efunc(positions, box, pairs, paramset.parameters)
    print(f"   ✓ Initial energy: {initial_energy:.6f} kJ/mol")
    
    # Example: Differentiate with respect to type 2 virtual site weights (if present)
    if "vsite_w2_type_2" in paramset.parameters["VirtualSite"]:
        print("\n8. Computing gradients with respect to type 2 vsite weights...")
        
        vsite_weights = paramset.parameters["VirtualSite"]["vsite_w2_type_2"]
        
        def energy_wrt_vsite_w2(weights):
            """Energy as a function of type 2 vsite weights only"""
            import copy
            params = copy.deepcopy(paramset.parameters)
            params["VirtualSite"]["vsite_w2_type_2"] = weights
            return efunc(positions, box, pairs, params)
        
        # Compute gradient
        grad_fn = jax.grad(energy_wrt_vsite_w2)
        grad_weights = grad_fn(vsite_weights)
        
        print(f"   ✓ Gradients computed for {len(vsite_weights)} vsite weights")
        print(f"      Original weights: {vsite_weights}")
        print(f"      Gradients: {grad_weights}")
        print(f"      Gradient range: [{jnp.min(grad_weights):.6e}, {jnp.max(grad_weights):.6e}]")
    
    # Example: Differentiate with respect to type 3 virtual site weights (if present)
    if "vsite_w2_type_3" in paramset.parameters["VirtualSite"]:
        print("\n9. Computing gradients with respect to type 3 vsite weights...")
        
        w2 = paramset.parameters["VirtualSite"]["vsite_w2_type_3"]
        w3 = paramset.parameters["VirtualSite"]["vsite_w3_type_3"]
        
        def energy_wrt_vsite_w23(w2_param, w3_param):
            """Energy as a function of type 3 vsite weights"""
            import copy
            params = copy.deepcopy(paramset.parameters)
            params["VirtualSite"]["vsite_w2_type_3"] = w2_param
            params["VirtualSite"]["vsite_w3_type_3"] = w3_param
            return efunc(positions, box, pairs, params)
        
        # Compute gradients with respect to both w2 and w3
        grad_fn = jax.grad(energy_wrt_vsite_w23, argnums=(0, 1))
        grad_w2, grad_w3 = grad_fn(w2, w3)
        
        print(f"   ✓ Gradients computed for type 3 vsite weights")
        print(f"      w2: {w2}, grad: {grad_w2}")
        print(f"      w3: {w3}, grad: {grad_w3}")
    
    # Example: Differentiate with respect to distance-based vsite parameters
    if "vsite_dist_type_2fd" in paramset.parameters["VirtualSite"]:
        print("\n10. Computing gradients with respect to 2fd vsite distances...")
        
        distances = paramset.parameters["VirtualSite"]["vsite_dist_type_2fd"]
        
        def energy_wrt_vsite_dist(dist_param):
            """Energy as a function of vsite distances"""
            import copy
            params = copy.deepcopy(paramset.parameters)
            params["VirtualSite"]["vsite_dist_type_2fd"] = dist_param
            return efunc(positions, box, pairs, params)
        
        # Compute gradient
        grad_fn = jax.grad(energy_wrt_vsite_dist)
        grad_dist = grad_fn(distances)
        
        print(f"   ✓ Gradients computed for {len(distances)} vsite distances")
        print(f"      Distances: {distances.flatten()}")
        print(f"      Gradients: {grad_dist.flatten()}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print("\nKey insights:")
    print("  - Virtual site weights are now differentiable parameters")
    print("  - Can optimize vsite positions for better force field accuracy")
    print("  - Different vsite types (average2, average3, 2fd, 3fd) supported")
    print("\nNext steps:")
    print("  - Use gradients for virtual site position optimization")
    print("  - Fit vsite parameters to QM data")
    print("  - Combine with charge optimization for joint parameter fitting")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
