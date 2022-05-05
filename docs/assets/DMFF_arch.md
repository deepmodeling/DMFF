```mermaid
flowchart TB
E(["atomic & topological params"])
subgraph G1["Parser & Typification"]
A["force field xml"] --> B(["parseElement"])
B --> C[Generators with forcefield params]
C --> D(["createPotential"])
D --> E
end
subgraph G2["ADMP Calculators"]
E --> |init|F(["General Pairwise Calculator"])
E --> |init|G(["Multipole PME Caculator"])
E --> |init|H(["Dispersion PME Calculator"])
end
subgraph G3["Classical Calculators"]
E --> |init|K["Intramol Calculators"]
E --> |init|L["Intermol Calculators"]
end
J["pairs (neighbor list from jax-md)"] --> |input|I
F --> |return|I(["potential(pos, box, pairs, params)"])
G --> |return|I
H --> |return|I
K --> |return|I
L --> |return|I
C --> |differentiable generator.params|I
```



```mermaid
graph 
A("kernel(dr, m, p0i, p0j, p1i, p1j, ...)") --> D([generate_pairwise_interaction])
B[covalent_map] --> D
C[static variables] --> D
D --> E(["pair_energy(pos, box, pairs, p0list, p1list, ...)"])
```

```mermaid
classDiagram
	class ADMPPmeForce
	ADMPPmeForce : +axis_type
	ADMPPmeForce : +axis_indices
	ADMPPmeForce : +rc
	ADMPPmeForce : +ethresh
	ADMPPmeForce : +kappa, K1, K2, K3
	ADMPPmeForce : +pme_order
	ADMPPmeForce : +covalent_map
	ADMPPmeForce : +lpol
	ADMPPmeForce : +n_atoms
	ADMPPmeForce : +__init__(self, box, axis_type, axis_indices, covalent_map, rc, ethresh, lmax, lpol=False)
    ADMPPmeForce : +update_env(attr, val)
    ADMPPmeForce : +get_energy(pos, box, pairs, Q_local, pol, tholes, mScales, pScales, dScales)

```



``` Mermaid
classDiagram
	class ADMPDispPmeForce
	ADMPDispPmeForce : +kappa, K1, K2, K3
	ADMPDispPmeForce : +pme_order
	ADMPDispPmeForce : +rc
	ADMPDispPmeForce : +ethresh
	ADMPDispPmeForce : +pmax
	ADMPDispPmeForce : +covalent_map
	ADMPDispPmeForce : +__init__(self, box, covalent_map, rc, ethresh, pmax)
	ADMPDispPmeForce : +update_env(attr, val)
	ADMPDispPmeForce : +get_energy(positions, box, pairs, c_list, mScales)
```



