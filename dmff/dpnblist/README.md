# dpnblist
 Resable and Modular Neighborlist Lib

### Installation
follow the below commands to build the project
```
pip install .
```
then a dpnblist package will be compiled and installed into the python environment.

### Usage
In our neighbor list library, we provide a full type of neighbor list and include three different types of algorithms: cell list, octree, and hash methods, and support CPU and GPU.  

1. First, it is necessary to define a simulation system through the `box` class, mainly including two parameters, the `length` and `angle` of the simulation system, as shown in the table below:

    |Box Parameter | Data Type |
    |---|---|
    |length  |  list |
    |amgle   |   list |

2. Then, specify different algorithms and platforms through the `NeighborList` class, with parameters as follows

    |Parameter| Data Type |Value|
    |---|---|---|
    |mode|string|"Linked_Cell-CPU"  "Linked_Cell-GPU"  "Octree-CPU"  "Octree-GPU"  "Hash-CPU"  "Hash-GPU"|

3. Finally, call the `build` method to generate the neighbor list, passing the `box`, `points`, and `cutoff` parameters. Then, use the `get_neighbor_pair` method to retrieve the neighbor list.

Here's an example of usage:  
```python
import dpnblist
import numpy as np

dpnblist.set_num_threads(4)
print(dpnblist.get_max_threads())
num_particle = 30000
shape = (num_particle, 3)
points = np.random.random(shape) * domain_size
domain_size = [50.0, 50.0, 50.0]
angle = [90.0, 90.0, 90.0]
cutoff = 6.0

box = dpnblist.Box(domain_size, angle)
nb = dpnblist.NeighborList("Linked_Cell-GPU")    # Linked_Cell-GPU  Linked_Cell-CPU  Octree-GPU  Octree-CPU  Hash-GPU  Hash-CPU
nb.build(box, points, cutoff)
pairs = nb.get_neighbor_pair()
```