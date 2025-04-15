#pragma once

#include <vector>
#include "box.h"
#include "nbSchAlg.h"
#include "SchAlg/hashSchAlgGPU.cuh"
#include "SchAlg/hashSchAlgCPU.h"
#include "SchAlg/octreeSchAlgGPU.cuh"
#include "SchAlg/octreeSchAlgCPU.h"
#include "SchAlg/cellSchAlgGPU.cuh"
#include "SchAlg/cellSchAlgCPU.h"
#include <algorithm>
#include <iostream>
#include <pybind11/numpy.h>

namespace dpnblist
{
    class NeighborList
    {
    public:
        NeighborList(std::string alg_type);

        ~NeighborList();

        void build(const Box& box, const std::vector<std::vector<float>>& xyz, float r_cutoff);
        void update();
        void reset();

        std::vector<int> get_neighbors(int seq);
        std::vector<std::vector<int>> get_neighbor_list();
        // std::vector<std::vector<int>> get_neighbor_pair();
        // std::vector<std::array<int,2>> get_neighbor_pair();
        // std::array<int,2>* get_neighbor_pair();
        pybind11::array_t<int> get_neighbor_pair();

    private:
        std::string alg_type;                             // The algorithm type

        // Valid algorithm types 
        const std::vector<std::string> valid_algorithm_types = {
                                                                "Linked_Cell-CPU", "Linked_Cell-GPU",
                                                                "Octree-CPU", "Octree-GPU",
                                                                "Hash-CPU", "Hash-GPU"
                                                            };

        SearchAlgorithm* s_alg;                           // Pointer to the base class for search algorithms

        Box box;                                          // The simulation domain
        std::vector<std::vector<float>> xyz;              // Coordinates of input particles coordinates

        std::vector<std::vector<int>> neighbor_list;      // The list for neighboring information

        bool search_sign = 0;                             // The sign indicating the search is conducted or not
        bool copy_sign = 0;                               // The sign indicating the copy is conducted or not
    };
}
