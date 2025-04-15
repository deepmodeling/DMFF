#pragma once

#include "box.h"
#include <stdexcept>
#include <vector>

namespace dpnblist
{
    class SearchAlgorithm
    {
    public:
        SearchAlgorithm() {};
        
        virtual void search() = 0;
        virtual void reset() = 0;
        virtual std::vector<std::vector<int>> get_particle_neighbor_info() = 0;

        void data2struc(std::vector<std::vector<float>> &xyz_cpu, vec3_float* xyz_h);

        virtual ~SearchAlgorithm() {}
    };
}