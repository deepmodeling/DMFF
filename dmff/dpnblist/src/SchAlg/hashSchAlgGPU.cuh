#pragma once

#include "../box.h"
#include "../nbSchAlg.h"
#include <vector>

namespace dpnblist
{
    class HashSearchGPU : public SearchAlgorithm
    {
    public:
        HashSearchGPU(const Box& box, const std::vector<std::vector<float>> &xyz, float r_cutoff);

        void search() override;
        void reset() override;

        double search_time();

        std::vector<std::vector<int>> get_particle_neighbor_info() override;

    private:
        Box box;                                      // The simulation domain
        vec3_float box_len;

        std::vector<std::vector<float>> xyz;          // Corrdinates of input particles
        int num_particles;                            // The number of particles

        float r_cutoff;                               // The cut-off radius
        float r_cutoff_2;

        float cell_size;

        int* morton_list;                             // The list of morton values - host
        int* d_morton_list;                           // The list of morton values - device

        int d_num_cells;                              // The number of cell = the max morton value

        int* new_indices;                             // The list of new indices for inputs - host
        int* d_new_indices;                           // The list of new indices for inputs - device
 
        pair2_int* compact_list;                      // The morton value: particle range - host
        pair2_int* d_compact_list;                    // The morton value: particle range - device

        std::vector<vec3_float> offset;               // The offset for searching neighbors of certain cell - host
        vec3_float* d_offset;                         // The offset for searching neighbors of certain cell - device

        int* neighbor_cells_list;                     // The list of cell neighbors - host
        int* d_neighbor_cells_list;                   // The list of cell neighbors - device

        int* particle_list;                           // The list of particle neighboring information - host
        int* d_particle_list;                         // The list of particle neighboring information - device

        int num_nb_particle;                          // The num of neighboring particle for each particle
        int min_num_nb_particle = 100;

        int search_sign = 0;                          // The sign indicating the search is conducted or not
    };
}