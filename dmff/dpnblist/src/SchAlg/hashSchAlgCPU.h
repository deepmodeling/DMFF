#pragma once

#include "../box.h"
#include "../nbSchAlg.h"

#include <vector>

namespace dpnblist
{
    class HashSearchCPU : public SearchAlgorithm
    {
    public:
        HashSearchCPU(const Box& box, const std::vector<std::vector<float>>& xyz, float r_cutoff);

        std::vector<float> get_min_diff(const std::vector<float>& xyz1, const std::vector<float>& xyz2);
        std::vector<float> get_min_diff_updated(const std::vector<float>& xyz1, const std::vector<float>& xyz2, int cell_index);

        void cal_morton_values(const std::vector<std::vector<float>>& inputs);
        void sort_by_morton();
        void cal_hash_list();
        std::vector<int> get_neighbor_cells(int cell_index);
        void get_neighbor_cells_list();

        void search() override;
        void reset() override;

        std::vector<std::vector<int>> get_particle_neighbor_info() override;

    private:
        Box box;                                                          // The simulation domain
        std::vector<std::vector<float>> xyz;                              // Corrdinates of input particles
        int num_particles;

        float r_cutoff;                                                   // The cut-off radius
        float r_cutoff_2;
        float cell_size;                                                  // The length of cells

        std::vector<float> cube_size;                                     // The size of cube domain

        std::vector<std::vector<float>> offset;                           // The offset for searching neighbors of certain cell

        std::vector<int> morton_values_list;                              // The list of morton values
        int max_morton_value;
        int num_cells;

        std::vector<int> new_indices;                                     // The list of new indices for inputs

        // std::vector<int> C_list;                                          // The compact list of "morton_values_list"

        std::vector<std::vector<int>> neighbor_cells_list;                // The list of cell neighbors

        std::vector<std::pair<int, int>> hash_list;                       // The morton value: particle range

        std::vector<int> particle_list;                                   // The list of particle neighboring information

        int num_nb_particle;                                              // The num of neighboring particle for each particle
        int min_num_nb_particle = 100;

        int search_sign = 0;                                              // The sign indicating the search is conducted or not
    
        // for New MIC
        std::vector<bool> boundary_cells_sign;  
    };
}