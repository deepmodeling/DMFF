#include "../box.h"
#include "../nbSchAlg.h"

namespace dpnblist
{
    class CellSearchGPU: public SearchAlgorithm
    {
        public:
            //CellSearchGPU(Box &box, float r_cutoff, float skin);
            CellSearchGPU(const Box& box, const std::vector<std::vector<float>>& xyz, float r_cutoff);
            //void search(std::vector<std::vector<float>> &xyz) override;
            void search() override;
            void update(const std::vector<std::vector<float>> &xyz);
            void reset() override {}
            //bool judge_update(float *xyz);
            std::vector<std::vector<int>> get_particle_neighbor_info();
            // pybind11::array_t<int> get_pair();
            ~CellSearchGPU();

        private:
            /* data */
            Box _box;
            int nnebs;
            std::vector<std::vector<float>> _xyz;
            float _r_cutoff;
            float *d_r_cutoff;
            //float *d_skin;
            float _box_len[3];
            float *d_box_len;
            int _cell_len[3];
            int *d_cell_len;
            int _natoms;
            int *d_natoms;
            int _ncells;
            int *d_ncells;

            int n_nebcells;
            int *d_n_nebcells;
            std::vector<std::vector<float>> pre_xyz;
            // std::vector<std::vector<int>> off_set_vec;
            int *d_off_set_vec_1d;
            int *d_nebcell_list;

            int *d_head, *d_lscl, *d_atom_cellindex, *d_cell_atoms_count;
            int *_cell_atoms_count;
            // int *d_neighborListArray;
            int *_neighborListArray;
            int time_cost;
    };
    
}
