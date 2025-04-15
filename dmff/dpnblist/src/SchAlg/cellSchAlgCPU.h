#include "../box.h"
#include "../nbSchAlg.h"
#include <vector>

// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>
namespace dpnblist
{

    class CellList
    {
        
    public:

        CellList(const Box &box, float r_cutoff);

        int get_cell_index(std::array<int, 3>&) const;

        std::array<int, 3> get_cell_vector(int) const;

        void build(std::vector<std::array<float, 3>> &xyz);

        void update(std::vector<std::array<float, 3>> &xyz);

        void reset();

        int get_ncells() const;

        std::vector<int> get_atoms_in_cell(int cell_index) const;

        // std::vector<int> get_neighbors(int cell_index, std::vector<Vec3<int>> &shifts) const;
        void get_neighbors(int cell_index, std::array<int, 27> &neighbors, std::vector<std::array<int, 3>> &shifts);

        std::vector<int> get_head();
        std::vector<int> get_lscl();
        std::vector<int> get_count();

        // for New MIC
        std::vector<bool> boundary_cells_sign; 
    private:
        Box _box;
        std::array<int, 3> _cell_length;
        std::array<float, 3> new_cutoff;
        // std::array<std::array<int, 3>, 26> offset;
        std::vector<std::array<int, 3>> offset;
        int _natoms;
        int _ncells;
        float _r_cutoff;
        std::vector<int> _head;
        std::vector<int> _lscl;
        std::vector<int> _count;
        // const int EMPTY = std::numeric_limits<int>::max();
 
    };

    class CellSearchCPU: public SearchAlgorithm
    {
        public:
            //CellSearchCPU(Box &box, float r_cutoff, float skin);
            CellSearchCPU(const Box& box, const std::vector<std::vector<float>>& xyz, float r_cutoff);

            //void search(std::vector<std::vector<float>> &xyz) override;
            void search() override;

            // using NeighborListArray = std::vector<std::vector<int>>;

            ~CellSearchCPU();

            void reset();

            void build(std::vector<std::array<float, 3>> &xyz);

            void update(std::vector<std::array<float, 3>> &xyz);

            void get_neighbor_cell_array();

            //bool judge_update(std::vector<Vec3<float>> &xyz);

            // NeighborListArray getNeighborList() override;
            std::vector<std::vector<int>> get_particle_neighbor_info() override;
            std::array<float,3> get_min_diff_updated(const std::array<float,3>& xyz1, const std::array<float,3>& xyz2, int cell_index);

        private:
            Box _box;
            std::vector<std::vector<float>> _xyz;
            float _r_cutoff;
            int _nnebs;
            int _ncells;
            int _natoms;
            CellList _cell_list;
            std::array<float,3> cube_size;
            std::vector<std::array<float, 3>> pre_xyz;
            int *_neighlist_1d;
            std::vector<std::vector<int>> _neighborListArray;
            std::vector<std::vector<float>> _neighborDisArray;
            std::vector<int> _neighborListArray_count;
            std::vector<std::array<int, 27>> _neighbor_cell_Array;
            std::vector<std::vector<std::array<int, 3>>> _neighbor_cell_Array_shifts;
            // const int EMPTY = std::numeric_limits<int>::max();
            // for New MIC
            std::vector<bool> _boundary_cells_sign;
    };
    
}
