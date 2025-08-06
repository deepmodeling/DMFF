#include "nbList_CPU.h"
#include <random>
#include <vector>
#include <iostream>
#include <chrono>
#include <fstream>

namespace dpnblist
{
    NeighborList::NeighborList(std::string alg_type)
    :alg_type(alg_type)
    {
        if (std::find(valid_algorithm_types.begin(), valid_algorithm_types.end(), alg_type) == valid_algorithm_types.end()) {
            throw std::invalid_argument("Invalid Neighbor-Search Algorithm Type!");
        }
    }

    NeighborList::~NeighborList() 
    {
        delete s_alg; // 删除基类指针所指向的对象
    }

    void NeighborList::build(const Box& box, const std::vector<std::vector<float>>& xyz, float r_cutoff)
    {
        if (alg_type == "Hash-CPU")
        {
            s_alg = new HashSearchCPU(box, xyz, r_cutoff);

            s_alg->search();
        }else if (alg_type == "Octree-CPU")
        {
            s_alg = new OctreeSearchCPU(box, xyz, r_cutoff);

            s_alg->search();
        }else if (alg_type == "Linked_Cell-CPU")
        {
            s_alg = new CellSearchCPU(box, xyz, r_cutoff);

            s_alg->search();
        }

        search_sign = 1;
    }

    std::vector<std::vector<int>> NeighborList::get_neighbor_list()
    {   
        std::vector<std::vector<int>> res;
        if (!copy_sign)
        {
            neighbor_list = s_alg->get_particle_neighbor_info();
        }

        for (int i = 0; i < neighbor_list.size(); i++)
        {
            for (int j = 0; j < neighbor_list[i].size(); j++)
            {
                int temp_seq = neighbor_list[i][j];
                res.push_back({i, temp_seq});
            }
        }

        std::sort(res.begin(), res.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
                return a[0] < b[0];
        });

        return neighbor_list;
    }

    std::vector<int> NeighborList::get_neighbors(int seq)
    {
        std::vector<int> neighbor_particles;
        
        if (!copy_sign)
        {
            neighbor_list = s_alg->get_particle_neighbor_info();
        }

        neighbor_particles = neighbor_list[seq];

        return neighbor_particles;
    }


    pybind11::array_t<int> NeighborList::get_neighbor_pair()
    {   
        if (!copy_sign)
        {
            neighbor_list = s_alg->get_particle_neighbor_info();
        }

        int size = 0;
        for (int i = 0; i < neighbor_list.size(); i++)
        {
            size += neighbor_list[i].size();
        }

        auto res = pybind11::array_t<int>({size, 2});
        auto mutable_data = res.mutable_unchecked<2>();
        int seq = 0;
        for (int i = 0; i < neighbor_list.size(); i++)
        {
            for (int j = 0; j < neighbor_list[i].size(); j++)
            {
                int temp_seq = neighbor_list[i][j];
                mutable_data(seq, 0) = i;
                mutable_data(seq, 1) = temp_seq;

                seq++;
            }
        }

        // std::sort(res.begin(), res.end(), [](const std::vector<int>& a, const std::vector<int>& b) {
        //         return a[0] < b[0];
        // });

        return res;
    }
}



