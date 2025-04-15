
#include "../src/SchAlg/cellSchAlgGPU.cuh"
#include "doctest/doctest.h"
#include "read_lmp.h"
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

namespace dpnblist {

    void nblist_test_CellGPU(std::string filename, std::string reffilename, float r_cutoff, float skin = 0.0){
        std::vector<std::vector<float>> xyz;
        std::array<float,3> box_len;
        read_lmp(filename, xyz, box_len);
        
        vec3_float box_length = {box_len[0], box_len[1], box_len[2]};
        vec3_float angles = {90, 90, 90};
        Box box(box_length, angles);

        CellSearchGPU s_alg(box, xyz, r_cutoff);
        s_alg.search();

        std::vector<std::vector<int>> listArray = s_alg.get_particle_neighbor_info();
        // Open a file ref file
        std::vector<std::vector<size_t>> ref_listArray(listArray.size());
        read_ref(reffilename, ref_listArray);
        CHECK_EQ(listArray.size(), ref_listArray.size());
        for (int i = 0; i < listArray.size(); ++i) {
            CHECK_EQ(listArray[i].size(), ref_listArray[i].size());
            if (listArray[i].size() != ref_listArray[i].size()) {
                std::cout << filename << " line wrong: " << i+1 << std::endl;
            }
        }
    }

    TEST_SUITE("Test Ref Cell List")
    {

        // TEST_CASE("CellListInitialization") {
        //     dpnblist::Box box({10, 10, 10}, {90, 90, 90});
        //     dpnblist::CellList cellList(&box, 2.0);

        //     // 测试初始化状态
        //     Vec3<int> cell_111(1, 1, 1);
        //     CHECK_EQ(cellList.get_cell_index(cell_111), 31);
        //     CHECK_EQ(cellList.get_cell_vector(31), cell_111);
        //     CHECK_EQ(cellList.get_ncells(), 125);
        // }

        // TEST_CASE("CellListBuildAndReset") {
        //     dpnblist::Box box({10, 10, 10}, {90, 90, 90});
        //     dpnblist::CellList cellList(&box, 2.0);

        //     std::vector<dpnblist::Vec3<double>> positions = {
        //         {0, 0, 0}, {1, 1, 1}, {2, 2, 2}, {4, 4, 4}, {10, 10, 10}};

        //     cellList.build(positions);

        //     // 测试建立后的状态
        //     CHECK_EQ(cellList.get_atoms_in_cell(0).size(), 2);
        //     CHECK_EQ(cellList.get_atoms_in_cell(31).size(), 1);
        //     CHECK_EQ(cellList.get_neighbors(31).size(), 27);
        //     CHECK_EQ(cellList.get_neighbors(31), std::vector<size_t>({0,1,2,5,6,7,10,11,12,25,26,27,30,32,35,36,37,50,51,52,55,56,57,60,61,62,31}));
        //     CHECK_EQ(cellList.get_neighbors(0), std::vector<size_t>({124,120,121,104,100,101,109,105,106, 24,20,21,4,1,9,5,6, 49,45,46,29,25,26,34,30,31,0}));
        // }

        
        TEST_CASE("Cell List on GPU"){
        // test1
            nblist_test_CellGPU("heterogeneous.lmp", "sort_output_heterogeneous.txt", 3.0, 0.0);

        // // test2
        //     // nblist_test("homogeneous_sparse.lmp", "sort_output_homogeneous_sparse.txt", 3.0, 0.0);

        // // test3
        //     nblist_test("homogeneous_dense.lmp", "sort_output_homogeneous_dense.txt", 3.0, 0.0);

        // // test4
        //     nblist_test("Nb_water.lmp", "sort_output_Nb_water.txt", 3.6, 0.0);

        // // test5
        //     nblist_test("protein.lmp", "sort_output_protein.txt", 3.6, 0.0);

        }

    }

}
