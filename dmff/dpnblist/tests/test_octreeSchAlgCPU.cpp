
#include "../src/SchAlg/octreeSchAlgCPU.h"
#include "doctest/doctest.h"
#include "read_lmp.h"
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>

namespace dpnblist {

    void nblist_test_OctreeCPU(std::string filename, std::string reffilename, float r_cutoff, float skin = 0.0){
        std::vector<std::vector<float>> xyz;
        std::array<float,3> box_len;
        read_lmp(filename, xyz, box_len);
        
        vec3_float box_length = {box_len[0], box_len[1], box_len[2]};
        vec3_float angles = {90, 90, 90};
        Box box(box_length, angles);

        OctreeSearchCPU s_alg(box, xyz, r_cutoff);
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

    TEST_SUITE("Test Ref Octree List")
    {

        
        TEST_CASE("Octree List on CPU"){
        // test1
            nblist_test_OctreeCPU("heterogeneous.lmp", "sort_output_heterogeneous.txt", 3.0, 0.0);

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
