#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <array>


namespace dpnblist {

    void read_lmp(std::string filename, std::vector<std::vector<float>> &xyz, std::array<float,3> &box_len);

    void read_ref(std::string filename, std::vector<std::vector<size_t>> &ref_listArray);
}