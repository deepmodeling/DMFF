#include "box.h"
#include "nbSchAlg.h"
#include <stdexcept>
#include <vector>

namespace dpnblist
{   
    ///////////////////////////////////////////////////////////////////////////
    // CPU parts
    void SearchAlgorithm::data2struc(std::vector<std::vector<float>> &xyz_cpu, vec3_float* xyz_h)
    {
        if (xyz_cpu.empty()) {
            throw std::invalid_argument("Empty Input Data!");
        }

        // 将数据复制到连续内存块中
        for (int i = 0; i < xyz_cpu.size(); i++)
        {
            xyz_h[i].x = xyz_cpu[i][0];
            xyz_h[i].y = xyz_cpu[i][1];
            xyz_h[i].z = xyz_cpu[i][2];
        }
    }


    // ///////////////////////////////////////////////////////////////////////////
    // // CPU parts
    // SearchAlgorithmCPU::SearchAlgorithmCPU(const Box& box, float r_cutoff)
    // : _box(box), _r_cutoff(r_cutoff)
    // {
    //     r_cutoff_2 = r_cutoff * r_cutoff;

    //     offset = {
    //         {-_rc,-_rc, -_rc}, {-_rc,-_rc, 0}, {-_rc,-_rc, _rc},
    //         {-_rc,   0, -_rc}, {-_rc,   0, 0}, {-_rc,   0, _rc},
    //         {-_rc, _rc, -_rc}, {-_rc, _rc, 0}, {-_rc, _rc, _rc},
    //         {   0,-_rc, -_rc}, {   0,-_rc, 0}, {   0,-_rc, _rc},
    //         {   0,   0, -_rc}, {   0,   0, 0}, {   0,   0, _rc},
    //         {   0, _rc, -_rc}, {   0, _rc, 0}, {   0, _rc, _rc},
    //         { _rc,-_rc, -_rc}, { _rc,-_rc, 0}, { _rc,-_rc, _rc},
    //         { _rc,   0, -_rc}, { _rc,   0, 0}, { _rc,   0, _rc},
    //         { _rc, _rc, -_rc}, { _rc, _rc, 0}, { _rc, _rc, _rc}
    //     };
    // }


    // ///////////////////////////////////////////////////////////////////////////
    // // GPU parts
    // SearchAlgorithmGPU::SearchAlgorithmGPU(const Box& box, float r_cutoff)
    // : _box(box), _r_cutoff(r_cutoff)
    // {
    //     r_cutoff_2 = r_cutoff * r_cutoff;

    //     box_len = _box.get_angles_gpu();
    // }

    // void SearchAlgorithmGPU::data2struc(std::vector<std::vector<float>> &xyz_cpu, vec3_float* xyz_h)
    // {
    //     if (xyz_cpu.empty()) {
    //         return nullptr; // 如果输入为空，则返回空指针
    //     }

    //     // 将数据复制到连续内存块中
    //     for (int i = 0; i < xyz_cpu.size(); i++)
    //     {
    //         xyz_h[i].x = xyz_cpu[i][0];
    //         xyz_h[i].y = xyz_cpu[i][1];
    //         xyz_h[i].z = xyz_cpu[i][2];
    //     }
    // }
}