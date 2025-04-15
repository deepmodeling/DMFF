#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include "octreeSchAlgCPU.h"

// #include "pybind11/pybind11.h"
// #include "pybind11/stl.h"

namespace dpnblist {

    OctreeNBL::OctreeNBL(std::array<float,3> center, std::array<float,3> box_len) : _center(center), _box_len(box_len) {
    }

    void OctreeNBL::insert(std::array<float,3> position, int pos_index, OctreeNode *node) {
        if (node->positions.size() < 8) {
            node->positions.push_back(position);
            node->pos_index.push_back(pos_index);
        } else {
            int is_right[3];
            for (int i = 0; i < 3; ++i) {
                if (position[i] > node->center[i]) {
                    is_right[i] = 1;
                } else {
                    is_right[i] = 0;
                }
            }
            int indices = is_right[0] + is_right[1] * 2 + is_right[2] * 4;
            if (node->children[indices] == nullptr) {
                // std::array<float,3> child_center = node->center + std::array<float,3>((is_right[0] - 0.5) * node->size[0] / 2, (is_right[1] - 0.5) * node->size[1] / 2, (is_right[2] - 0.5) * node->size[2] / 2);
                std::array<float,3> node_center = node->center;
                std::array<float,3> child_center;
                for (int i = 0; i < 3; ++i) {
                    child_center[i] = node_center[i] + (is_right[i] - 0.5) * node->size[i] / 2;
                }
                std::array<float,3> half_node_size = {node->size[0] / 2, node->size[1] / 2, node->size[2] / 2};
                node->children[indices] = new OctreeNode(child_center, half_node_size);
            }
            insert(position, pos_index, node->children[indices]);
        }
    }

    OctreeNode* OctreeNBL::constructor(const std::vector<std::array<float,3>>& positions, std::vector<int> &pos_index) {

        OctreeNode* root = new OctreeNode(_center, _box_len);
        for (int i = 0; i < positions.size(); ++i) {
            insert(positions[i], pos_index[i], root);
        }

        return root;
    }

    std::array<int,3> OctreeSearchCPU::aa(std::vector<std::array<int,3>> &a, int l, int r){
        std::array<int,3> tmp = {0,0,0};
        // tmp = a[l] + a[r];
        for (int i = 0; i < 3; i++){
            tmp[i] = a[l][i] + a[r][i];
        }
        return tmp;
    }

    std::array<int,3> OctreeSearchCPU::aaa(std::vector<std::array<int,3>> &a, int l, int r, int c){
        std::array<int,3> tmp = {0,0,0};
        // tmp = a[l] + a[r] + a[c];
        for (int i = 0; i < 3; i++){
            tmp[i] = a[l][i] + a[r][i] + a[c][i];
        }
        return tmp;
    }

    std::vector<std::array<float,3>> OctreeSearchCPU::period(const std::array<float,3> posi, float radius) {
        std::array<float,3> min = {0,0,0};
        std::array<float,3> max = _box_len;
        std::vector<std::array<int,3>> a;
        for (int i = 0; i < 3; ++i) {
            std::array<int,3> tmp = {0,0,0};
            if (posi[i] - min[i] <= radius){
                tmp[i] = -1;
                a.push_back(tmp);
            }
            else if (max[i] - posi[i] <= radius){
                tmp[i] = 1;
                a.push_back(tmp);
            }
        }
        if (a.size() == 2){
            a.push_back(aa(a, 0, 1));
        }
        else if (a.size() == 3){
            a.push_back(aa(a, 0, 1));
            a.push_back(aa(a, 0, 2));
            a.push_back(aa(a, 1, 2));
            a.push_back(aaa(a, 0, 1, 2));
        }
        std::vector<std::array<float,3>> posis = {posi};
        
        for (auto ai : a) {
            // std::array<float,3> posi_cp = posi - ai * (max - min);
            std::array<float,3> posi_cp;
            for (int i = 0; i < 3; ++i) {
                posi_cp[i] = posi[i] - ai[i] * (max[i] - min[i]);
            }
            posis.push_back(posi_cp);
        }
        
        return posis;
    }

    void OctreeSearchCPU::build(OctreeNode *rnode, const std::vector<std::array<float,3>>& positions, float distance) {
        // 构建neighborlist
        particles.clear();
        particles.resize(positions.size());
        float lo[3], hi[3];
        #pragma omp parallel for
        for (int i = 0; i < positions.size(); ++i) {
            std::vector<std::array<float,3>> posis = period(positions[i], distance);
            
            std::vector<int> nb_particles_temp;
            for (auto posi : posis) {
                std::vector<int> nb_particles_tempi = query(posi, distance, rnode);
                nb_particles_temp.insert(nb_particles_temp.end(), nb_particles_tempi.begin(), nb_particles_tempi.end());
            }

            particles[i] = nb_particles_temp;
        }
    }

    std::vector<int> OctreeSearchCPU::query(const std::array<float,3>& position, float distance, OctreeNode *node) {
        std::vector<int> nearby_positions;
        // AABB接触法，判断是否进行查询
        std::array<float,3> min_bound = {node->center[0] - node->size[0] / 2, node->center[1] - node->size[1] / 2, node->center[2] - node->size[2] / 2};
        std::array<float,3> max_bound = {node->center[0] + node->size[0] / 2, node->center[1] + node->size[1] / 2, node->center[2] + node->size[2] / 2};

        std::array<float,3> min_bound_distance;
        std::array<float,3> max_bound_distance;
        for (int i = 0; i < 3; ++i) {
            min_bound_distance[i] = min_bound[i] - (position[i] + distance);
            max_bound_distance[i] = (position[i] - distance) - max_bound[i];
        }

        // 检查查询范围是否与格子边界有交集
        bool intersects = (min_bound[0] <= position[0] + distance) && (max_bound[0] >= position[0] - distance)
                        && (min_bound[1] <= position[1] + distance) && (max_bound[1] >= position[1] - distance)
                        && (min_bound[2] <= position[2] + distance) && (max_bound[2] >= position[2] - distance);
        float rc2 = distance * distance;
        if (intersects) {
            for (int i = 0; i < node->positions.size(); ++i) {
                // std::array<float,3> dpos = node->positions[i] - position;
                std::array<float,3> dpos;
                for (int j = 0; j < 3; ++j) {
                    dpos[j] = node->positions[i][j] - position[j];
                }
                float dis2 = dpos[0] * dpos[0] + dpos[1] * dpos[1] + dpos[2] * dpos[2];
                
                if ((dis2 - rc2) <= 1e-7) {
                    nearby_positions.push_back(node->pos_index[i]);
                }
            }
            for (OctreeNode* child : node->children) {
                if (child != nullptr) {
                    std::vector<int> child_nearby_positions = query(position, distance, child);
                    nearby_positions.insert(nearby_positions.end(), child_nearby_positions.begin(), child_nearby_positions.end());
                }
            }
        }

        return nearby_positions;
    }
    

    OctreeSearchCPU::OctreeSearchCPU(const Box& box, const std::vector<std::vector<float>>& xyz, float r_cutoff) : _box(box), _xyz(xyz), _r_cutoff(r_cutoff) {
        _box_len = box.get_lengths_cpu3();
    }

    void OctreeSearchCPU::search() {
        std::vector<std::array<float,3>> xyz;
        xyz.resize(_xyz.size());
        for (int i = 0; i < _xyz.size(); ++i) {
            xyz[i][0] = _xyz[i][0];
            xyz[i][1] = _xyz[i][1];
            xyz[i][2] = _xyz[i][2];
        }
        std::vector<std::vector<std::array<float,3>>> xyz8;
        xyz8.resize(8);
        std::vector<std::vector<int>> xyz8_index;
        xyz8_index.resize(8);
        for (int i = 0; i < xyz.size(); i++){
            std::array<int,3> temp = {0, 0, 0};
            for (int j = 0; j < 3; j++){
                if (xyz[i][j] - _box_len[j] / 2 < 0){
                    temp[j] = 0;
                }
                else{
                    temp[j] = 1;
                }
            }
            int index = temp[0] * 1 + temp[1] * 2 + temp[2] * 4;
            xyz8[index].emplace_back(xyz[i]);
            xyz8_index[index].emplace_back(i);
        }
        OctreeNBL octree8[8];
        OctreeNode *root8[8];

        std::vector<std::array<float,3>> center8 = {
            {_box_len[0] / 4, _box_len[1] / 4, _box_len[2] / 4},
            {3 * _box_len[0] / 4, _box_len[1] / 4, _box_len[2] / 4},
            {_box_len[0] / 4, 3 * _box_len[1] / 4, _box_len[2] / 4},
            {3 * _box_len[0] / 4, 3 * _box_len[1] / 4, _box_len[2] / 4},
            {_box_len[0] / 4, _box_len[1] / 4, 3 * _box_len[2] / 4},
            {3 * _box_len[0] / 4, _box_len[1] / 4, 3 * _box_len[2] / 4},
            {_box_len[0] / 4, 3 * _box_len[1] / 4, 3 * _box_len[2] / 4},
            {3 * _box_len[0] / 4, 3 * _box_len[1] / 4, 3 * _box_len[2] / 4}
        };    
        auto start_time = std::chrono::high_resolution_clock::now();

        std::array<float,3> half_box_len = {_box_len[0] / 2, _box_len[1] / 2, _box_len[2] / 2};
        // #pragma omp parallel for num_threads(8)
        for (int i = 0; i < 8; i++){
            int thread_id = omp_get_thread_num();
            // std::cout << "thread_id: " << thread_id << std::endl;
            if (xyz8[i].size() != 0){
                octree8[i] = OctreeNBL(center8[i], half_box_len);
                root8[i] = octree8[i].constructor(xyz8[i], xyz8_index[i]);
            }
        }
        OctreeNode *root = new OctreeNode(half_box_len, _box_len);
        for (int i = 0; i < 8; i++){
            if (xyz8[i].size() != 0){
                root->children[i] = root8[i];
            }
        }
        auto construct_time = std::chrono::high_resolution_clock::now();
        auto construct_duration = std::chrono::duration_cast<std::chrono::milliseconds>(construct_time - start_time);
        //std::cout << "parallel constructor time: " << construct_duration.count() << " milliseconds" << std::endl;
        build(root, xyz, _r_cutoff);
        auto search_time = std::chrono::high_resolution_clock::now();
        auto search_duration = std::chrono::duration_cast<std::chrono::milliseconds>(search_time - construct_time);
        //std::cout << "parallel search time: " << search_duration.count() << " milliseconds" << std::endl;

        //删除内存
        delete root;

    }

    std::vector<std::vector<int>> OctreeSearchCPU::get_particle_neighbor_info() {
        std::vector<std::vector<int>> particles2;
        particles2.resize(particles.size());
        for (int i = 0; i < particles.size(); ++i) {
            for (int j = 0; j < particles[i].size(); ++j) {
                if (particles[i][j] != i) {
                    particles2[i].push_back(particles[i][j]);
                }
            }
        }
        return particles2;
    }
}

