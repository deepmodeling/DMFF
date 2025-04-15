#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
#include "octreeSchAlgGPU.cuh"
#include <omp.h>


// using namespace dpnblist;
namespace dpnblist
{
     
    OctreeNBLGPU::OctreeNBLGPU(Vecc3<float> center, Vecc3<float> box_len) : center(center), size(box_len){
        
    }

    void OctreeNBLGPU::insert(Vecc3<float> position, int pos_index, int node_index, std::vector<NodeGPU> &nodelist, int threadid) {
        if (node_index == -1) {
            return;
        }
        int count = 0;
        for (int i = 0; i < 4; ++i) {
            if (nodelist[node_index].pos_index[i] != -1) {
                count++;
            }
        }
        if (count < 4) {
            nodelist[node_index].pos_index[count] = pos_index;
        } else {
            // int is_right[3];
            Vecc3<float> is_right(0.0, 0.0, 0.0);
            for (int i = 0; i < 3; ++i) {
                if (position[i] > nodelist[node_index].center[i]) {
                    is_right[i] = 1.0f;
                } else {
                    is_right[i] = 0.0f;
                }
            }
            int indices = is_right[0] + is_right[1] * 2 + is_right[2] * 4;
            if (nodelist[node_index].children[indices] == -1) {
                // Vecc3<float> child_center = nodelist[node_index].center + Vecc3<float>((is_right[0] - 0.5) * nodelist[node_index].size[0] / 2, (is_right[1] - 0.5) * nodelist[node_index].size[1] / 2, (is_right[2] - 0.5) * nodelist[node_index].size[2] / 2);
                Vecc3<float> child_center = nodelist[node_index].center + (is_right - Vecc3<float>(0.5, 0.5, 0.5)) * nodelist[node_index].size / 2;
                nodelist.push_back(NodeGPU(child_center, nodelist[node_index].size / 2, threadid));
                nodelist[node_index].children[indices] = nodelist.size() - 1;
            }
            insert(position, pos_index, nodelist[node_index].children[indices], nodelist, threadid);
        }
    }
    std::vector<NodeGPU> OctreeNBLGPU::build_tree(std::vector<Vecc3<float>>& positions, std::vector<int> &pos_index, int threadid) {
        NodeGPU *root = new NodeGPU(center, size, threadid);
        std::vector<NodeGPU> nodes;
        nodes.push_back(*root);
        for(int i = 0; i < positions.size(); ++i){
            insert(positions[i], pos_index[i], 0, nodes, threadid);
        }

        return nodes;
    }

    __global__ void initializeArrayO(int *array, int size) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < size) {
            array[tid] = -1;
        }
    }

    __device__ Vecc3<int> aa(Vecc3<int>* a3, int l, int r) {
            Vecc3<int> tmp(0,0,0);
            tmp = a3[l] + a3[r];
            return tmp;
    }

    __device__ Vecc3<int> aaa(Vecc3<int>* a3, int l, int r, int s) {
            Vecc3<int> tmp(0,0,0);
            tmp = a3[l] + a3[r] + a3[s];
            return tmp;
    }

    __device__ void period(Vecc3<float>* posi, Vecc3<int>* shift, int num, int* flag) {
        if (num == 1){
            for (int i = 0; i < 3; ++i) {
                if (flag[i] != 0) {
                    shift[1][i] = flag[i];
                }
            }
        }
        else if (num == 2){
            Vecc3<int> a3[2];
            int count = 0;
            for (int i = 0; i < 3; ++i) {
                if (flag[i] != 0) {
                    a3[count][i] = flag[i];
                    shift[count+1][i] = flag[i];
                    count++;
                }
            }
            shift[3] = aa(a3, 0, 1);
        }
        else if (num == 3){
            Vecc3<int> a3[3];
            for (int i = 0; i < 3; ++i) {
                a3[i][i] = flag[i];
                shift[i+1][i] = flag[i];
            }
            shift[4] = aa(a3, 0, 1);
            shift[5] = aa(a3, 0, 2);
            shift[6] = aa(a3, 1, 2);
            shift[7] = aaa(a3, 0, 1, 2);
        }

    }

    __device__ int getneb(NodeGPU *nodes, int idx, Vecc3<float> posai, Vecc3<float>* positions, int* neblist, int num, float cut, int* nodes_shift, int d_nnebs) {

        int stack[500];
        int top = -1;
        stack[++top] = 0;
        float cut2 = cut * cut;

        Vecc3<float> min_bound;
        Vecc3<float> max_bound;
        Vecc3<float> max_bound_distance = posai + Vecc3<float>(cut, cut, cut);
        Vecc3<float> min_bound_distance = posai - Vecc3<float>(cut, cut, cut);
        NodeGPU* node;
        Vecc3<float> dpos;
        while (top >= 0){
            node = &nodes[stack[top--]];

            min_bound = node->center - node->size / 2;
            max_bound = node->center + node->size / 2;

            bool intersects = (min_bound[0] <= max_bound_distance[0]) && (max_bound[0] >= min_bound_distance[0])
                            && (min_bound[1] <= max_bound_distance[1]) && (max_bound[1] >= min_bound_distance[1])
                            && (min_bound[2] <= max_bound_distance[2]) && (max_bound[2] >= min_bound_distance[2]);
            if (intersects) {
                for (int i = 0; i < 4; ++i) {
                    if (node->pos_index[i] != -1) {
                        dpos = positions[node->pos_index[i]] - posai;
                        float dis = dpos[0]*dpos[0]+ dpos[1]*dpos[1] + dpos[2]*dpos[2];
                        if ((dis - cut2) <= 1e-7) {
                            neblist[idx * d_nnebs + num] = node->pos_index[i];
                            num++;
                        }
                    }
                }
                for (int i = 0; i < 8; ++i) {
                    if (node->children[i] != -1) {
                        stack[++top] = node->children[i] + nodes_shift[node->threadid];
                    }
                }
            }
        }
        return num;
    }

    __global__ void build(NodeGPU* nodes, Vecc3<float>* pos, float* dcut, int* neblist, Vecc3<float>* dmin, Vecc3<float>* dmax, int* nodes_shift, int d_nnebs, int size){
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        
        if (idx < size) {
            float cut = *dcut;
            Vecc3<float> min = *dmin;
            Vecc3<float> max = *dmax;

            int count = 0;
            int flag[3] = {0, 0, 0};

            for (int i = 0; i < 3; ++i) {
                if (pos[idx][i] - min[i] <= cut) {
                    flag[i] = -1;
                    count++;
                } else if (max[i] - pos[idx][i] <= cut) {
                    flag[i] = 1;
                    count++;
                }
            }

            if (count == 0) {
                // printf("pos %d count == 0 flag: %d, %d, %d\n", idx, flag[0], flag[1], flag[2]);
                int num = 0;
                num = getneb(nodes, idx, pos[idx], pos, neblist, num, cut, nodes_shift, d_nnebs);
                // printf("num: %d\n", num);
            }
            else if (count == 1) {
                // printf("pos %d count == 1 flag: %d, %d, %d\n", idx, flag[0], flag[1], flag[2]);
                Vecc3<int> shift[2];
                period(&pos[idx], shift, 1, flag);
                int num = 0;
                for (int i = 0; i < 2; ++i) {
                    Vecc3<float> posai = pos[idx] - (max - min) * shift[i];
                    num = getneb(nodes, idx, posai, pos, neblist, num, cut, nodes_shift, d_nnebs);
                    // printf("num: %d\n", num);
                }
            }
            else if (count == 2) {
                // printf("pos %d count == 2 flag: %d, %d, %d\n", idx, flag[0], flag[1], flag[2]);
                Vecc3<int> shift[4];
                period(&pos[idx], shift, 2, flag);
                int num = 0;
                for (int i = 0; i < 4; ++i) {
                    Vecc3<float> posai = pos[idx] - (max - min) * shift[i];
                    num = getneb(nodes, idx, posai, pos, neblist, num, cut, nodes_shift, d_nnebs);
                    // printf("num: %d\n", num);
                }
            }
            else if (count == 3) {
                Vecc3<int> shift[8];
                period(&pos[idx], shift, 3, flag);
                int num = 0;
                for (int i = 0; i < 8; ++i) {
                    Vecc3<float> posai = pos[idx] - (max - min) * shift[i];
                    num = getneb(nodes, idx, posai, pos, neblist, num, cut, nodes_shift, d_nnebs);
                    // printf("num: %d\n", num);
                }
            }
        }
    }

    OctreeSearchGPU::OctreeSearchGPU(const Box& box, const std::vector<std::vector<float>>& xyz, float r_cutoff) : _box(box), _xyz(xyz), _r_cutoff(r_cutoff){
        // nnebs = 500;
        std::array<float,3> cube_size = box.get_lengths_cpu3();
        // float volume = box_length[0] * box_length[1] * box_length[2];
        // float pre_nnebs = xyz.size() / volume * 4 * 3.1415926 * r_cutoff * r_cutoff * r_cutoff / 3;
        // nnebs = int(pre_nnebs * 1.11 + 61);
        nnebs = static_cast<int>(std::ceil(1.5 * (5 * xyz.size() / (cube_size[0] * cube_size[1] * cube_size[2])) * r_cutoff * r_cutoff * r_cutoff + 45));
        if (nnebs < 100) nnebs = 100;
    }

    //void OctreeSearchGPU::search(std::vector<std::vector<float>> &_xyz) {
    void OctreeSearchGPU::search() {
        Vecc3<float> _box_len(_box.get_lengths_cpu3()[0], _box.get_lengths_cpu3()[1], _box.get_lengths_cpu3()[2]);
        Vecc3<float> half_box_len(_box_len[0] / 2, _box_len[1] / 2, _box_len[2] / 2);
        Vecc3<float> center[8] = {
            Vecc3<float>(_box_len[0] / 4, _box_len[1] / 4, _box_len[2] / 4),
            Vecc3<float>(3 * _box_len[0] / 4, _box_len[1] / 4, _box_len[2] / 4),
            Vecc3<float>(_box_len[0] / 4, 3 * _box_len[1] / 4, _box_len[2] / 4),
            Vecc3<float>(3 * _box_len[0] / 4, 3 * _box_len[1] / 4, _box_len[2] / 4),
            Vecc3<float>(_box_len[0] / 4, _box_len[1] / 4, 3 * _box_len[2] / 4),
            Vecc3<float>(3 * _box_len[0] / 4, _box_len[1] / 4, 3 * _box_len[2] / 4),
            Vecc3<float>(_box_len[0] / 4, 3 * _box_len[1] / 4, 3 * _box_len[2] / 4),
            Vecc3<float>(3 * _box_len[0] / 4, 3 * _box_len[1] / 4, 3 * _box_len[2] / 4)
        };
        _natoms = _xyz.size();
        std::vector<Vecc3<float>> xyz;
        xyz.resize(_xyz.size());

        // std::vector<std::vector<Vecc3<float>>> xyz8;
        // xyz8.resize(8);
        // std::vector<std::vector<int>> xyz8_index;
        // xyz8_index.resize(8);
        std::vector<Vecc3<float>> xyz8[8];
        std::vector<int> xyz8_index[8];
        Vecc3<int> temp(0,0,0);

        for (int i = 0; i < _natoms; i++){
            xyz[i][0] = _xyz[i][0];
            xyz[i][1] = _xyz[i][1];
            xyz[i][2] = _xyz[i][2];
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
        OctreeNBLGPU octree8[8];
        // std::cout << "init ok!" << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<NodeGPU>> nodes8;
        nodes8.resize(8);
        #pragma omp parallel for num_threads(8)
        for (int i = 0; i < 8; i++){
            int thread_id = omp_get_thread_num();
            // std::cout << "thread_id: " << thread_id << std::endl;
            octree8[thread_id] = OctreeNBLGPU(center[thread_id], half_box_len);
            nodes8[thread_id] = octree8[thread_id].build_tree(xyz8[thread_id], xyz8_index[thread_id], thread_id);
        }
        #pragma omp barrier

        std::vector<NodeGPU> nodes;
        NodeGPU root(half_box_len, _box_len, 0);
        nodes.push_back(root);
        int nodes_shift[8];
        int shift = 1;
        for (int i = 0; i < 8; i++){
            nodes_shift[i] = shift;
            root.children[i] = shift - 1;
            nodes.insert(nodes.end(), nodes8[i].begin(), nodes8[i].end());
            shift += nodes8[i].size();
        }
        nodes[0] = root;

        auto end_time1 = std::chrono::high_resolution_clock::now();
        auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time1-start_time);
        //std::cout << "build tree time: " << duration1.count() << "ms" << std::endl;
        
        NodeGPU* d_nodes;
        cudaMalloc((void**)&d_nodes, nodes.size() * sizeof(NodeGPU));
        cudaMemcpy(d_nodes, nodes.data(), nodes.size() * sizeof(NodeGPU), cudaMemcpyHostToDevice);

        int* d_nodes_shift;
        cudaMalloc((void**)&d_nodes_shift, 8 * sizeof(int));
        cudaMemcpy(d_nodes_shift, nodes_shift, 8 * sizeof(int), cudaMemcpyHostToDevice);

        auto end_time2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time2-end_time1);
        //std::cout << "copy node data to device: " << duration2.count() << "ms" << std::endl;

        Vecc3<float>* d_pos;
        cudaMalloc((void**)&d_pos, _natoms * sizeof(Vecc3<float>));
        cudaMemcpy(d_pos, xyz.data(), _natoms * sizeof(Vecc3<float>), cudaMemcpyHostToDevice);

        listarray = new int[_natoms * nnebs];
        
        // for (int i = 0; i < _natoms * nnebs; ++i) {
        //     listarray[i] = -1;
        // }
        // int* d_listarray;
        // cudaMalloc((void**)&d_listarray, nnebs * sizeof(int) * _natoms);
        // cudaMemcpy(d_listarray, listarray, nnebs * sizeof(int) * _natoms, cudaMemcpyHostToDevice);

        int arraySize = _natoms * nnebs;
        int *d_listarray;
        cudaMalloc((void**)&d_listarray, arraySize * sizeof(int));
        
        int threadsPerBlock = 256;
        int blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock;
        // Launch kernel to initialize the array
        initializeArrayO<<<blocksPerGrid, threadsPerBlock>>>(d_listarray, arraySize);
        cudaDeviceSynchronize(); // Ensure kernel execution is complete before proceeding

        // Vecc3<float>* min = new Vecc3<float>(0.0, 0.0, 0.0);
        // Vecc3<float>* max = new Vecc3<float>(_box_len[0], _box_len[1], _box_len[2]);
        Vecc3<float> min(0.0, 0.0, 0.0), max(_box_len[0], _box_len[1], _box_len[2]);
        Vecc3<float>* d_min, *d_max;
        cudaMalloc((void**)&d_min, sizeof(Vecc3<float>));
        cudaMemcpy(d_min, &min, sizeof(Vecc3<float>), cudaMemcpyHostToDevice);
        cudaMalloc((void**)&d_max, sizeof(Vecc3<float>));
        cudaMemcpy(d_max, &max, sizeof(Vecc3<float>), cudaMemcpyHostToDevice);
        float *d_cut;
        cudaMalloc((void**)&d_cut, sizeof(float));
        cudaMemcpy(d_cut, &_r_cutoff, sizeof(float), cudaMemcpyHostToDevice);

        auto end_time22 = std::chrono::high_resolution_clock::now();
        // int blocksize = 256;
        blocksPerGrid = (_natoms + threadsPerBlock - 1) / threadsPerBlock;
        build<<<blocksPerGrid, threadsPerBlock>>>(d_nodes, d_pos, d_cut, d_listarray, d_min, d_max, d_nodes_shift, nnebs, _natoms);
        cudaDeviceSynchronize();

        auto end_time3 = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time3-end_time22);
        //std::cout << "build nblist time: " << duration3.count() << "ms" << std::endl;

        cudaMemcpy(listarray, d_listarray, nnebs * sizeof(int) * _natoms, cudaMemcpyDeviceToHost);
        auto end_time4 = std::chrono::high_resolution_clock::now();
        auto duration4 = std::chrono::duration_cast<std::chrono::milliseconds>(end_time4-end_time3);
        //std::cout << "cpoy data to host: " << duration4.count() << "ms" << std::endl;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time-start_time);
        //std::cout << "total time: " << duration.count() << "ms" << std::endl;

        // gpu menory free
        cudaFree(d_nodes);
        cudaFree(d_nodes_shift);
        cudaFree(d_pos);
        cudaFree(d_listarray);
        cudaFree(d_min);
        cudaFree(d_max);
        cudaFree(d_cut);
        
        // cpu menory free
        // delete[] listarray;
    }

    std::vector<std::vector<int>> OctreeSearchGPU::get_particle_neighbor_info() {
        std::vector<std::vector<int>> neighbor_info;
        neighbor_info.resize(_natoms);
        for (int i = 0; i < _natoms; ++i) {
            for (int j = 0; j < nnebs; ++j) {
                int temp = listarray[i * nnebs + j];
                if (temp != -1 && temp != i) {
                    neighbor_info[i].push_back(temp);
                }
            }
        }
        return neighbor_info;
    }
}

