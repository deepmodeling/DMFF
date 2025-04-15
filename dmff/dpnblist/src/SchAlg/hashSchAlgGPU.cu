#include "hashSchAlgGPU.cuh"
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <algorithm>
#include <chrono>

namespace dpnblist
{
    // Initialization
    HashSearchGPU::HashSearchGPU(const Box& box, const std::vector<std::vector<float>> &xyz, float r_cutoff)
    : box(box), xyz(xyz), r_cutoff(r_cutoff) 
    {
        num_particles = xyz.size();

        r_cutoff_2 = r_cutoff * r_cutoff;
    
        box_len = box.get_lengths_gpu();

        cell_size = std::max(r_cutoff, box_len.x / std::floor(box_len.x / r_cutoff));

        offset = {
                {-cell_size, -cell_size, -cell_size},{-cell_size, -cell_size, 0},{-cell_size, -cell_size, cell_size},
                {-cell_size, 0, -cell_size},{-cell_size, 0, 0},{-cell_size, 0, cell_size},
                {-cell_size, cell_size, -cell_size},{-cell_size, cell_size, 0},{-cell_size, cell_size, cell_size},
                {0, -cell_size, -cell_size},{0, -cell_size, 0},{0, -cell_size, cell_size},
                {0, 0, -cell_size},{0, 0, 0},{0, 0, cell_size},
                {0, cell_size, -cell_size},{0, cell_size, 0},{0, cell_size, cell_size},
                {cell_size, -cell_size, -cell_size},{cell_size, -cell_size, 0},{cell_size, -cell_size, cell_size},
                {cell_size, 0, -cell_size},{cell_size, 0, 0},{cell_size, 0, cell_size},
                {cell_size, cell_size, -cell_size},{cell_size, cell_size, 0},{cell_size, cell_size, cell_size}
        };

        int n_neighbor_cells = offset.size();
        cudaMalloc((void**)&d_offset, n_neighbor_cells * sizeof(vec3_int));
        cudaMemcpy(d_offset, offset.data(), n_neighbor_cells * sizeof(vec3_int), cudaMemcpyHostToDevice);
    }

    ////////////////////////////////////////////////////////////////////////////
    /*CUDA code for search algorithm*/

    // encode and decode
    __device__ int d_spreadBits(int x, int offset)
    {
        x = (x | (x << 10)) & 0x000F801F;
        x = (x | (x <<  4)) & 0x00E181C3;
        x = (x | (x <<  2)) & 0x03248649;
        x = (x | (x <<  2)) & 0x09249249;

        return x << offset;
    }

    __device__ int d_encodeMortonNumber(vec3_float xyz, float rc)
    {
        int x, y, z;
        x = static_cast<int>(xyz.x / rc);
        y = static_cast<int>(xyz.y / rc);
        z = static_cast<int>(xyz.z / rc);

        return d_spreadBits(x, 0) | d_spreadBits(y, 1) | d_spreadBits(z, 2);
    }


    __device__ int d_compressBits(int x)
    {
        x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
        x = (x ^ (x >> 2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
        x = (x ^ (x >> 4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
        x = (x ^ (x >> 8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
        x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
        return x;
    }

    __device__ vec3_float d_decodeMortonNumber(int morton, float rc)
    {
        int x, y, z;
    
        x = d_compressBits(morton);
        y = d_compressBits(morton >> 1);
        z = d_compressBits(morton >> 2);

        vec3_float xyz;
        // Use rc to scale the coordinate
        xyz.x = x * rc;
        xyz.y = y * rc;
        xyz.z = z * rc;

        return xyz;
    }

    // Calculate Morton codes of each particle and create a corresponding list 
    __global__ void cal_morton_values(int num_particles, float rc, vec3_float *inputs, int *_d_morton_values_list)
    {
        int ind = blockDim.x * blockIdx.x + threadIdx.x;

        if (ind < num_particles)
        {
            _d_morton_values_list[ind] = d_encodeMortonNumber(inputs[ind], rc);
        }
    }

    // Create a compact list
    __global__ void cal_hash_list(int n, int *mv_list, pair2_int *c_mv_list)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x; 

        if (i < n)
        {
            if (i == 0) c_mv_list[mv_list[i]].l = i;

            if (i > 0)
            {
                if (mv_list[i] != mv_list[i - 1])
                {
                    c_mv_list[mv_list[i]].l = i;
                    c_mv_list[mv_list[i - 1]].r = i;
                } 
            }

            if (i == n - 1)
            {
                c_mv_list[mv_list[i]].r = i + 1;
            }
        }
    } 

    // Get indices of neighboring cells
    __device__ void get_neighbor_cells(int cell_index, float rc , vec3_float* offset, vec3_float box_length, 
                                        int* neighbor_cells_list)
    {
        vec3_float cell_vector = d_decodeMortonNumber(cell_index, rc);

        vec3_float cell_vector_temp;
        int cell_ind_temp;
        for (int i = 0; i < 27; i++)
        {
            cell_vector_temp.x = cell_vector.x + offset[i].x + 1e-5;
            cell_vector_temp.y = cell_vector.y + offset[i].y + 1e-5;
            cell_vector_temp.z = cell_vector.z + offset[i].z + 1e-5;

            // 周期性边界处理
            if (cell_vector_temp.x < 0)
            {
                cell_vector_temp.x += box_length.x;
            } else if (cell_vector_temp.x >= box_length.x)
            {
                cell_vector_temp.x -= box_length.x;
            }

            if (cell_vector_temp.y < 0)
            {
                cell_vector_temp.y += box_length.y;
            } else if (cell_vector_temp.y >= box_length.y)
            {
                cell_vector_temp.y -= box_length.y;
            }

            if (cell_vector_temp.z < 0)
            {
                cell_vector_temp.z += box_length.z;
            } else if (cell_vector_temp.z >= box_length.z)
            {
                cell_vector_temp.z -= box_length.z;
            }

            cell_ind_temp = d_encodeMortonNumber(cell_vector_temp, rc);

            // if (cell_index == 114880)
            // {
            //     printf("base x: %f, y: %f, z: %f\n", cell_vector.x, cell_vector.y, cell_vector.z);
            //     printf("x: %f, y: %f, z: %f\n", cell_vector_temp.x, cell_vector_temp.y, cell_vector_temp.z);
            //     printf("morton code : %d\n", cell_ind_temp);
            // }

            neighbor_cells_list[27 * cell_index + i] = cell_ind_temp;
        }
    }

    __global__ void get_neighbor_cells_list(int num_cells, float rc, vec3_float box_length, vec3_float* offset, pair2_int* compact_list,
                                            int* neighbor_cells_list) 
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x; 

        if (i <= num_cells)
        {
            if (compact_list[i].l != -1)
            {
                get_neighbor_cells(i, rc, offset, box_length, neighbor_cells_list);
            }
        }
    }

    // Cal the periodic distance between particles
    __device__ float cal_sqrt_distance_p(vec3_float pos_i, vec3_float pos_j, vec3_float box_length) {
        vec3_float diff;
        float diff_temp;

        // x
        diff_temp = pos_i.x - pos_j.x;
        if (diff_temp < -0.5 * box_length.x)
        {
            diff.x = diff_temp + box_length.x;
        } else if (diff_temp > 0.5 * box_length.x)
        {
            diff.x = diff_temp - box_length.x;
        } else 
        {
            diff.x = diff_temp;
        }

        // y
        diff_temp = pos_i.y - pos_j.y;
        if (diff_temp < -0.5 * box_length.y)
        {
            diff.y = diff_temp + box_length.y;
        } else if (diff_temp > 0.5 * box_length.y)
        {
            diff.y = diff_temp - box_length.y;
        } else 
        {
            diff.y = diff_temp;
        }

        // z
        diff_temp = pos_i.z - pos_j.z;
        if (diff_temp < -0.5 * box_length.z)
        {
            diff.z = diff_temp + box_length.z;
        } else if (diff_temp > 0.5 * box_length.z)
        {
            diff.z = diff_temp - box_length.z;
        } else 
        {
            diff.z = diff_temp;
        }

        return (diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
    }


    __global__ void  search_neighbors(float rc2, int num_samples, int num_neighbor_particle, vec3_float box_length, 
                                    vec3_float *inputs, int* new_indices, int* morton_list, int* neighbor_cells_list, pair2_int *compact_list,
                                    int* particle_list)
    {
        
        int particle_seq = blockIdx.x * blockDim.x + threadIdx.x;
        if (particle_seq < num_samples)
        {
            vec3_float particle_xyz = inputs[new_indices[particle_seq]];
            int particle_mv = morton_list[particle_seq];

            // if (new_indices[particle_seq] == 63)
            // {
            //     printf("device - morton value: %d\n", particle_mv);
            // }

            int count = 0;
            for (int i = 0; i < 27; i++)
            {
                int neighbor_cell_temp = neighbor_cells_list[particle_mv * 27 + i];

                // if (new_indices[particle_seq] == 63)
                // {
                //     printf("device - neighbor cell: %d\n", neighbor_cell_temp);
                // }

                if (compact_list[neighbor_cell_temp].l != compact_list[neighbor_cell_temp].r)
                {
                    for (int near_neighbor_seq = compact_list[neighbor_cell_temp].l; 
                        near_neighbor_seq < compact_list[neighbor_cell_temp].r; 
                        near_neighbor_seq++)
                    {
                        if (near_neighbor_seq == particle_seq) continue;

                        vec3_float near_neighbor_xyz = inputs[new_indices[near_neighbor_seq]];

                        float distance2 = cal_sqrt_distance_p(particle_xyz, near_neighbor_xyz, box_length);
                        
                        // if (new_indices[particle_seq] == 63 && new_indices[near_neighbor_seq] == 524)
                        // {
                        //     printf("device - distance2: %f\n", distance2);
                        // }

                        if (distance2 - rc2 < 1e-10)
                        {
                            particle_list[new_indices[particle_seq] * num_neighbor_particle + count] = new_indices[near_neighbor_seq];
                            count++;
                        }
                    }
                }

            }
        }
    }

    void HashSearchGPU::search() 
    {
        reset();

        // Set the number of neighboring particles
        num_nb_particle = static_cast<int>(std::ceil(1.5 * (5 * xyz.size() / (box_len.x * box_len.y * box_len.z)) * r_cutoff_2 * r_cutoff + 45));
        if (num_nb_particle < 100) num_nb_particle = min_num_nb_particle;

        // Move the data to gpu
        vec3_float* h_xyz = new vec3_float[xyz.size()];
        data2struc(xyz, h_xyz);

        vec3_float *d_xyz;
        cudaMalloc((void**)&d_xyz, num_particles * sizeof(vec3_float));
        cudaMemcpy(d_xyz, h_xyz, num_particles * sizeof(vec3_float), cudaMemcpyHostToDevice);

        // Cal morton values
        cudaMalloc((void**)&d_morton_list, num_particles * sizeof(int));

        int blockSize_cmv = 128;
        int gridSize_cmv = (num_particles + blockSize_cmv - 1) / blockSize_cmv;
        cal_morton_values<<<gridSize_cmv, blockSize_cmv>>>(num_particles, cell_size, d_xyz, d_morton_list);
        cudaDeviceSynchronize();

        // 按照morton values对indices进行排序
        new_indices = new int[num_particles];
        for (int i = 0; i < num_particles; i++)
        {
            new_indices[i] = i;
        }
        
        cudaMalloc((void**)&d_new_indices, num_particles * sizeof(int));
        cudaMemcpy(d_new_indices, new_indices, num_particles * sizeof(int), cudaMemcpyHostToDevice);

        thrust::device_ptr<int> t_d_morton_list(d_morton_list);
        thrust::device_ptr<int> t_d_new_indices(d_new_indices);
        thrust::sort_by_key(t_d_morton_list, t_d_morton_list + num_particles, t_d_new_indices);

        // Calculate the max morton value = the num of cells
        int d_max_result = -1;
        // thrust::device_ptr<int> t_d_morton_list(d_morton_list);
        d_max_result = *(thrust::max_element(t_d_morton_list, t_d_morton_list + num_particles));
        d_num_cells = d_max_result;

        // Calculate the compact list
        compact_list = new pair2_int[d_num_cells + 1]; 
        cudaMalloc((void**)&d_compact_list, (d_num_cells + 1) * sizeof(pair2_int)); 
        cudaMemcpy(d_compact_list, compact_list, (d_num_cells + 1) * sizeof(pair2_int), cudaMemcpyHostToDevice);

        int blockSize_cl = 128;
        int gridSize_cl = (num_particles + blockSize_cl - 1) / blockSize_cl;
        cal_hash_list<<<gridSize_cl, blockSize_cl>>>(num_particles, d_morton_list, d_compact_list);

        cudaDeviceSynchronize();

        // Calculate the neighboring cells of all cells
        cudaMalloc((void**)&d_neighbor_cells_list, 27 * (d_num_cells + 1) * sizeof(int));

        int blockSize_gncl = 128;
        int gridSize_gncl = (d_num_cells + blockSize_gncl - 1) / blockSize_gncl;

        get_neighbor_cells_list<<<gridSize_gncl, blockSize_gncl>>>(d_num_cells, cell_size, box_len, d_offset, d_compact_list, 
                                                                        d_neighbor_cells_list);
        cudaDeviceSynchronize();

        // Core search
        int blockSize_build = 128;
        int gridSize_build = (num_particles + blockSize_build - 1) / blockSize_build;

        particle_list = new int[num_nb_particle * num_particles];
        for (int i = 0; i < num_nb_particle * num_particles; i++)
        {
            particle_list[i] = -1;
        }

        // this->d_particle_list;
        cudaMalloc((void**)&d_particle_list, num_nb_particle * num_particles * sizeof(int));
        cudaMemcpy(d_particle_list, particle_list, num_nb_particle * num_particles * sizeof(int), cudaMemcpyHostToDevice);

        search_neighbors<<<gridSize_build, blockSize_build>>>(r_cutoff_2, num_particles, num_nb_particle, box_len, 
                                                            d_xyz, d_new_indices, d_morton_list, d_neighbor_cells_list, d_compact_list,
                                                            d_particle_list);
        cudaDeviceSynchronize();

        delete[] h_xyz;

        cudaMemcpy(particle_list, d_particle_list, num_nb_particle * num_particles * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(new_indices, d_new_indices, num_particles * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_xyz);
        cudaFree(d_morton_list);
        cudaFree(d_new_indices);
        cudaFree(d_compact_list);
        cudaFree(d_neighbor_cells_list);
        cudaFree(d_particle_list);

        search_sign = 1;
    }

    double HashSearchGPU::search_time()
    {
        reset();

        // Set the number of neighboring particles
        num_nb_particle = static_cast<int>(std::ceil(xyz.size() / (box_len.x * box_len.y * box_len.z)) * cell_size) * 100;
        if (num_nb_particle < 100) num_nb_particle = min_num_nb_particle;

        // Move the data to gpu
        vec3_float* h_xyz = new vec3_float[xyz.size()];
        data2struc(xyz, h_xyz);

        vec3_float *d_xyz;
        cudaMalloc((void**)&d_xyz, num_particles * sizeof(vec3_float));
        cudaMemcpy(d_xyz, h_xyz, num_particles * sizeof(vec3_float), cudaMemcpyHostToDevice);

        auto start_time = std::chrono::steady_clock::now();
        // Cal morton values
        cudaMalloc((void**)&d_morton_list, num_particles * sizeof(int));

        int blockSize_cmv = 128;
        int gridSize_cmv = (num_particles + blockSize_cmv - 1) / blockSize_cmv;
        cal_morton_values<<<gridSize_cmv, blockSize_cmv>>>(num_particles, cell_size, d_xyz, d_morton_list);
        cudaDeviceSynchronize();

        // 按照morton values对indices进行排序
        new_indices = new int[num_particles];
        for (int i = 0; i < num_particles; i++)
        {
            new_indices[i] = i;
        }
        
        cudaMalloc((void**)&d_new_indices, num_particles * sizeof(int));
        cudaMemcpy(d_new_indices, new_indices, num_particles * sizeof(int), cudaMemcpyHostToDevice);

        thrust::device_ptr<int> t_d_morton_list(d_morton_list);
        thrust::device_ptr<int> t_d_new_indices(d_new_indices);
        thrust::sort_by_key(t_d_morton_list, t_d_morton_list + num_particles, t_d_new_indices);

        // Calculate the max morton value = the num of cells
        int d_max_result = -1;
        // thrust::device_ptr<int> t_d_morton_list(d_morton_list);
        d_max_result = *(thrust::max_element(t_d_morton_list, t_d_morton_list + num_particles));
        d_num_cells = d_max_result;

        // Calculate the compact list
        compact_list = new pair2_int[d_num_cells + 1]; 
        cudaMalloc((void**)&d_compact_list, (d_num_cells + 1) * sizeof(pair2_int)); 
        cudaMemcpy(d_compact_list, compact_list, (d_num_cells + 1) * sizeof(pair2_int), cudaMemcpyHostToDevice);

        int blockSize_cl = 128;
        int gridSize_cl = (num_particles + blockSize_cl - 1) / blockSize_cl;
        cal_hash_list<<<gridSize_cl, blockSize_cl>>>(num_particles, d_morton_list, d_compact_list);

        cudaDeviceSynchronize();

        // Calculate the neighboring cells of all cells
        cudaMalloc((void**)&d_neighbor_cells_list, 27 * (d_num_cells + 1) * sizeof(int));

        int blockSize_gncl = 128;
        int gridSize_gncl = (d_num_cells + blockSize_gncl - 1) / blockSize_gncl;

        get_neighbor_cells_list<<<gridSize_gncl, blockSize_gncl>>>(d_num_cells, cell_size, box_len, d_offset, d_compact_list, 
                                                                        d_neighbor_cells_list);
        cudaDeviceSynchronize();

        // Core search
        int blockSize_build = 128;
        int gridSize_build = (num_particles + blockSize_build - 1) / blockSize_build;

        particle_list = new int[num_nb_particle * num_particles];
        for (int i = 0; i < num_nb_particle * num_particles; i++)
        {
            particle_list[i] = -1;
        }

        // this->d_particle_list;
        cudaMalloc((void**)&d_particle_list, num_nb_particle * num_particles * sizeof(int));
        cudaMemcpy(d_particle_list, particle_list, num_nb_particle * num_particles * sizeof(int), cudaMemcpyHostToDevice);

        search_neighbors<<<gridSize_build, blockSize_build>>>(r_cutoff_2, num_particles, num_nb_particle, box_len, 
                                                            d_xyz, d_new_indices, d_morton_list, d_neighbor_cells_list, d_compact_list,
                                                            d_particle_list);
        cudaDeviceSynchronize();
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        std::cout << "GPU Search Execution time: " << duration << " milliseconds" << std::endl;

        delete[] h_xyz;

        cudaMemcpy(particle_list, d_particle_list, num_nb_particle * num_particles * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(new_indices, d_new_indices, num_particles * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_xyz);
        cudaFree(d_morton_list);
        cudaFree(d_new_indices);
        cudaFree(d_compact_list);
        cudaFree(d_neighbor_cells_list);
        cudaFree(d_particle_list);

        search_sign = 1;

        return duration;
    }

    void HashSearchGPU::reset() 
    {
        // std::cout << 0 << std::endl;
        // if (new_indices != nullptr)
        // {
        //     std::cout << "intern" << std::endl;
        //     delete[] new_indices;
        //     std::cout << "intern" << std::endl;
        // }
        // std::cout << 1 << std::endl;

        // if (compact_list != nullptr)
        // {
        //     delete[] compact_list;
        // }

        // if (particle_list != nullptr)
        // {
        //     delete[] particle_list;
        // }
    }

    std::vector<std::vector<int>> HashSearchGPU::get_particle_neighbor_info()
    {
        std::vector<std::vector<int>> res;
        res.resize(num_particles);

        if (search_sign)
        {
            for (int i = 0; i < num_particles; i++)
            {
                std::vector<int> temp_vec;
                for (int j = 0; j < num_nb_particle; j++)
                {
                    int temp_seq = particle_list[i * num_nb_particle + j];
                    if (temp_seq >= 0)
                    {
                        temp_vec.push_back(temp_seq);
                    } else 
                    {
                        res[i] = temp_vec;
                        break;
                    }                    
                }
            }
        } 
        return res;
    }

}