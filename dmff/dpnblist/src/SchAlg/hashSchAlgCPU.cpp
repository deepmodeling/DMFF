#include "hashSchAlgCPU.h"

#include <vector>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include <chrono>

namespace dpnblist
{
    int spreadBits(int x, int offset) {
        x = (x | (x << 10)) & 0x000F801F;
        x = (x | (x <<  4)) & 0x00E181C3;
        x = (x | (x <<  2)) & 0x03248649;
        x = (x | (x <<  2)) & 0x09249249;

        return x << offset;
    }

    // 将三维坐标映射为32的Z-order curve结果
    int encodeMortonNumber(const std::vector<float>& xyz, float rc) {
        int x, y, z;
        x = static_cast<int>(xyz[0] / rc);
        y = static_cast<int>(xyz[1] / rc);
        z = static_cast<int>(xyz[2] / rc);

        return spreadBits(x, 0) | spreadBits(y, 1) | spreadBits(z, 2);
    }

    int compressBits(int x) {
        x &= 0x09249249;                  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
        x = (x ^ (x >> 2)) & 0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
        x = (x ^ (x >> 4)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
        x = (x ^ (x >> 8)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
        x = (x ^ (x >> 16)) & 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
        return x;
    }

    // decode xyz from 32bits morton codes
    std::vector<float> decodeMortonNumber(int morton, float rc) {
        int x, y, z;
        
        x = compressBits(morton);
        y = compressBits(morton >> 1);
        z = compressBits(morton >> 2);

        // 使用 rc 进行缩放，得到实际坐标
        std::vector<float> xyz = {x * rc, y * rc, z * rc};
        return xyz;
    }

    HashSearchCPU::HashSearchCPU(const Box& box, const std::vector<std::vector<float>>& xyz, float r_cutoff)
    : box(box), xyz(xyz), r_cutoff(r_cutoff)
    {
        num_particles = xyz.size();
        // std::cout << "1" << std::endl;

        r_cutoff_2 = r_cutoff * r_cutoff;
        // std::cout << "2" << std::endl;

        cube_size = box.get_lengths_cpu();
        // std::cout << "3" << std::endl;

        cell_size = std::max(r_cutoff, cube_size[0] / std::floor(cube_size[0] / r_cutoff));
        // std::cout << "cell_size: " << cell_size << std::endl;

        offset = {
            {-cell_size,-cell_size, -cell_size}, {-cell_size,-cell_size, 0}, {-cell_size,-cell_size, cell_size},
            {-cell_size,   0, -cell_size}, {-cell_size,   0, 0}, {-cell_size,   0, cell_size},
            {-cell_size, cell_size, -cell_size}, {-cell_size, cell_size, 0}, {-cell_size, cell_size, cell_size},
            {   0,-cell_size, -cell_size}, {   0,-cell_size, 0}, {   0,-cell_size, cell_size},
            {   0,   0, -cell_size}, {   0,   0, 0}, {   0,   0, cell_size},
            {   0, cell_size, -cell_size}, {   0, cell_size, 0}, {   0, cell_size, cell_size},
            { cell_size,-cell_size, -cell_size}, { cell_size,-cell_size, 0}, { cell_size,-cell_size, cell_size},
            { cell_size,   0, -cell_size}, { cell_size,   0, 0}, { cell_size,   0, cell_size},
            { cell_size, cell_size, -cell_size}, { cell_size, cell_size, 0}, { cell_size, cell_size, cell_size}
        };
    }

    std::vector<float> HashSearchCPU::get_min_diff(const std::vector<float>& xyz1, const std::vector<float>& xyz2) {
        std::vector<float> difference(3);
        float diff;
        for (int i = 0; i < 3; i++) {
            diff = xyz1[i] - xyz2[i];
            // difference[i] = diff;

            if (diff < -cube_size[i] / 2) {
                difference[i] = diff + cube_size[i];
            }
            else if (diff > cube_size[i] / 2) {
                difference[i] = diff - cube_size[i];
            }else {
                difference[i] = diff;  // 在没有超出范围时，直接使用原始差值
            }
        }
        return difference;
    }

    std::vector<float> HashSearchCPU::get_min_diff_updated(const std::vector<float>& xyz1, const std::vector<float>& xyz2, int cell_index) {
        std::vector<float> difference(3);

        float diff;
        for (int i = 0; i < 3; i++) {
            diff = xyz1[i] - xyz2[i];
            difference[i] = diff;

            if (boundary_cells_sign[cell_index])
            {
                if (diff < -cube_size[i] / 2) {
                    difference[i] = diff + cube_size[i];
                }
                else if (diff > cube_size[i] / 2) {
                    difference[i] = diff - cube_size[i];
                }else {
                    difference[i] = diff;  // 在没有超出范围时，直接使用原始差值
                }
            }
        }

        return difference;
    }

    // Calculate the morton code of each xyz
    void HashSearchCPU::cal_morton_values(const std::vector<std::vector<float>>& inputs) {
        morton_values_list.resize(inputs.size());
        max_morton_value = -1;
        int temp;
        for (int i = 0; i < inputs.size(); i++)
        {
            temp = encodeMortonNumber(inputs[i], cell_size);
            if (temp > max_morton_value) max_morton_value = temp;
            morton_values_list[i] = temp;
        }
        num_cells = max_morton_value;
    }

    void HashSearchCPU::sort_by_morton() {
        new_indices.resize(morton_values_list.size());
        for (size_t i = 0; i < new_indices.size(); ++i) {
            new_indices[i] = i;
        }

        // 使用 std::sort，排序 indices，按照 b 中对应元素的值排序
        std::sort(new_indices.begin(), new_indices.end(), [this](int i, int j) {
            return morton_values_list[i] < morton_values_list[j];
        });
    }

    // Create the relationtion between positions and cells
    void HashSearchCPU::cal_hash_list() 
    {
        hash_list.resize(num_cells + 1, std::make_pair(-1, -1));

        for (int i = 0; i < new_indices.size(); i++) 
        {
            if (i == 0) hash_list[morton_values_list[new_indices[i]]].first = i;

            if (i > 0)
            {
                if (morton_values_list[new_indices[i]] != morton_values_list[new_indices[i - 1]])
                {
                    hash_list[morton_values_list[new_indices[i]]].first = i;
                    hash_list[morton_values_list[new_indices[i - 1]]].second = i;
                } 
            }

            if (i == new_indices.size() - 1)
            {
                hash_list[morton_values_list[new_indices[i]]].second = i + 1;
            }
        }
    }

    std::vector<int> HashSearchCPU::get_neighbor_cells(int cell_index) {
        std::vector<int> neighbor_cells(27, -1);

        // Calculate the morton value
        std::vector<float> cell_vector(3);
        cell_vector = decodeMortonNumber(cell_index, cell_size);

        // Tranverse those offs to get near neighbors
        std::vector<float> cell_vector_temp(3);
        // std::cout << "offset size: " << offset.size() << std::endl;
        for (int i = 0; i < offset.size(); i++)
        {
            // Obtain the neighboring cell vector
            cell_vector_temp[0] = cell_vector[0] + offset[i][0] + 1e-5;
            cell_vector_temp[1] = cell_vector[1] + offset[i][1] + 1e-5;
            cell_vector_temp[2] = cell_vector[2] + offset[i][2] + 1e-5;

            // Deal with the periodic boundary 
            if (cell_vector_temp[0] < 0) {
                cell_vector_temp[0] += cube_size[0];

                boundary_cells_sign[cell_index] = true; // for New MIC
            } else if (cell_vector_temp[0] - cube_size[0]>= 1e-6) {
                cell_vector_temp[0] -= cube_size[0];

                boundary_cells_sign[cell_index] = true; // for New MIC
            }

            if (cell_vector_temp[1] < 0) {
                cell_vector_temp[1] += cube_size[1];

                boundary_cells_sign[cell_index] = true; // for New MIC
            } else if (cell_vector_temp[1] - cube_size[1]>= 1e-6) {
                cell_vector_temp[1] -= cube_size[1];

                boundary_cells_sign[cell_index] = true; // for New MIC
            }

            if (cell_vector_temp[2] < 0) {
                cell_vector_temp[2] += cube_size[2];

                boundary_cells_sign[cell_index] = true; // for New MIC
            } else if (cell_vector_temp[2]- cube_size[1]>= 1e-6) {
                cell_vector_temp[2] -= cube_size[2];

                boundary_cells_sign[cell_index] = true; // for New MIC
            }

            int cell_ind_temp = encodeMortonNumber(cell_vector_temp, cell_size);

            // if (cell_index == 265)
            // {
            //     std::cout << cell_vector_temp[0] << " " << cell_vector_temp[1] << " " << cell_vector_temp[2] << std::endl;
            //     std::cout << cell_ind_temp << " ";
            //     std::cout << std::endl;
            // }
       
            neighbor_cells[i] = cell_ind_temp;
        }

        return neighbor_cells; 
    }

    void HashSearchCPU::get_neighbor_cells_list() {
        neighbor_cells_list.resize(num_cells + 1);

        // for New MIC
        boundary_cells_sign.resize(num_cells + 1);
        boundary_cells_sign.assign(num_cells + 1, false); 

        for (int i = 0; i < num_cells + 1; i++) 
        {
            if (hash_list[i].first != -1) {
                neighbor_cells_list[i] = this->get_neighbor_cells(i);
            }
        }

    }

    void HashSearchCPU::search()
    {   
        reset();

        num_nb_particle = static_cast<int>(std::ceil(1.5 * (5 * xyz.size() / (cube_size[0] * cube_size[1] * cube_size[2])) * r_cutoff_2 * r_cutoff + 45));
        if (num_nb_particle < 100) num_nb_particle = min_num_nb_particle;
        // std::cout << "num_neighbor_particles: " << num_nb_particle << std::endl;

        particle_list.resize((num_particles + 1) * num_nb_particle, -1);
        int particle_list_size = (num_particles + 1) * num_nb_particle; 
        // std::cout << "particle list size: " << particle_list_size << std::endl;
        
        cal_morton_values(xyz);
        sort_by_morton();
        cal_hash_list();
        get_neighbor_cells_list(); 

        // std::cout << "max morton values: " << num_cells << std::endl;
        // std::cout << hash_list[4095].first << " " << hash_list[4095].second << std::endl;

        // std::cout << "stage 1!" << std::endl;

        // Tranverse by particles
        #pragma omp parallel for schedule(dynamic)
        for (int particle_seq = 0; particle_seq < num_particles; particle_seq++)
        {
            std::vector<float> particle_xyz = xyz[new_indices[particle_seq]];
            int particle_morton = morton_values_list[new_indices[particle_seq]];

            // if (new_indices[particle_seq] == 4)
            // {
            //     std::cout << "4 morton value: " << particle_morton << std::endl;
            // }

            int count = 0;

            for (auto& neighbor_cell : neighbor_cells_list[particle_morton])
            {
                // if (new_indices[particle_seq] == 4)
                // {
                //     std::cout << "4 neighbor cell: " << neighbor_cell << std::endl;
                // }

                if (hash_list[neighbor_cell].first == -1) continue;
                
                for (int near_neighbor_seq = hash_list[neighbor_cell].first; 
                    near_neighbor_seq < hash_list[neighbor_cell].second; 
                    near_neighbor_seq++)
                {
                    if (near_neighbor_seq == particle_seq) continue;

                    std::vector<float> near_neighbor_xyz = xyz[new_indices[near_neighbor_seq]];

                    // std::vector<float> diff = get_min_diff(particle_xyz, near_neighbor_xyz); 

                    std::vector<float> diff = get_min_diff_updated(particle_xyz, near_neighbor_xyz, particle_morton);  // for New MIC

                    float distance2 = diff[0] * diff[0] + 
                                        diff[1] * diff[1] + 
                                        diff[2] * diff[2];
                    
                    // if (new_indices[particle_seq] == 4 && new_indices[near_neighbor_seq] == 919)
                    // {
                    //     std::cout << particle_xyz[0] << " " << particle_xyz[1] << " " << particle_xyz[2] << std::endl;
                    //     std::cout << near_neighbor_xyz[0] << " " << near_neighbor_xyz[1] <<  " " << near_neighbor_xyz[2] << std::endl;

                    //     std::cout << "Distance2: " << distance2 << " r_cutoff_2: " << r_cutoff_2 << " sign: " << (distance2 - r_cutoff_2 <= 1e-7) << std::endl;
                    // }

                    if (r_cutoff_2 - distance2 >= 1e-7)
                    {
                        particle_list[new_indices[particle_seq] * num_nb_particle + count] = new_indices[near_neighbor_seq];
                        if (new_indices[particle_seq] * num_nb_particle + count > particle_list_size)
                        {
                            std::cout << "list size: " << particle_list_size << std::endl;
                            std::cout << "base: " << new_indices[particle_seq] * num_nb_particle << std::endl;
                            std::cout << "adder: " << count << std::endl;
                            abort();
                        }
                        count++;

                        // try {
                        //     particle_list[new_indices[particle_seq] * num_nb_particle + count] = new_indices[near_neighbor_seq];
                        //     count++;
                        // } catch (const std::out_of_range& e) {
                        //     // 处理数组越界异常的代码
                        //     std::cerr << "Array out of bounds: " << e.what() << std::endl;
                        // }

                    }
                }
            }
        }

        search_sign = 1;
    }

    void HashSearchCPU::reset()
    {
        morton_values_list.clear();
        new_indices.clear();

        // C_list.clear();
        neighbor_cells_list.clear();
        hash_list.clear();                               

        particle_list.clear();                           
    }


    std::vector<std::vector<int>> HashSearchCPU::get_particle_neighbor_info()
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
                        // if (num_particles == 7308)
                        // {
                        //     std::cout << temp_seq << " ";
                        // }
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