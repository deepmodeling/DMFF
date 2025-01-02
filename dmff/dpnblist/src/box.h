#pragma once
#include <vector>
#include <array>
#include <cmath>

namespace dpnblist
{
    struct vec3_float {
        float x;
        float y;
        float z;
    };

    struct vec3_int {
        int x;
        int y;
        int z;
    };

    struct vec3_bool {
        bool x;
        bool y;
        bool z;
    };

    struct pair2_int {
        int l = -1;
        int r = -1;
    };
    
    class Box
    {

        public:
            Box();
            Box(std::vector<float> lengths, std::vector<float> angles = {90, 90, 90});
            Box(vec3_float lengths, vec3_float angles = {90, 90, 90});

            // wrap
            void wrap(vec3_float& position);
            void wrap(std::vector<float>& position);
            void wrap(std::array<float, 3>& position);

            void wrap_p(vec3_float* positions, int num_particle);
            void wrap_p(std::vector<std::vector<float>>& positions);
            void wrap_p(std::vector<std::array<float,3>>& positions);
            
            // 计算距离
            float calc_sqrt_distance(std::vector<float>& point1, std::vector<float>& point2);
            float calc_sqrt_distance(vec3_float& r1, vec3_float& r2);
            float calc_distance2(std::array<float,3>& r1, std::array<float,3>& r2);

            // set & get
            void set_periodic(bool b_xp, bool b_yp, bool b_zp);

            void set_lengths_and_angles(std::vector<float> lengths, std::vector<float> angles);
            void set_lengths_and_angles(vec3_float lengths, vec3_float angles);

            void set_lengths(std::vector<float> lengths);
            void set_lengths(vec3_float lengths);

            void set_angles(std::vector<float> angles);
            void set_angles(vec3_float angles);

            std::vector<bool> get_periodic_cpu() const;
            vec3_bool get_periodic_gpu() const;

            std::vector<float> get_lengths_cpu() const;
            std::array<float,3> get_lengths_cpu3() const;
            vec3_float get_lengths_gpu() const;

            std::vector<float> get_angles_cpu() const;
            vec3_float get_angles_gpu() const;

        private:
            vec3_bool _periodic;
            vec3_float _lengths;
            vec3_float _angles;
    };

}
