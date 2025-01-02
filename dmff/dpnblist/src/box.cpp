#include "box.h"
#include <stdexcept>


namespace dpnblist
{   
    Box::Box() {}

    Box::Box(std::vector<float> lengths, std::vector<float> angles)
    {       
        set_lengths_and_angles(lengths, angles);
    }

    Box::Box(vec3_float lengths, vec3_float angles)
    {
        set_lengths_and_angles(lengths, angles);
    }

    void Box::set_periodic(bool b_xp, bool b_yp, bool b_zp)
    {
        _periodic = {b_xp, b_yp, b_zp};
    }

    void Box::wrap(vec3_float& position)
    {
        // x
        if (_periodic.x)
        {
            if (position.x < 0)
            {
                position.x = fmod(position.x, _lengths.x) + _lengths.x;
            } else if (position.x > _lengths.x)
            {
                position.x = fmod(position.x, _lengths.x);
            }
        }

        // y
        if (_periodic.y)
        {
            if (position.y < 0)
            {
                position.y = fmod(position.y, _lengths.y) + _lengths.y;
            } else if (position.y > _lengths.y)
            {
                position.y = fmod(position.y, _lengths.y);
            }
        }

        // z
        if (_periodic.z)
        {
            if (position.z < 0)
            {
                position.z = fmod(position.z, _lengths.z) + _lengths.z;
            } else if (position.z > _lengths.z)
            {
                position.z = fmod(position.z, _lengths.z);
            }
        }
    }

    void Box::wrap(std::vector<float>& position)
    {   
        // Deal with the periodic condtion in each direction
        if (_periodic.x)
        {
            if (position[0] < 0)
            {
                position[0] = fmod(position[0], _lengths.x) + _lengths.x;
            } else if (position[0] > _lengths.x)
            {
                position[0] = fmod(position[0], _lengths.x);
            }
        }

        if (_periodic.y)
        {
            if (position[1] < 0)
            {
                position[1] = fmod(position[1], _lengths.y) + _lengths.y;
            } else if (position[1] > _lengths.y)
            {
                position[1] = fmod(position[1], _lengths.y);
            }
        }

        if (_periodic.z)
        {
            if (position[2] < 0)
            {
                position[2] = fmod(position[2], _lengths.z) + _lengths.z;
            } else if (position[2] > _lengths.z)
            {
                position[2] = fmod(position[2], _lengths.z);
            }
        }
    }

    void Box::wrap(std::array<float, 3>& position)
    {   
        // Deal with the periodic condtion in each direction
        if (_periodic.x)
        {
            if (position[0] < 0)
            {
                position[0] = fmod(position[0], _lengths.x) + _lengths.x;
            } else if (position[0] > _lengths.x)
            {
                position[0] = fmod(position[0], _lengths.x);
            }
        }

        if (_periodic.y)
        {
            if (position[1] < 0)
            {
                position[1] = fmod(position[1], _lengths.y) + _lengths.y;
            } else if (position[1] > _lengths.y)
            {
                position[1] = fmod(position[1], _lengths.y);
            }
        }

        if (_periodic.z)
        {
            if (position[2] < 0)
            {
                position[2] = fmod(position[2], _lengths.z) + _lengths.z;
            } else if (position[2] > _lengths.z)
            {
                position[2] = fmod(position[2], _lengths.z);
            }
        }
    }

    void Box::wrap_p(vec3_float* positions, int num_particle)
    {
        for (int seq = 0; seq < num_particle; seq++)
        {
            wrap(positions[seq]);
        }
    }

    void Box::wrap_p(std::vector<std::vector<float>>& positions)
    {
        for (int seq = 0; seq < positions.size(); seq++)
        {
            wrap(positions[seq]);
        }
    }

    void Box::wrap_p(std::vector<std::array<float,3>>& positions)
    {
        for (int seq = 0; seq < positions.size(); seq++)
        {
            wrap(positions[seq]);
        }
    }

    float Box::calc_sqrt_distance(std::vector<float>& r1, std::vector<float>& r2)
    {
        std::vector<float> diff2 = {0.f, 0.f, 0.f};
        float temp;
        std::vector<float> _v_lengths = {_lengths.x, _lengths.y, _lengths.z};

        for (int i = 0; i < 3; i++)
        {
            temp = r1[i] - r2[i] + 0.5 * _v_lengths[i];
            if (temp < 0)
            {
                diff2[i] = (temp + 0.5 * _v_lengths[i]) * (temp + 0.5 * _v_lengths[i]);
            } else if (temp > _v_lengths[i])
            {
                diff2[i] = (temp - 1.5 * _v_lengths[i]) * (temp - 1.5 * _v_lengths[i]);
            } else 
            {
                diff2[i] = (temp - 0.5 * _v_lengths[i]) * (temp - 0.5 * _v_lengths[i]);
            }
        }

        return diff2[0] + diff2[1] + diff2[2];
    }

    float Box::calc_sqrt_distance(vec3_float& r1, vec3_float& r2)
    {
        vec3_float diff2 = {0,0,0};
        float temp;
        
        temp = r1.x - r2.x + 0.5 * _lengths.x;
        if (temp < 0)
        {
            diff2.x = (temp + 0.5 * _lengths.x) * (temp + 0.5 * _lengths.x);
        } else if (temp > _lengths.x)
        {
            diff2.x = (temp - 1.5 * _lengths.x) * (temp - 1.5 * _lengths.x);
        } else 
        {
            diff2.x = (temp - 0.5 * _lengths.x) * (temp - 0.5 * _lengths.x);
        }

        temp = r1.y - r2.y + 0.5 * _lengths.y;
        if (temp < 0)
        {
            diff2.y = (temp + 0.5 * _lengths.y) * (temp + 0.5 * _lengths.y);
        } else if (temp > _lengths.y)
        {
            diff2.y = (temp - 1.5 * _lengths.y) * (temp - 1.5 * _lengths.y);
        } else 
        {
            diff2.y = (temp - 0.5 * _lengths.y) * (temp - 0.5 * _lengths.y);
        }

        temp = r1.z - r2.z + 0.5 * _lengths.z;
        if (temp < 0)
        {
            diff2.z = (temp + 0.5 * _lengths.z) * (temp + 0.5 * _lengths.z);
        } else if (temp > _lengths.z)
        {
            diff2.z = (temp - 1.5 * _lengths.z) * (temp - 1.5 * _lengths.z);
        } else 
        {
            diff2.z = (temp - 0.5 * _lengths.z) * (temp - 0.5 * _lengths.z);
        }

        return diff2.x + diff2.y + diff2.z;
    }

    float Box::calc_distance2(std::array<float,3>& r1, std::array<float,3>& r2)
    {
        std::vector<float> diff2 = {0.f, 0.f, 0.f};
        float temp;
        std::vector<float> _v_lengths = {_lengths.x, _lengths.y, _lengths.z};

        for (int i = 0; i < 3; i++)
        {
            float half_length = 0.5 * _v_lengths[i];
            temp = r1[i] - r2[i];
            if (temp < -half_length)
            {
                diff2[i] = (temp + _v_lengths[i]) * (temp + _v_lengths[i]);
            } else if (temp > half_length)
            {
                diff2[i] = (temp - _v_lengths[i]) * (temp - _v_lengths[i]);
            } else 
            {
                diff2[i] = temp * temp;
            }
        }

        return diff2[0] + diff2[1] + diff2[2];
    }

    void Box::set_lengths_and_angles(std::vector<float> lengths, std::vector<float> angles)
    {
        set_lengths(lengths);
        set_angles(angles);
    }
    
    void Box::set_lengths_and_angles(vec3_float lengths, vec3_float angles)
    {
        set_lengths(lengths);
        set_angles(angles);
    }

    void Box::set_lengths(std::vector<float> lengths)
    {
        _lengths = {lengths[0], lengths[1], lengths[2]};
    }

    void Box::set_lengths(vec3_float lengths)
    {
        _lengths = lengths;
    }

    void Box::set_angles(std::vector<float> angles)
    {
        _angles = {angles[0], angles[1], angles[2]};
    }

    void Box::set_angles(vec3_float angles)
    {
        _angles = angles;
    }

    std::vector<float> Box::get_lengths_cpu() const
    {   
        std::vector<float> res = {_lengths.x, _lengths.y, _lengths.z};

        return res;
    }

    std::array<float,3> Box::get_lengths_cpu3() const
    {
        std::array<float,3> res = {_lengths.x, _lengths.y, _lengths.z};

        return res;
    }

    vec3_float Box::get_lengths_gpu() const
    {   
        return _lengths;
    }

    std::vector<float> Box::get_angles_cpu() const
    {
        std::vector<float> res = {_angles.x, _angles.y, _angles.z};

        return res;
    }

    vec3_float Box::get_angles_gpu() const
    {
        return _angles;
    }

    std::vector<bool> Box::get_periodic_cpu() const
    {
        std::vector<bool> res = {_periodic.x, _periodic.y, _periodic.z};

        return res;
    }

    vec3_bool Box::get_periodic_gpu() const
    {
        return _periodic;
    }
}
