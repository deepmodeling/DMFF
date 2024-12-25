#include "../box.h"
#include "../nbSchAlg.h"
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
namespace dpnblist
{
    template <typename T>
    struct Vecc3 {
        T elements[3];

        __host__ __device__ Vecc3() {
            elements[0] = elements[1] = elements[2] = 0;
        }
        
        __host__ __device__ Vecc3(T x, T y, T z) {
            elements[0] = x;
            elements[1] = y;
            elements[2] = z;
        }

        __host__ __device__ T& operator[](int index) {
            return elements[index];
        }

        __host__ __device__ Vecc3<T> operator+(const Vecc3<T>& v) {
            return Vecc3<T>(elements[0] + v.elements[0], elements[1] + v.elements[1], elements[2] + v.elements[2]);
        }

        __host__ __device__ Vecc3<T> operator-(const Vecc3<T>& v) {
            return Vecc3<T>(elements[0] - v.elements[0], elements[1] - v.elements[1], elements[2] - v.elements[2]);
        }

        // __host__ __device__ Vecc3<T> operator*(const Vecc3<T>& v) {
        //     return Vecc3<T>(elements[0] * v.elements[0], elements[1] * v.elements[1], elements[2] * v.elements[2]);
        // }

        template <typename U>
        __host__ __device__ Vecc3<T> operator*(const Vecc3<U>& v) const {
            return Vecc3<T>(elements[0] * static_cast<T>(v.elements[0]), elements[1] * static_cast<T>(v.elements[1]), elements[2] * static_cast<T>(v.elements[2]));
        }

        __host__ __device__ Vecc3<T> operator/(const T& t) {
            return Vecc3<T>(elements[0] / t, elements[1] / t, elements[2] / t);
        }

    };

    template <typename T>
    struct Vecc4 {
        T elements[4];

        Vecc4(T x, T y, T z, T w) {
            elements[0] = x;
            elements[1] = y;
            elements[2] = z;
            elements[3] = w;
        }

        __host__ __device__ T& operator[](int index) {
            return elements[index];
        }
    };

    template <typename T>
    struct Vecc8 {
        T elements[8];

        Vecc8(T x, T y, T z, T w, T a, T b, T c, T d) {
            elements[0] = x;
            elements[1] = y;
            elements[2] = z;
            elements[3] = w;
            elements[4] = a;
            elements[5] = b;
            elements[6] = c;
            elements[7] = d;
        }

        __host__ __device__ T& operator[](int index) {
            return elements[index];
        }
    };

    struct NodeGPU {
        Vecc3<float> center;
        Vecc3<float> size;
        Vecc4<int> pos_index;
        Vecc8<int> children;
        int threadid;
        NodeGPU(Vecc3<float> center, Vecc3<float> size, int threadid) : center(center), size(size), threadid(threadid), pos_index(-1, -1, -1, -1), children(-1, -1, -1, -1, -1, -1, -1, -1) {
        }
    };

    class OctreeNBLGPU{
        public:
            OctreeNBLGPU(){}
            OctreeNBLGPU(Vecc3<float> center, Vecc3<float> box_len);
            ~OctreeNBLGPU(){}
            void insert(Vecc3<float> positions, int pos_index, int node_index, std::vector<NodeGPU> &nodelist, int threadid);
            std::vector<NodeGPU> build_tree(std::vector<Vecc3<float>>& positions, std::vector<int> &pos_index, int threadid);
        private:
            Vecc3<float> center;
            Vecc3<float> size;
    };

    class OctreeSearchGPU : public SearchAlgorithm
    {
        public:
            //OctreeSearchGPU(Box& box, float r_cutoff, float skin);
            OctreeSearchGPU(const Box& box, const std::vector<std::vector<float>>& xyz, float r_cutoff);
            ~OctreeSearchGPU(){ delete[] listarray; }
            //void search(std::vector<std::vector<float>> &xyz) override;
            void search() override;
            void reset() override {}

            std::vector<std::vector<int>> get_particle_neighbor_info() override;
        private:
            Box _box;
            int nnebs;
            std::vector<std::vector<float>> _xyz;
            float _r_cutoff;
            int _natoms;
            int *listarray;
    };
}
