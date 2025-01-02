#include "../nbSchAlg.h"
#include <vector>

namespace dpnblist {

    class OctreeNode {
        public:
            std::array<float,3> center;
            std::array<float,3> size;
            OctreeNode* children[8];
            std::vector<std::array<float,3>> positions;
            std::vector<int> pos_index;

            OctreeNode(const std::array<float,3>& _center, std::array<float,3> _size) : center(_center), size(_size) {
                for (int i = 0; i < 8; ++i) {
                    children[i] = nullptr;
                }
            }

            ~OctreeNode() {
                for (int i = 0; i < 8; ++i) {
                    delete children[i]; // 递归地释放子节点的内存
            }
    }
    };

    class OctreeNBL
    {
        public:

            OctreeNBL(){}
            OctreeNBL(std::array<float,3> box_len, std::array<float,3> center);
            void insert(std::array<float,3> positions, int pos_index, OctreeNode *node);
            OctreeNode* constructor(const std::vector<std::array<float,3>>& positions, std::vector<int> &pos_index);

        private:
            std::array<float,3> _box_len;
            std::array<float,3> _center;
    };

    class OctreeSearchCPU : public SearchAlgorithm
    {
        public:

            //OctreeSearchCPU(Box &box, float r_cutoff, float skin);
            OctreeSearchCPU(const Box& box, const std::vector<std::vector<float>>& xyz, float r_cutoff);
            //void search(std::vector<std::vector<float>> &xyz) override;
            void search() override;
            std::array<int,3> aa(std::vector<std::array<int,3>> &a, int l, int r);
            std::array<int,3> aaa(std::vector<std::array<int,3>> &a, int l, int r, int c);
            std::vector<std::array<float,3>> period(const std::array<float,3> posi, float radius);
            void build(OctreeNode *rnode, const std::vector<std::array<float,3>>& positions, float distance);
            std::vector<int> query(const std::array<float,3>& position, float distance, OctreeNode *node);
            void reset() override {}
            //std::vector<std::vector<int>> get_particle_neighbor_info() override;
            std::vector<std::vector<int>> get_particle_neighbor_info() override;
        private:
            Box _box;
            std::vector<std::vector<float>> _xyz;
            float _r_cutoff;
            std::array<float,3> _box_len;
            int _natoms;
            //std::vector<std::vector<int>> particles;
            std::vector<std::vector<int>> particles;
    };
}
