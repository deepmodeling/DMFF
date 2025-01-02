#include "read_lmp.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <array>


namespace dpnblist {

    void read_lmp(std::string filename, std::vector<std::vector<float>> &xyz, std::array<float,3> &box_len){
        
        std::ifstream file(filename);

        if (file.is_open()) {
            std::string line;

            // read atom numbers
            while (std::getline(file, line) && line.find("atoms") == std::string::npos);
            // for(int i = 0; i < 3; ++i){
            //     std::getline(file, line);
            // }
            std::istringstream atomLineStream(line);
            int numAtoms;
            atomLineStream >> numAtoms;
            xyz.resize(numAtoms,std::vector<float>(3,0.0f));
            
            // read box size
            while (std::getline(file, line) && line.find("xlo xhi") == std::string::npos);
            
            float lo, hi;
            for (int i = 0; i < 3; ++i) {
                std::istringstream boxLineStream(line);
                boxLineStream >> lo >> hi;
                box_len[i] = hi - lo;
                std::getline(file, line);
            }

            // jump exter lines
            while (std::getline(file, line) && line.find("Atoms") == std::string::npos);
            std::getline(file, line);
            // while (std::getline(file, line) && !line.empty());

            // read atom coordinates
            int id, type;
            float x, y, z;
            for (int i = 0; i < numAtoms; ++i) {
                std::getline(file, line);
                std::istringstream iss(line);
                iss >> id >> type >> x >> y >> z;
                xyz[i][0] = x;
                xyz[i][1] = y;
                xyz[i][2] = z;
            }

            file.close();
        } else {
            std::cout << "Unable to open file: " << filename << std::endl;
        }
    }

    void read_ref(std::string filename, std::vector<std::vector<size_t>> &ref_listArray){
            std::ifstream file(filename);
            if (!file.is_open()) {
                std::cout << "无法打开文件." << std::endl;
            }
            
            std::string line;
            int n = 0;
            while (std::getline(file, line)) {
                std::vector<std::string> tokens;
                std::stringstream ss(line);
                std::string token;
                while (ss >> token) {
                    tokens.push_back(token);
                }
                int len = tokens.size();
                if(len>1){
                    for(int i = 1; i < len; ++i){
                        ref_listArray[n].push_back(stof(tokens[i]));
                    }
                    n++;
                }
            }
            file.close();
    }
}