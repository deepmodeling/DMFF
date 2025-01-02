#include "nbList.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <omp.h>




namespace py = pybind11;
PYBIND11_MODULE(dpnblist, m) {
    m.doc() = "Neighbor List Module";
    m.def("get_max_threads", &omp_get_max_threads);
    m.def("set_num_threads", &omp_set_num_threads);

    py::class_<dpnblist::Box>(m, "Box")
        .def(py::init<std::vector<float>, std::vector<float>>());
        // .def(py::init<std::vector<float>, std::vector<float>>(), py::arg("lengths"), py::arg("angles") = std::vector<float>(90.0, 90.0, 90.0));
    
    
    py::class_<dpnblist::NeighborList>(m, "NeighborList")
        .def(py::init<std::string>())
        .def("build", &dpnblist::NeighborList::build)
        .def("get_neighbor_list", &dpnblist::NeighborList::get_neighbor_list)
        .def("get_neighbor_pair", &dpnblist::NeighborList::get_neighbor_pair)
        .def("get_neighbors", &dpnblist::NeighborList::get_neighbors);
}



