#include <pybind11/pybind11.h>
#include "fun/sum.hpp"

namespace py = pybind11;

PYBIND11_MODULE(colss, m) {
    py::class_<Sum>(m, "Sum")
        .def(py::init<>())
        .def("sum_double", &Sum::sum<double>)
        .def("sum_i64", &Sum::sum<std::int64_t>);
}
