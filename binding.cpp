#include "fun/query.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;
PYBIND11_MODULE(_colss, m) {
    m.def("query", &query, py::arg("expr"), py::arg("scalar_dict"));
}
