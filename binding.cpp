#include "fun/mean.hpp"
#include "fun/median.hpp"
#include "fun/prod.hpp"
#include "fun/query.hpp"
#include "fun/sd.hpp"
#include "fun/sum.hpp"
#include "fun/var.hpp"
#include <pybind11/pybind11.h>
namespace py = pybind11;
PYBIND11_MODULE(_colss, m) {
    m.def("query", &query, py::arg("expr"), py::arg("scalar_dict"));
    m.def("sum", &sum, py::arg("expr"), py::arg("scalar_dict"));
    m.def("prod", &prod, py::arg("expr"), py::arg("scalar_dict"));
    m.def("mean", &mean, py::arg("expr"), py::arg("scalar_dict"));
    m.def("var", &var, py::arg("expr"), py::arg("scalar_dict"));
    m.def("sd", &sd, py::arg("expr"), py::arg("scalar_dict"));
    m.def("median", &median, py::arg("expr"), py::arg("scalar_dict"));
}
