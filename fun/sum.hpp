#pragma once

#include <pybind11/numpy.h>
#include <cstdint>
#include <omp.h>

namespace py = pybind11;

class Sum {
public:
    template<typename T>
    T sum(py::array_t<T> arr) {
        auto buf = arr.template unchecked<2>();
        T share = 0;

        #pragma omp parallel for reduction(+:share)
        for (ssize_t i = 0; i < buf.shape(0); i++) {
            share += buf(i, 0);
        }

        return share;
    }
};