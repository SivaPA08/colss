#pragma once
#include "../include/compact.hpp"
#include "../include/eval.hpp"
#include "../include/quickselect.hpp"
#include <cmath>
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
namespace py = pybind11;

inline double median(std::string expr, py::dict scalar_dict,
                     py::kwargs arrays) {
    using T = double;

    std::unordered_map<std::string, double> scalars;
    for (auto item : scalar_dict)
        scalars[item.first.cast<std::string>()] = item.second.cast<double>();

    size_t n_vars = arrays.size();

    std::vector<std::string> names;
    std::vector<const double *> ptrs;
    std::vector<ssize_t> sizes;
    std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>>
        handles;

    names.reserve(n_vars);
    ptrs.reserve(n_vars);
    sizes.reserve(n_vars);
    handles.reserve(n_vars);

    for (auto item : arrays) {
        std::string name = item.first.cast<std::string>();
        using arr_t =
            py::array_t<double, py::array::c_style | py::array::forcecast>;
        auto arr = item.second.cast<arr_t>();
        if (arr.ndim() != 1)
            throw std::runtime_error("Array must be 1D");
        names.push_back(name);
        ptrs.push_back(arr.data());
        sizes.push_back(arr.size());
        handles.push_back(arr);
    }

    ssize_t n = n_vars > 0 ? sizes[0] : 1;
    for (size_t j = 1; j < n_vars; ++j)
        if (sizes[j] != n)
            throw std::runtime_error("Array size mismatch");

    // intermediate buffer to hold evaluated results
    std::vector<double> evaluated(n);

    ssize_t nan_index = -1;

    {
        py::gil_scoped_release release;

#pragma omp parallel
        {
            exprtk::symbol_table<T> symbol_table;
            exprtk::expression<T> expression;
            exprtk::parser<T> local_parser;

            std::vector<T> variables(n_vars, 0.0);

            for (size_t j = 0; j < n_vars; ++j)
                symbol_table.add_variable(names[j], variables[j]);

            std::unordered_map<std::string, T> scalar_locals(scalars.begin(),
                                                             scalars.end());
            for (auto &[name, val] : scalar_locals)
                symbol_table.add_variable(name, scalar_locals[name]);

            symbol_table.add_constants();
            expression.register_symbol_table(symbol_table);

            if (!local_parser.compile(expr, expression)) {
#pragma omp critical
                if (nan_index == -1)
                    nan_index = -2;
            }

#pragma omp for schedule(static)
            for (ssize_t i = 0; i < n; ++i) {
                if (nan_index != -1)
                    continue;

                for (size_t j = 0; j < n_vars; ++j)
                    variables[j] = ptrs[j][i];

                T val = expression.value();

                if (!std::isfinite(val)) [[unlikely]] {
#pragma omp critical
                    if (nan_index == -1)
                        nan_index = i;
                    continue;
                }

                evaluated[i] = val;
            }
        }

        // now compute median on the evaluated buffer
        if (nan_index == -1) {
            long mid = n / 2;
            if (n % 2 == 1) {
                quickselect(evaluated.data(), 0, n - 1, mid);
            } else {
                quickselect(evaluated.data(), 0, n - 1, mid - 1);
                quickselect(evaluated.data(), 0, n - 1, mid);
            }
        }

    } // GIL reacquired here

    if (nan_index == -2)
        throw std::runtime_error("Expression failed to compile: " + expr);
    if (nan_index >= 0)
        throw std::runtime_error("NaN/Inf detected at index " +
                                 to_string(nan_index));

    long mid = n / 2;
    double val;
    if (n % 2 == 1) {
        val = evaluated[mid];
    } else {
        val = (evaluated[mid - 1] + evaluated[mid]) / 2.0;
    }

    return val;
}
