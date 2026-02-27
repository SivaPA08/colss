#pragma once
#include "../include/compact.hpp"
#include "../include/eval.hpp"
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

inline double var(std::string expr, py::dict scalar_dict, py::kwargs arrays) {
    using T = double;

    unordered_map<string, double> scalars;
    for (auto item : scalar_dict)
        scalars[item.first.cast<string>()] = item.second.cast<double>();

    size_t n_vars = arrays.size();

    vector<string> names;
    vector<const double *> ptrs;
    vector<ssize_t> sizes;
    vector<py::array_t<double, py::array::c_style | py::array::forcecast>>
        handles; // MODIFIED

    names.reserve(n_vars);
    ptrs.reserve(n_vars);
    sizes.reserve(n_vars);
    handles.reserve(n_vars); // MODIFIED

    for (auto item : arrays) {
        string name = item.first.cast<string>();
        using arr_t =
            py::array_t<double, py::array::c_style | py::array::forcecast>;
        auto arr = item.second.cast<arr_t>();
        if (arr.ndim() != 1)
            throw runtime_error("Array must be 1D");
        names.push_back(name);
        ptrs.push_back(arr.data());
        sizes.push_back(arr.size());
        handles.push_back(arr); // MODIFIED
    }

    ssize_t n = n_vars ? sizes[0] : 1;
    for (size_t j = 1; j < n_vars; ++j)
        if (sizes[j] != n)
            throw runtime_error("Array size mismatch");

    if (n <= 0)
        throw runtime_error("Invalid input size");

    vector<T> results(n);
    double sum = 0.0;
    ssize_t nan_index =
        -1; // MODIFIED: -1=ok, -2=compile error, >=0=NaN/Inf index

    { // MODIFIED: release GIL before OMP — nothing below touches Python objects
        py::gil_scoped_release release;

#pragma omp parallel
        {
            exprtk::symbol_table<T> symbol_table;
            exprtk::expression<T> expression;
            exprtk::parser<T>
                local_parser; // MODIFIED: per-thread, no data race

            vector<T> variables(n_vars, 0.0);

            for (size_t j = 0; j < n_vars; ++j)
                symbol_table.add_variable(names[j], variables[j]);

            unordered_map<string, T> scalar_locals(scalars.begin(),
                                                   scalars.end());
            for (auto &p : scalar_locals)
                symbol_table.add_variable(p.first, p.second);

            symbol_table.add_constants();
            expression.register_symbol_table(symbol_table);

            // MODIFIED: store sentinel instead of throwing inside OMP
            if (!local_parser.compile(expr, expression)) {
#pragma omp critical
                if (nan_index == -1)
                    nan_index = -2;
            }

#pragma omp for reduction(+ : sum)
            for (ssize_t i = 0; i < n; ++i) {
                if (nan_index != -1)
                    continue; // MODIFIED: cheap bail on error

                for (size_t j = 0; j < n_vars; ++j)
                    variables[j] = ptrs[j][i];

                double val = expression.value();

                // MODIFIED: store sentinel instead of throwing inside OMP
                if (!std::isfinite(val)) [[unlikely]] {
#pragma omp critical
                    if (nan_index == -1)
                        nan_index = i;
                    continue;
                }

                results[i] = val;
                sum += val;
            }
        }

        // second OMP loop is pure C++ — safe inside GIL release block
        double mean_val = sum / n;
        double var_sum = 0.0;

#pragma omp parallel for reduction(+ : var_sum)
        for (ssize_t i = 0; i < n; ++i) {
            double d = results[i] - mean_val;
            var_sum += d * d;
        }

        sum =
            var_sum / n; // MODIFIED: reuse sum to carry result out of GIL block

    } // MODIFIED: GIL reacquired here, handles safely destroyed after this

    // MODIFIED: all throws happen here, outside OMP and after GIL reacquired
    if (nan_index == -2)
        throw runtime_error("Expression failed to compile: " + expr);
    if (nan_index >= 0)
        throw runtime_error("NaN/Inf detected at index " +
                            to_string(nan_index));

    return sum; // holds var_sum / n
}
