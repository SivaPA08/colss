#pragma once
#include "../include/compact.hpp"
#include "../include/eval.hpp"
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;
namespace py = pybind11;

inline double prod(string expr, py::dict scalar_dict, py::kwargs arrays) {
    using T = double;

    // --- Parse scalars --
    unordered_map<string, double> scalars;
    for (auto item : scalar_dict)
        scalars[item.first.cast<string>()] = item.second.cast<double>();

    // --- Parse arrays ---
    size_t n_vars = arrays.size();
    if (n_vars == 0 && scalars.empty())
        throw runtime_error("No variables provided");

    vector<string> names;
    vector<const double *> ptrs;
    vector<ssize_t> sizes;
    names.reserve(n_vars);
    ptrs.reserve(n_vars);
    sizes.reserve(n_vars);

    for (auto item : arrays) {
        string name = item.first.cast<string>();
        using arr_t =
            py::array_t<double, py::array::c_style | py::array::forcecast>;
        auto arr = item.second.cast<arr_t>();
        if (arr.ndim() != 1)
            throw runtime_error("Array '" + name + "' must be 1D");
        names.push_back(name);
        ptrs.push_back(arr.data());
        sizes.push_back(arr.size());
    }

    ssize_t n = n_vars > 0 ? sizes[0] : 1;
    for (size_t j = 1; j < n_vars; ++j)
        if (sizes[j] != n)
            throw runtime_error("Array size mismatch: '" + names[0] + "' has " +
                                to_string(n) + " elements but '" + names[j] +
                                "' has " + to_string(sizes[j]));

    double share = 1.0;

#pragma omp parallel reduction(* : share)
    {
        exprtk::symbol_table<T> symbol_table;
        exprtk::expression<T> expression;
        exprtk::parser<T> parser;

        vector<T> variables(n_vars, 0.0);
        for (size_t j = 0; j < n_vars; ++j)
            symbol_table.add_variable(names[j], variables[j]);

        // Scalars: registered once per thread, never updated in loop
        unordered_map<string, T> scalar_locals(scalars.begin(), scalars.end());
        for (auto &[name, val] : scalar_locals)
            symbol_table.add_variable(name, scalar_locals[name]);

        symbol_table.add_constants();
        expression.register_symbol_table(symbol_table);

        if (!parser.compile(expr, expression)) {
            string err = "Expression compile failed: ";
            for (size_t i = 0; i < parser.error_count(); ++i)
                err += parser.get_error(i).diagnostic + " ";
            throw runtime_error(err);
        }

#pragma omp for schedule(static)
        for (ssize_t i = 0; i < n; ++i) {
#pragma omp simd
            for (size_t j = 0; j < n_vars; ++j)
                variables[j] = ptrs[j][i];
            share *= expression.value();
        }
    }
    return share;
}
