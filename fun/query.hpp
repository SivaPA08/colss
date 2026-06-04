#pragma once

#include "../include/compact.hpp"
#include "../include/eval.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

struct CacheKey {
    std::string expr;
    std::vector<std::string> var_names;

    bool operator==(const CacheKey &o) const {
        return expr == o.expr && var_names == o.var_names;
    }
};

struct CacheKeyHash {
    std::size_t operator()(const CacheKey &k) const {
        std::size_t h = std::hash<std::string>{}(k.expr);
        for (const auto &v : k.var_names) {
            h ^= std::hash<std::string>{}(v) + 0x9e3779b97f4a7c15ULL +
                 (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct ProgramCache {
    std::mutex mutex;
    std::unordered_map<CacheKey, std::shared_ptr<const evalpp::Program>,
                       CacheKeyHash>
        map;

    std::shared_ptr<const evalpp::Program>
    get_or_compile(const std::string &expr,
                   const std::vector<std::string_view> &var_name_views,
                   const std::vector<std::string> &var_names_owned) {
        CacheKey key{expr, var_names_owned};

        {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = map.find(key);
            if (it != map.end()) {
                return it->second;
            }
        }

        auto prog = std::make_shared<evalpp::Program>(
            evalpp::compile(expr, var_name_views));

        {
            std::lock_guard<std::mutex> lock(mutex);
            auto [it, inserted] = map.emplace(std::move(key), prog);
            if (!inserted) {
                return it->second;
            }
        }

        return prog;
    }
};

inline ProgramCache &get_program_cache() {
    static ProgramCache cache;
    return cache;
}

static py::array_t<double>
require_double_c_contiguous(py::handle obj, const std::string &name) {
    if (!py::isinstance<py::array>(obj)) {
        throw std::runtime_error("Variable '" + name +
                                 "' is not a numpy array");
    }

    auto arr = obj.cast<py::array>();

    if (arr.dtype().kind() != 'f' || arr.dtype().itemsize() != 8) {
        throw std::runtime_error("Variable '" + name +
                                 "' must be float64 (got " +
                                 std::string(py::str(arr.dtype())) + ")");
    }

    if (!(arr.flags() & py::array::c_style)) {
        throw std::runtime_error("Variable '" + name +
                                 "' must be C-contiguous");
    }

    return arr.cast<py::array_t<double>>();
}

inline py::array_t<double> query(std::string expr, py::kwargs arrays) {
    std::vector<std::string> all_var_names;
    std::vector<const double *> vector_ptrs;
    std::vector<py::array_t<double>> handles;

    const size_t n_arrays = arrays.size();

    all_var_names.reserve(n_arrays);
    vector_ptrs.reserve(n_arrays);
    handles.reserve(n_arrays);

    for (auto item : arrays) {
        std::string name = item.first.cast<std::string>();
        auto arr = require_double_c_contiguous(item.second, name);

        all_var_names.push_back(name);
        vector_ptrs.push_back(arr.data());
        handles.push_back(std::move(arr));
    }

    ssize_t n = handles.empty() ? 1 : handles[0].size();
    for (size_t j = 1; j < handles.size(); ++j) {
        if (handles[j].size() != n) {
            throw std::runtime_error(
                "Array size mismatch: '" + all_var_names[0] + "' has " +
                std::to_string(n) + " elements but '" + all_var_names[j] +
                "' has " + std::to_string(handles[j].size()));
        }
    }

    std::vector<std::string_view> compile_var_names;
    compile_var_names.reserve(n_arrays);
    for (const auto &name : all_var_names) {
        compile_var_names.push_back(name);
    }

    std::shared_ptr<const evalpp::Program> prog;
    try {
        prog = get_program_cache().get_or_compile(expr, compile_var_names,
                                                  all_var_names);
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Expression failed to compile: ") +
                                 expr + " — " + e.what());
    }

    py::array_t<double> result(n);
    double *result_ptr = result.mutable_data();

    constexpr ssize_t OMP_THRESHOLD = 8192;
    constexpr ssize_t BLOCK_SIZE = 4096;

    const size_t num_variables = n_arrays;

    {
        py::gil_scoped_release release;

#pragma omp parallel if (n >= OMP_THRESHOLD)
        {
            thread_local std::vector<const double *> tl_ptrs;
            if (tl_ptrs.size() < num_variables) {
                tl_ptrs.resize(num_variables);
            }

#pragma omp for schedule(static)
            for (ssize_t i = 0; i < n; i += BLOCK_SIZE) {
                const ssize_t block_len = std::min(BLOCK_SIZE, n - i);

                for (size_t j = 0; j < num_variables; ++j) {
                    tl_ptrs[j] = vector_ptrs[j] + i;
                }

                prog->eval(block_len, result_ptr + i,
                           num_variables ? tl_ptrs.data() : nullptr);
            }
        }
    }

    return result;
}
