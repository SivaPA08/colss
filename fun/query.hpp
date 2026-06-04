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
#include <mutex>
#include <memory>

using namespace std;
namespace py = pybind11;

struct ProgramCache {
    std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<const evalpp::Program>> map;

    std::shared_ptr<const evalpp::Program> get_or_compile(const std::string& expr, const std::vector<std::string_view>& var_names) {
        std::string cache_key = expr + "||";
        for (const auto& name : var_names) {
            cache_key += std::string(name) + ",";
        }

        {
            std::lock_guard<std::mutex> lock(mutex);
            auto it = map.find(cache_key);
            if (it != map.end()) {
                return it->second;
            }
        }

        auto prog = std::make_shared<evalpp::Program>(evalpp::compile(expr, var_names));

        {
            std::lock_guard<std::mutex> lock(mutex);
            map[cache_key] = prog;
        }

        return prog;
    }
};

inline ProgramCache& get_program_cache() {
    static ProgramCache cache;
    return cache;
}

inline py::array_t<double> query(std::string expr, py::dict scalar_dict,
                                 py::kwargs arrays) {
    // Collect all variable names and values/ptrs
    std::vector<std::string> all_var_names;
    std::vector<bool> is_vector;
    std::vector<const double *> vector_ptrs;
    std::vector<double> scalar_vals;

    // Keep Python array references alive during GIL release
    std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>>
        handles;

    all_var_names.reserve(arrays.size() + scalar_dict.size());
    is_vector.reserve(arrays.size() + scalar_dict.size());
    vector_ptrs.reserve(arrays.size() + scalar_dict.size());
    scalar_vals.reserve(arrays.size() + scalar_dict.size());
    handles.reserve(arrays.size());

    for (auto item : arrays) {
        std::string name = item.first.cast<std::string>();
        using arr_t =
            py::array_t<double, py::array::c_style | py::array::forcecast>;
        auto arr = item.second.cast<arr_t>();
        
        all_var_names.push_back(name);
        is_vector.push_back(true);
        vector_ptrs.push_back(arr.data());
        scalar_vals.push_back(0.0);
        handles.push_back(arr);
    }

    for (auto item : scalar_dict) {
        std::string name = item.first.cast<std::string>();
        double val = item.second.cast<double>();

        all_var_names.push_back(name);
        is_vector.push_back(false);
        vector_ptrs.push_back(nullptr);
        scalar_vals.push_back(val);
    }

    // Determine target array size n
    ssize_t n = handles.empty() ? 1 : handles[0].size();
    for (size_t j = 1; j < handles.size(); ++j) {
        if (handles[j].size() != n) {
            throw std::runtime_error("Array size mismatch");
        }
    }

    // Precompile the expression outside of parallel blocks (or retrieve from global cache)
    std::vector<std::string_view> compile_var_names;
    compile_var_names.reserve(all_var_names.size());
    for (const auto &name : all_var_names) {
        compile_var_names.push_back(name);
    }

    std::shared_ptr<const evalpp::Program> prog;
    try {
        prog = get_program_cache().get_or_compile(expr, compile_var_names);
    } catch (const std::exception &e) {
        throw std::runtime_error("Expression failed to compile: " + expr);
    }

    py::array_t<double> result(n);
    double *result_ptr = result.mutable_data();

    ssize_t nan_index = -1;
    bool has_error = false;
    std::string error_message = "";

    constexpr ssize_t BLOCK_SIZE = 4096;
    size_t num_variables = all_var_names.size();

    {
        py::gil_scoped_release release;

#pragma omp parallel if(n >= 8192) shared(has_error, error_message, nan_index)
        {
            // Thread-local preallocated buffers for variables
            thread_local std::vector<double> thread_local_scalar_buffers;
            thread_local std::vector<const double *> thread_local_ptrs;

            if (thread_local_scalar_buffers.size() < num_variables * BLOCK_SIZE) {
                thread_local_scalar_buffers.resize(num_variables * BLOCK_SIZE);
            }
            if (thread_local_ptrs.size() < num_variables) {
                thread_local_ptrs.resize(num_variables);
            }

#pragma omp for schedule(static)
            for (ssize_t i = 0; i < n; i += BLOCK_SIZE) {
                if (has_error || nan_index != -1)
                    continue;

                ssize_t block_len = std::min(BLOCK_SIZE, n - i);

                // Set up variable pointers for this block
                for (size_t j = 0; j < num_variables; ++j) {
                    if (is_vector[j]) {
                        thread_local_ptrs[j] = vector_ptrs[j] + i;
                    } else {
                        double val = scalar_vals[j];
                        double *buf =
                            &thread_local_scalar_buffers[j * BLOCK_SIZE];
                        std::fill_n(buf, block_len, val);
                        thread_local_ptrs[j] = buf;
                    }
                }

                // Evaluate block
                try {
                    prog->eval(block_len, result_ptr + i,
                               thread_local_ptrs.data());
                } catch (const std::exception &e) {
#pragma omp critical
                    {
                        if (!has_error) {
                            has_error = true;
                            error_message = e.what();
                        }
                    }
                    continue;
                }

                // Check for NaN/Inf in the evaluated block
                for (ssize_t k = 0; k < block_len; ++k) {
                    if (!std::isfinite(result_ptr[i + k])) {
#pragma omp critical
                        {
                            if (nan_index == -1 || i + k < nan_index) {
                                nan_index = i + k;
                            }
                        }
                        break;
                    }
                }
            }
        }
    }

    if (has_error) {
        throw std::runtime_error(error_message);
    }
    if (nan_index >= 0) {
        throw std::runtime_error("NaN/Inf detected at index " +
                                 std::to_string(nan_index));
    }

    return result;
}
