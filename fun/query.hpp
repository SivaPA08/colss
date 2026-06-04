#pragma once
#include "../include/compact.hpp"
#include "../include/eval.hpp"
#include <cmath>
#include <cstring>    // memcpy
#include <functional> // std::hash
#include <memory>
#include <mutex>
#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;
namespace py = pybind11;

// ---------------------------------------------------------------------------
// Hashed cache key — no heap-growing string concatenation per lookup
// ---------------------------------------------------------------------------
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
            // Combine hashes with a good mixer (boost-style)
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
            if (it != map.end())
                return it->second;
        }

        auto prog = std::make_shared<evalpp::Program>(
            evalpp::compile(expr, var_name_views));

        {
            std::lock_guard<std::mutex> lock(mutex);
            map.emplace(std::move(key), prog);
        }

        return prog;
    }
};

inline ProgramCache &get_program_cache() {
    static ProgramCache cache;
    return cache;
}

// ---------------------------------------------------------------------------
// Require strict double/C-contiguous arrays — no silent forcecast copies.
// ---------------------------------------------------------------------------
static py::array_t<double>
require_double_c_contiguous(py::handle obj, const std::string &name) {
    if (!py::isinstance<py::array>(obj))
        throw std::runtime_error("Variable '" + name +
                                 "' is not a numpy array");

    auto arr = obj.cast<py::array>();

    if (arr.dtype().kind() != 'f' || arr.dtype().itemsize() != 8)
        throw std::runtime_error("Variable '" + name +
                                 "' must be float64 (got " +
                                 std::string(py::str(arr.dtype())) + ")");

    if (!arr.flags() & py::array::c_style)
        throw std::runtime_error("Variable '" + name +
                                 "' must be C-contiguous");

    return arr.cast<py::array_t<double>>();
}

inline py::array_t<double> query(std::string expr, py::dict scalar_dict,
                                 py::kwargs arrays) {

    std::vector<std::string> all_var_names;
    std::vector<uint8_t> is_vector;
    std::vector<const double *> vector_ptrs;
    std::vector<double> scalar_vals;
    std::vector<py::array_t<double>> handles;

    const size_t n_arrays = arrays.size();
    const size_t n_scalars = scalar_dict.size();
    const size_t n_vars = n_arrays + n_scalars;

    all_var_names.reserve(n_vars);
    is_vector.reserve(n_vars);
    vector_ptrs.reserve(n_vars);
    scalar_vals.reserve(n_vars);
    handles.reserve(n_arrays);

    for (auto item : arrays) {
        std::string name = item.first.cast<std::string>();
        auto arr = require_double_c_contiguous(item.second, name);

        all_var_names.push_back(name);
        is_vector.push_back(1);
        vector_ptrs.push_back(arr.data());
        scalar_vals.push_back(0.0); // unused slot
        handles.push_back(std::move(arr));
    }

    for (auto item : scalar_dict) {
        std::string name = item.first.cast<std::string>();
        double val = item.second.cast<double>();

        all_var_names.push_back(name);
        is_vector.push_back(0);
        vector_ptrs.push_back(nullptr);
        scalar_vals.push_back(val);
    }

    // -----------------------------------------------------------------------
    // Size check
    // -----------------------------------------------------------------------
    ssize_t n = handles.empty() ? 1 : handles[0].size();
    for (size_t j = 1; j < handles.size(); ++j) {
        if (handles[j].size() != n)
            throw std::runtime_error(
                "Array size mismatch: '" + all_var_names[0] + "' has " +
                std::to_string(n) + " elements but '" + all_var_names[j] +
                "' has " + std::to_string(handles[j].size()));
    }

    // -----------------------------------------------------------------------
    // Compile once, using owned names for cache key
    // -----------------------------------------------------------------------
    std::vector<std::string_view> compile_var_names;
    compile_var_names.reserve(n_vars);
    for (const auto &name : all_var_names)
        compile_var_names.push_back(name);

    std::shared_ptr<const evalpp::Program> prog;
    try {
        prog = get_program_cache().get_or_compile(expr, compile_var_names,
                                                  all_var_names);
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("Expression failed to compile: ") +
                                 expr + " — " + e.what());
    }

    // -----------------------------------------------------------------------
    // Pre-scan input arrays for NaN/Inf.
    // Build a bitmask so NaN slots are skipped entirely (no wasted compute).
    // -----------------------------------------------------------------------
    std::vector<uint8_t> nan_mask(n, 0);
    bool any_nan_in_inputs = false;

    for (size_t j = 0; j < handles.size(); ++j) {
        const double *ptr = handles[j].data();
        for (ssize_t i = 0; i < n; ++i) {
            if (!std::isfinite(ptr[i])) {
                nan_mask[i] = 1;
                any_nan_in_inputs = true;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Output buffer
    // -----------------------------------------------------------------------
    py::array_t<double> result(n);
    double *result_ptr = result.mutable_data();

    // Pre-fill NaN slots so they're correct even if we skip them
    if (any_nan_in_inputs) {
        for (ssize_t i = 0; i < n; ++i) {
            if (nan_mask[i])
                result_ptr[i] = std::numeric_limits<double>::quiet_NaN();
        }
    }

    ssize_t nan_output_index = -1;
    bool has_error = false;
    std::string error_message;

    constexpr ssize_t OMP_THRESHOLD = 8192;
    constexpr ssize_t BLOCK_SIZE = 4096;

    const size_t num_variables = n_vars;

    {
        py::gil_scoped_release release;

#pragma omp parallel if (n >= OMP_THRESHOLD)                                   \
    shared(has_error, error_message, nan_output_index)
        {
            // ------------------------------------------------------------------
            // FIXED SCALAR BROADCASTING:
            // Allocate a full BLOCK_SIZE buffer for each scalar ONCE per
            // thread. This prevents out-of-bounds reads when prog->eval strides
            // by 1.
            // ------------------------------------------------------------------
            thread_local std::vector<const double *> tl_ptrs;
            thread_local std::vector<double> tl_scalar_bufs;

            if (tl_ptrs.size() < num_variables)
                tl_ptrs.resize(num_variables);

            if (tl_scalar_bufs.size() < num_variables * BLOCK_SIZE)
                tl_scalar_bufs.resize(num_variables * BLOCK_SIZE);

            // Fill the scalar buffers once per thread
            for (size_t j = 0; j < num_variables; ++j) {
                if (!is_vector[j]) {
                    double *buf_ptr = &tl_scalar_bufs[j * BLOCK_SIZE];
                    std::fill_n(buf_ptr, BLOCK_SIZE, scalar_vals[j]);
                    tl_ptrs[j] = buf_ptr;
                }
            }

#pragma omp for schedule(static)
            for (ssize_t i = 0; i < n; i += BLOCK_SIZE) {
                if (has_error || nan_output_index != -1)
                    continue;

                const ssize_t block_len = std::min(BLOCK_SIZE, n - i);

                // Fast-path: skip if whole block is masked
                if (any_nan_in_inputs) {
                    bool all_nan = true;
                    for (ssize_t k = 0; k < block_len; ++k) {
                        if (!nan_mask[i + k]) {
                            all_nan = false;
                            break;
                        }
                    }
                    if (all_nan)
                        continue;
                }

                // Set vector variable pointers for this block.
                // Scalar ptrs remain pointed to the start of their thread_local
                // buffer.
                for (size_t j = 0; j < num_variables; ++j) {
                    if (is_vector[j])
                        tl_ptrs[j] = vector_ptrs[j] + i;
                }

                try {
                    if (any_nan_in_inputs) {
                        bool block_has_nan = false;
                        for (ssize_t k = 0; k < block_len; ++k) {
                            if (nan_mask[i + k]) {
                                block_has_nan = true;
                                break;
                            }
                        }

                        if (block_has_nan) {
                            // Evaluate only non-NaN elements individually
                            for (ssize_t k = 0; k < block_len; ++k) {
                                if (nan_mask[i + k])
                                    continue;

                                std::vector<const double *> elem_ptrs(
                                    num_variables);
                                for (size_t j = 0; j < num_variables; ++j) {
                                    // For vectors we add k to get the specific
                                    // element. For scalars we can just point to
                                    // the start of the buffer.
                                    elem_ptrs[j] = is_vector[j]
                                                       ? (tl_ptrs[j] + k)
                                                       : tl_ptrs[j];
                                }
                                prog->eval(1, result_ptr + i + k,
                                           elem_ptrs.data());
                            }
                        } else {
                            prog->eval(block_len, result_ptr + i,
                                       tl_ptrs.data());
                        }
                    } else {
                        prog->eval(block_len, result_ptr + i, tl_ptrs.data());
                    }
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

                // NaN/Inf output validation
                for (ssize_t k = 0; k < block_len; ++k) {
                    if (any_nan_in_inputs && nan_mask[i + k])
                        continue;
                    if (!std::isfinite(result_ptr[i + k])) {
#pragma omp critical
                        {
                            if (nan_output_index == -1 ||
                                i + k < nan_output_index)
                                nan_output_index = i + k;
                        }
                        break;
                    }
                }
            }
        }
    }

    if (has_error)
        throw std::runtime_error(error_message);

    if (nan_output_index >= 0)
        throw std::runtime_error("NaN/Inf in output at index " +
                                 std::to_string(nan_output_index));

    return result;
}
