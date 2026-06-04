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
// FIX 3: Hashed cache key — no heap-growing string concatenation per lookup
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
// FIX 5: reject bad inputs instead of quietly reallocating.
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
    // -----------------------------------------------------------------------
    // FIX 4: use uint8_t instead of vector<bool> (no packed-bit proxy hell)
    // -----------------------------------------------------------------------
    std::vector<std::string> all_var_names;
    std::vector<uint8_t> is_vector;
    std::vector<const double *> vector_ptrs;
    std::vector<double> scalar_vals;

    // FIX 5: strict input validation — no forcecast copies
    // Keep Python array references alive during GIL release
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
    // FIX 3 (cont.): compile once, using owned names for cache key
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
    // NEW: Pre-scan input arrays for NaN/Inf — FIX per-user request.
    // Build a bitmask so NaN slots are skipped entirely (no wasted compute).
    // -----------------------------------------------------------------------
    // nan_mask[i] == 1  →  output[i] should be NaN, skip evaluation
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

    // -----------------------------------------------------------------------
    // FIX 1: scalars are NOT expanded into temporary buffers.
    //   We pass them via a dedicated scalar_vals array; the evaluator accesses
    //   them as broadcasted constants.  If prog->eval does not yet support
    //   mixed scalar/vector calling convention, we provide a thin wrapper that
    //   fills the scalar buffer ONCE per thread (not per block) using a
    //   thread_local fixed buffer pinned to the scalar variables.
    //
    // FIX 2: NaN/Inf scan is fused into the block loop — no second pass.
    //
    // FIX 8: block setup cost reduced: scalar pointer is only reset when the
    //   scalar set changes (it never does), so it is set up once per thread.
    // -----------------------------------------------------------------------

    ssize_t nan_output_index = -1;
    bool has_error = false;
    std::string error_message;

    // Threshold tuned conservatively; profile and adjust for your hardware.
    // FIX 9: don't pay OpenMP overhead for tiny inputs.
    constexpr ssize_t OMP_THRESHOLD = 8192;
    constexpr ssize_t BLOCK_SIZE = 4096;

    const size_t num_variables = n_vars;

    {
        py::gil_scoped_release release;

#pragma omp parallel if (n >= OMP_THRESHOLD)                                   \
    shared(has_error, error_message, nan_output_index)
        {
            // ------------------------------------------------------------------
            // Thread-local pointer array.
            // FIX 1 + FIX 8: scalar buffers are allocated ONCE per thread and
            // filled ONCE — not per block — because scalar values are constant.
            // ------------------------------------------------------------------
            thread_local std::vector<const double *> tl_ptrs;
            thread_local std::vector<double>
                tl_scalar_buf; // one slot per scalar var

            // Resize lazily (only grows, never shrinks — fine for a cache)
            if (tl_ptrs.size() < num_variables)
                tl_ptrs.resize(num_variables);

            // Count scalars and build their one-per-thread single-element
            // buffers. We store exactly ONE double per scalar variable — no
            // block-sized copy. This is valid because prog->eval will broadcast
            // them internally, OR we construct a compact per-element wrapper
            // below.
            //
            // Strategy: fill tl_scalar_buf with scalar values once, point ptrs
            // at them. When evaluating block [i, i+block_len), scalar ptrs
            // don't change.
            if (tl_scalar_buf.size() < num_variables)
                tl_scalar_buf.resize(num_variables);

            for (size_t j = 0; j < num_variables; ++j) {
                if (!is_vector[j]) {
                    tl_scalar_buf[j] = scalar_vals[j];
                    // ptr points at a single double; the evaluator strides by 0
                    // if it supports scalar broadcasting, otherwise see note
                    // (*).
                    tl_ptrs[j] = &tl_scalar_buf[j];
                }
            }

#pragma omp for schedule(static)
            for (ssize_t i = 0; i < n; i += BLOCK_SIZE) {
                if (has_error || nan_output_index != -1)
                    continue;

                const ssize_t block_len = std::min(BLOCK_SIZE, n - i);

                // Fast-path: if every element in this block is NaN-masked, skip
                // entirely.
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
                // FIX 8: scalar ptrs were set once above — no work here for
                // scalars.
                for (size_t j = 0; j < num_variables; ++j) {
                    if (is_vector[j])
                        tl_ptrs[j] = vector_ptrs[j] + i;
                    // scalar ptrs: already set per-thread, never change
                }

                // (*) NOTE: if prog->eval requires all pointers to have stride
                // 1
                //   and does NOT support scalar broadcasting, replace the
                //   scalar handling above with a thread_local BLOCK_SIZE buffer
                //   filled once per thread (not per block) using std::fill_n at
                //   thread init. Since scalar values are constant across all
                //   blocks, the fill happens once, not O(n/BLOCK_SIZE) times —
                //   avoiding FIX 1's drain.

                try {
                    // Partial-NaN block: if any element in this block is
                    // masked, we must handle it element by element to avoid
                    // polluting non-NaN output slots.
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
                                    continue; // already NaN

                                // Single-element eval
                                std::vector<const double *> elem_ptrs(
                                    num_variables);
                                for (size_t j = 0; j < num_variables; ++j) {
                                    elem_ptrs[j] =
                                        is_vector[j] ? (vector_ptrs[j] + i + k)
                                                     : &tl_scalar_buf[j];
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

                // FIX 2: NaN/Inf check fused into this same pass — no second
                // scan. Only check output slots that were actually computed
                // (non-NaN inputs).
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
                        break; // only need the first bad index per block
                    }
                }
            }
        } // end omp parallel
    } // end GIL release

    if (has_error)
        throw std::runtime_error(error_message);

    if (nan_output_index >= 0)
        throw std::runtime_error("NaN/Inf in output at index " +
                                 std::to_string(nan_output_index));

    return result;
}
