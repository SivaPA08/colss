
/*
    evalpp
    Copyright (c) 2026 Siva

    Licensed under the MIT License.

    Permission is hereby granted, free of charge, to any person obtaining
    a copy of this software and associated documentation files to deal
    in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute,
    sublicense, and/or sell copies of the Software.
*/

#pragma once
#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace evalpp {

// ---------------------------------------------------------------------------
// Vectorized Math Abstraction (xsimd, SLEEF, Intel SVML)
// ---------------------------------------------------------------------------
#if !defined(USE_XSIMD) && __has_include(<xsimd/xsimd.hpp>)
#define USE_XSIMD 1
#endif

#if !defined(USE_SLEEF) && __has_include(<sleef.h>)
#define USE_SLEEF 1
#endif

#if defined(USE_XSIMD)
#include <xsimd/xsimd.hpp>

inline void vec_sin(const double *in, double *out, std::size_t n) {
    using batch_type = xsimd::batch<double>;
    constexpr std::size_t simd_size = batch_type::size;
    std::size_t vec_size = n - (n % simd_size);
    for (std::size_t i = 0; i < vec_size; i += simd_size) {
        auto b = xsimd::load_unaligned(&in[i]);
        auto r = xsimd::sin(b);
        r.store_unaligned(&out[i]);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::sin(in[i]);
    }
}

inline void vec_cos(const double *in, double *out, std::size_t n) {
    using batch_type = xsimd::batch<double>;
    constexpr std::size_t simd_size = batch_type::size;
    std::size_t vec_size = n - (n % simd_size);
    for (std::size_t i = 0; i < vec_size; i += simd_size) {
        auto b = xsimd::load_unaligned(&in[i]);
        auto r = xsimd::cos(b);
        r.store_unaligned(&out[i]);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::cos(in[i]);
    }
}

inline void vec_log(const double *in, double *out, std::size_t n) {
    using batch_type = xsimd::batch<double>;
    constexpr std::size_t simd_size = batch_type::size;
    std::size_t vec_size = n - (n % simd_size);
    for (std::size_t i = 0; i < vec_size; i += simd_size) {
        auto b = xsimd::load_unaligned(&in[i]);
        auto r = xsimd::log(b);
        r.store_unaligned(&out[i]);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::log(in[i]);
    }
}

inline void vec_pow(const double *in_base, const double *in_exp, double *out,
                    std::size_t n) {
    using batch_type = xsimd::batch<double>;
    constexpr std::size_t simd_size = batch_type::size;
    std::size_t vec_size = n - (n % simd_size);
    for (std::size_t i = 0; i < vec_size; i += simd_size) {
        auto b = xsimd::load_unaligned(&in_base[i]);
        auto e = xsimd::load_unaligned(&in_exp[i]);
        auto r = xsimd::pow(b, e);
        r.store_unaligned(&out[i]);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::pow(in_base[i], in_exp[i]);
    }
}

#elif defined(USE_SVML)

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
#include <immintrin.h>

extern "C" {
__m128d _mm_sin_pd(__m128d);
__m128d _mm_cos_pd(__m128d);
__m128d _mm_log_pd(__m128d);
__m128d _mm_pow_pd(__m128d, __m128d);

__m256d _mm256_sin_pd(__m256d);
__m256d _mm256_cos_pd(__m256d);
__m256d _mm256_log_pd(__m256d);
__m256d _mm256_pow_pd(__m256d, __m256d);

__m512d _mm512_sin_pd(__m512d);
__m512d _mm512_cos_pd(__m512d);
__m512d _mm512_log_pd(__m512d);
__m512d _mm512_pow_pd(__m512d, __m512d);
}

#if defined(__AVX512F__)
inline void vec_sin(const double *in, double *out, std::size_t n) {
    std::size_t vec_size = n - (n % 8);
    for (std::size_t i = 0; i < vec_size; i += 8) {
        __m512d b = _mm512_loadu_pd(&in[i]);
        __m512d r = _mm512_sin_pd(b);
        _mm512_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::sin(in[i]);
    }
}
inline void vec_cos(const double *in, double *out, std::size_t n) {
    std::size_t vec_size = n - (n % 8);
    for (std::size_t i = 0; i < vec_size; i += 8) {
        __m512d b = _mm512_loadu_pd(&in[i]);
        __m512d r = _mm512_cos_pd(b);
        _mm512_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::cos(in[i]);
    }
}
inline void vec_log(const double *in, double *out, std::size_t n) {
    std::size_t vec_size = n - (n % 8);
    for (std::size_t i = 0; i < vec_size; i += 8) {
        __m512d b = _mm512_loadu_pd(&in[i]);
        __m512d r = _mm512_log_pd(b);
        _mm512_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::log(in[i]);
    }
}
inline void vec_pow(const double *in_base, const double *in_exp, double *out,
                    std::size_t n) {
    std::size_t vec_size = n - (n % 8);
    for (std::size_t i = 0; i < vec_size; i += 8) {
        __m512d b = _mm512_loadu_pd(&in_base[i]);
        __m512d e = _mm512_loadu_pd(&in_exp[i]);
        __m512d r = _mm512_pow_pd(b, e);
        _mm512_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::pow(in_base[i], in_exp[i]);
    }
}
#elif defined(__AVX2__) || defined(__AVX__)
inline void vec_sin(const double *in, double *out, std::size_t n) {
    std::size_t vec_size = n - (n % 4);
    for (std::size_t i = 0; i < vec_size; i += 4) {
        __m256d b = _mm256_loadu_pd(&in[i]);
        __m256d r = _mm256_sin_pd(b);
        _mm256_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::sin(in[i]);
    }
}
inline void vec_cos(const double *in, double *out, std::size_t n) {
    std::size_t vec_size = n - (n % 4);
    for (std::size_t i = 0; i < vec_size; i += 4) {
        __m256d b = _mm256_loadu_pd(&in[i]);
        __m256d r = _mm256_cos_pd(b);
        _mm256_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::cos(in[i]);
    }
}
inline void vec_log(const double *in, double *out, std::size_t n) {
    std::size_t vec_size = n - (n % 4);
    for (std::size_t i = 0; i < vec_size; i += 4) {
        __m256d b = _mm256_loadu_pd(&in[i]);
        __m256d r = _mm256_log_pd(b);
        _mm256_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::log(in[i]);
    }
}
inline void vec_pow(const double *in_base, const double *in_exp, double *out,
                    std::size_t n) {
    std::size_t vec_size = n - (n % 4);
    for (std::size_t i = 0; i < vec_size; i += 4) {
        __m256d b = _mm256_loadu_pd(&in_base[i]);
        __m256d e = _mm256_loadu_pd(&in_exp[i]);
        __m256d r = _mm256_pow_pd(b, e);
        _mm256_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::pow(in_base[i], in_exp[i]);
    }
}
#elif defined(__SSE2__)
inline void vec_sin(const double *in, double *out, std::size_t n) {
    std::size_t vec_size = n - (n % 2);
    for (std::size_t i = 0; i < vec_size; i += 2) {
        __m128d b = _mm_loadu_pd(&in[i]);
        __m128d r = _mm_sin_pd(b);
        _mm_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::sin(in[i]);
    }
}
inline void vec_cos(const double *in, double *out, std::size_t n) {
    std::size_t vec_size = n - (n % 2);
    for (std::size_t i = 0; i < vec_size; i += 2) {
        __m128d b = _mm_loadu_pd(&in[i]);
        __m128d r = _mm_cos_pd(b);
        _mm_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::cos(in[i]);
    }
}
inline void vec_log(const double *in, double *out, std::size_t n) {
    std::size_t vec_size = n - (n % 2);
    for (std::size_t i = 0; i < vec_size; i += 2) {
        __m128d b = _mm_loadu_pd(&in[i]);
        __m128d r = _mm_log_pd(b);
        _mm_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::log(in[i]);
    }
}
inline void vec_pow(const double *in_base, const double *in_exp, double *out,
                    std::size_t n) {
    std::size_t vec_size = n - (n % 2);
    for (std::size_t i = 0; i < vec_size; i += 2) {
        __m128d b = _mm_loadu_pd(&in_base[i]);
        __m128d e = _mm_loadu_pd(&in_exp[i]);
        __m128d r = _mm_pow_pd(b, e);
        _mm_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::pow(in_base[i], in_exp[i]);
    }
}
#else
#define SVML_FALLBACK 1
#endif
#else
#define SVML_FALLBACK 1
#endif

#endif

#if defined(USE_SLEEF)

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
#include <immintrin.h>

#if __has_include(<sleef.h>)
#include <sleef.h>
#else
extern "C" {
__m128d Sleef_sind2_u10sse2(__m128d);
__m128d Sleef_cosd2_u10sse2(__m128d);
__m128d Sleef_logd2_u10sse2(__m128d);
__m128d Sleef_powd2_u10sse2(__m128d, __m128d);

__m256d Sleef_sind4_u10avx2(__m256d);
__m256d Sleef_cosd4_u10avx2(__m256d);
__m256d Sleef_logd4_u10avx2(__m256d);
__m256d Sleef_powd4_u10avx2(__m256d, __m256d);

__m512d Sleef_sind8_u10avx512f(__m512d);
__m512d Sleef_cosd8_u10avx512f(__m512d);
__m512d Sleef_logd8_u10avx512f(__m512d);
__m512d Sleef_powd8_u10avx512f(__m512d, __m512d);
}
#endif

#if defined(__AVX512F__)
inline void vec_sin(const double *in, double *out, std::size_t n) {
    std::size_t vec_size = n - (n % 8);
    for (std::size_t i = 0; i < vec_size; i += 8) {
        __m512d b = _mm512_loadu_pd(&in[i]);
        __m512d r = Sleef_sind8_u10avx512f(b);
        _mm512_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::sin(in[i]);
    }
}
inline void vec_cos(const double *in, double *out, std::size_t n) {
    std::size_t vec_size = n - (n % 8);
    for (std::size_t i = 0; i < vec_size; i += 8) {
        __m512d b = _mm512_loadu_pd(&in[i]);
        __m512d r = Sleef_cosd8_u10avx512f(b);
        _mm512_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::cos(in[i]);
    }
}
inline void vec_log(const double *in, double *out, std::size_t n) {
    std::size_t vec_size = n - (n % 8);
    for (std::size_t i = 0; i < vec_size; i += 8) {
        __m512d b = _mm512_loadu_pd(&in[i]);
        __m512d r = Sleef_logd8_u10avx512f(b);
        _mm512_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::log(in[i]);
    }
}
inline void vec_pow(const double *in_base, const double *in_exp, double *out,
                    std::size_t n) {
    std::size_t vec_size = n - (n % 8);
    for (std::size_t i = 0; i < vec_size; i += 8) {
        __m512d b = _mm512_loadu_pd(&in_base[i]);
        __m512d e = _mm512_loadu_pd(&in_exp[i]);
        __m512d r = Sleef_powd8_u10avx512f(b, e);
        _mm512_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::pow(in_base[i], in_exp[i]);
    }
}
#elif defined(__AVX2__) || defined(__AVX__)
inline void vec_sin(const double *in, double *out, std::size_t n) {
    std::size_t vec_size = n - (n % 4);
    for (std::size_t i = 0; i < vec_size; i += 4) {
        __m256d b = _mm256_loadu_pd(&in[i]);
        __m256d r = Sleef_sind4_u10avx2(b);
        _mm256_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::sin(in[i]);
    }
}
inline void vec_cos(const double *in, double *out, std::size_t n) {
    std::size_t vec_size = n - (n % 4);
    for (std::size_t i = 0; i < vec_size; i += 4) {
        __m256d b = _mm256_loadu_pd(&in[i]);
        __m256d r = Sleef_cosd4_u10avx2(b);
        _mm256_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::cos(in[i]);
    }
}
inline void vec_log(const double *in, double *out, std::size_t n) {
    std::size_t vec_size = n - (n % 4);
    for (std::size_t i = 0; i < vec_size; i += 4) {
        __m256d b = _mm256_loadu_pd(&in[i]);
        __m256d r = Sleef_logd4_u10avx2(b);
        _mm256_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::log(in[i]);
    }
}
inline void vec_pow(const double *in_base, const double *in_exp, double *out,
                    std::size_t n) {
    std::size_t vec_size = n - (n % 4);
    for (std::size_t i = 0; i < vec_size; i += 4) {
        __m256d b = _mm256_loadu_pd(&in_base[i]);
        __m256d e = _mm256_loadu_pd(&in_exp[i]);
        __m256d r = Sleef_powd4_u10avx2(b, e);
        _mm256_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::pow(in_base[i], in_exp[i]);
    }
}
#elif defined(__SSE2__)
inline void vec_sin(const double *in, double *out, std::size_t n) {
    std::size_t vec_size = n - (n % 2);
    for (std::size_t i = 0; i < vec_size; i += 2) {
        __m128d b = _mm_loadu_pd(&in[i]);
        __m128d r = Sleef_sind2_u10sse2(b);
        _mm_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::sin(in[i]);
    }
}
inline void vec_cos(const double *in, double *out, std::size_t n) {
    std::size_t vec_size = n - (n % 2);
    for (std::size_t i = 0; i < vec_size; i += 2) {
        __m128d b = _mm_loadu_pd(&in[i]);
        __m128d r = Sleef_cosd2_u10sse2(b);
        _mm_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::cos(in[i]);
    }
}
inline void vec_log(const double *in, double *out, std::size_t n) {
    std::size_t vec_size = n - (n % 2);
    for (std::size_t i = 0; i < vec_size; i += 2) {
        __m128d b = _mm_loadu_pd(&in[i]);
        __m128d r = Sleef_logd2_u10sse2(b);
        _mm_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::log(in[i]);
    }
}
inline void vec_pow(const double *in_base, const double *in_exp, double *out,
                    std::size_t n) {
    std::size_t vec_size = n - (n % 2);
    for (std::size_t i = 0; i < vec_size; i += 2) {
        __m128d b = _mm_loadu_pd(&in_base[i]);
        __m128d e = _mm_loadu_pd(&in_exp[i]);
        __m128d r = Sleef_powd2_u10sse2(b, e);
        _mm_storeu_pd(&out[i], r);
    }
    for (std::size_t i = vec_size; i < n; ++i) {
        out[i] = std::pow(in_base[i], in_exp[i]);
    }
}
#else
#define SLEEF_FALLBACK 1
#endif
#else
#define SLEEF_FALLBACK 1
#endif

#endif

#if !defined(USE_XSIMD) && !(defined(USE_SVML) && !defined(SVML_FALLBACK)) &&  \
    !(defined(USE_SLEEF) && !defined(SLEEF_FALLBACK))
inline void vec_sin(const double *in, double *out, std::size_t n) {
#pragma omp simd
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = std::sin(in[i]);
    }
}

inline void vec_cos(const double *in, double *out, std::size_t n) {
#pragma omp simd
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = std::cos(in[i]);
    }
}

inline void vec_log(const double *in, double *out, std::size_t n) {
#pragma omp simd
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = std::log(in[i]);
    }
}

inline void vec_pow(const double *in_base, const double *in_exp, double *out,
                    std::size_t n) {
#pragma omp simd
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = std::pow(in_base[i], in_exp[i]);
    }
}
#endif

enum class TokenType {
    End,
    Number,
    Identifier,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Amp,
    Pipe,
    Caret,
    Tilde,
    ShiftLeft,
    ShiftRight,
    EqualEqual,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    AndAnd,
    OrOr,
    Not,
    Question,
    Colon,
    LParen,
    RParen,
    Comma
};

struct Token {
    TokenType type{TokenType::End};
    std::string_view text;
    double number{0.0};
    std::size_t pos{0};
};

inline bool iequals(std::string_view a, std::string_view b) noexcept {
    if (a.size() != b.size())
        return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) !=
            std::tolower(static_cast<unsigned char>(b[i]))) {
            return false;
        }
    }
    return true;
}

class Lexer {
  public:
    explicit Lexer(std::string_view input) : input_(input) {}

    std::vector<Token> tokenize() {
        std::vector<Token> tokens;
        while (true) {
            while (!eof() && std::isspace(static_cast<unsigned char>(peek())))
                advance();
            if (eof()) {
                tokens.push_back({TokenType::End, "", 0.0, pos_});
                break;
            }
            const char c = peek();
            const std::size_t start = pos_;

            if (std::isdigit(static_cast<unsigned char>(c)) ||
                (c == '.' && pos_ + 1 < input_.size() &&
                 std::isdigit(static_cast<unsigned char>(input_[pos_ + 1])))) {
                tokens.push_back(scanNum());
                continue;
            }
            if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
                tokens.push_back(scanId());
                continue;
            }
            switch (c) {
            case '+':
                advance();
                tokens.push_back({TokenType::Plus, "+", 0.0, start});
                break;
            case '-':
                advance();
                tokens.push_back({TokenType::Minus, "-", 0.0, start});
                break;
            case '*':
                advance();
                tokens.push_back({TokenType::Star, "*", 0.0, start});
                break;
            case '/':
                advance();
                tokens.push_back({TokenType::Slash, "/", 0.0, start});
                break;
            case '%':
                advance();
                tokens.push_back({TokenType::Percent, "%", 0.0, start});
                break;
            case '~':
                advance();
                tokens.push_back({TokenType::Tilde, "~", 0.0, start});
                break;
            case '?':
                advance();
                tokens.push_back({TokenType::Question, "?", 0.0, start});
                break;
            case ':':
                advance();
                tokens.push_back({TokenType::Colon, ":", 0.0, start});
                break;
            case '(':
                advance();
                tokens.push_back({TokenType::LParen, "(", 0.0, start});
                break;
            case ')':
                advance();
                tokens.push_back({TokenType::RParen, ")", 0.0, start});
                break;
            case ',':
                advance();
                tokens.push_back({TokenType::Comma, ",", 0.0, start});
                break;
            case '&':
                if (match("&&"))
                    tokens.push_back({TokenType::AndAnd, "&&", 0.0, start});
                else {
                    advance();
                    tokens.push_back({TokenType::Amp, "&", 0.0, start});
                }
                break;
            case '|':
                if (match("||"))
                    tokens.push_back({TokenType::OrOr, "||", 0.0, start});
                else {
                    advance();
                    tokens.push_back({TokenType::Pipe, "|", 0.0, start});
                }
                break;
            case '^':
                advance();
                tokens.push_back({TokenType::Caret, "^", 0.0, start});
                break;
            case '!':
                if (match("!="))
                    tokens.push_back({TokenType::NotEqual, "!=", 0.0, start});
                else {
                    advance();
                    tokens.push_back({TokenType::Not, "!", 0.0, start});
                }
                break;
            case '=':
                if (match("=="))
                    tokens.push_back({TokenType::EqualEqual, "==", 0.0, start});
                else
                    throw std::runtime_error(
                        "Lexer error: Single '=' not supported, use '=='");
                break;
            case '<':
                if (match("<<"))
                    tokens.push_back({TokenType::ShiftLeft, "<<", 0.0, start});
                else if (match("<="))
                    tokens.push_back({TokenType::LessEqual, "<=", 0.0, start});
                else {
                    advance();
                    tokens.push_back({TokenType::Less, "<", 0.0, start});
                }
                break;
            case '>':
                if (match(">>"))
                    tokens.push_back({TokenType::ShiftRight, ">>", 0.0, start});
                else if (match(">="))
                    tokens.push_back(
                        {TokenType::GreaterEqual, ">=", 0.0, start});
                else {
                    advance();
                    tokens.push_back({TokenType::Greater, ">", 0.0, start});
                }
                break;
            default:
                throw std::runtime_error("Lexer error: Unknown character");
            }
        }
        return tokens;
    }

  private:
    std::string_view input_;
    std::size_t pos_{0};

    bool eof() const { return pos_ >= input_.size(); }
    char peek() const { return input_[pos_]; }
    void advance(std::size_t n = 1) { pos_ += n; }

    bool match(std::string_view s) {
        if (input_.size() - pos_ >= s.size() &&
            input_.substr(pos_, s.size()) == s) {
            advance(s.size());
            return true;
        }
        return false;
    }

    Token scanNum() {
        std::size_t start = pos_;
        if (peek() == '.')
            advance();
        while (!eof() && std::isdigit(static_cast<unsigned char>(peek())))
            advance();
        if (!eof() && peek() == '.') {
            advance();
            while (!eof() && std::isdigit(static_cast<unsigned char>(peek())))
                advance();
        }
        if (!eof() && (peek() == 'e' || peek() == 'E')) {
            advance();
            if (!eof() && (peek() == '+' || peek() == '-'))
                advance();
            while (!eof() && std::isdigit(static_cast<unsigned char>(peek())))
                advance();
        }

        double val = 0.0;
        auto [ptr, ec] =
            std::from_chars(input_.data() + start, input_.data() + pos_, val);
        if (ec != std::errc()) {
            throw std::runtime_error("Lexer error: Invalid number format");
        }

        return {TokenType::Number, input_.substr(start, pos_ - start), val,
                start};
    }

    Token scanId() {
        std::size_t start = pos_;
        advance();
        while (!eof() && (std::isalnum(static_cast<unsigned char>(peek())) ||
                          peek() == '_'))
            advance();
        std::string_view text = input_.substr(start, pos_ - start);
        if (iequals(text, "and")) {
            return {TokenType::AndAnd, text, 0.0, start};
        }
        if (iequals(text, "or")) {
            return {TokenType::OrOr, text, 0.0, start};
        }
        if (iequals(text, "not")) {
            return {TokenType::Not, text, 0.0, start};
        }
        return {TokenType::Identifier, text, 0.0, start};
    }
};

enum class Op {
    PushC,
    LoadVar,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
    Not,
    Tilde,
    JmpZ,
    Jmp,
    Call,
    EndProg
};

enum class Builtin {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sqrt,
    Pow,
    Log,
    Log10,
    Exp,
    Abs,
    Floor,
    Ceil,
    Round,
    Min,
    Max
};

struct Inst {
    Op op;
    uint32_t arg;
};

// -------------------------------------------------------------------------
// Case-insensitive hash and equality for var_lookup.
// Both carry is_transparent so that C++20 unordered_map::find() can accept
// a std::string_view key directly without constructing a std::string.
// -------------------------------------------------------------------------
struct CaseInsensitiveHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view sv) const noexcept {
        // FNV-1a over the lowercased bytes — no heap allocation.
        std::size_t h = 14695981039346656037ULL;
        for (unsigned char c : sv) {
            h ^= static_cast<std::size_t>(
                std::tolower(static_cast<unsigned char>(c)));
            h *= 1099511628211ULL;
        }
        return h;
    }
    // Overload for the stored std::string keys so the map can hash them.
    std::size_t operator()(const std::string &s) const noexcept {
        return (*this)(std::string_view{s});
    }
};

struct CaseInsensitiveEqual {
    using is_transparent = void;

    bool operator()(std::string_view a, std::string_view b) const noexcept {
        if (a.size() != b.size())
            return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (std::tolower(static_cast<unsigned char>(a[i])) !=
                std::tolower(static_cast<unsigned char>(b[i])))
                return false;
        }
        return true;
    }
    bool operator()(const std::string &a, std::string_view b) const noexcept {
        return (*this)(std::string_view{a}, b);
    }
    bool operator()(std::string_view a, const std::string &b) const noexcept {
        return (*this)(a, std::string_view{b});
    }
    bool operator()(const std::string &a, const std::string &b) const noexcept {
        return (*this)(std::string_view{a}, std::string_view{b});
    }
};

class Program {
  public:
    std::vector<Inst> code;
    std::vector<double> consts;
    std::vector<std::string> var_names;

    // -----------------------------------------------------------------------
    // Precompiled variable storage.
    //   values[slot]  — written by set(), read by eval(values.data()).
    //   var_lookup    — maps variable name (case-insensitive) → slot index.
    //
    // Both are mutable so that set() / eval() are callable on a const Program,
    // e.g. when the Program is stored as a const in a larger structure.
    // -----------------------------------------------------------------------
    mutable std::vector<double> values;

    std::unordered_map<std::string, uint32_t, CaseInsensitiveHash,
                       CaseInsensitiveEqual>
        var_lookup;

    // -----------------------------------------------------------------------
    // set(name, value)
    //   Stores a variable value into the precompiled slot.
    //   • O(1) hash lookup — no string compares in the hot path.
    //   • No heap allocations.
    //   • No recompilation.
    // -----------------------------------------------------------------------
    void set(std::string_view name, double value) const {
        // find() uses the transparent hash/equal, so the std::string_view is
        // compared directly against the stored std::string keys without any
        // temporary std::string being constructed.
        auto it = var_lookup.find(std::string(name));
        if (it == var_lookup.end())
            throw std::runtime_error("evalpp::Program::set — unknown variable");
        values[it->second] = value;
    }

    // -----------------------------------------------------------------------
    // eval()  — scripting API convenience overload.
    //   Uses the values[] vector populated by set().
    //   No allocations; delegates straight to the fast VM path below.
    // -----------------------------------------------------------------------
    double eval() const { return eval(values.data()); }

    // -----------------------------------------------------------------------
    // eval(n, out, vars)  — vectorized VM fast path.
    // -----------------------------------------------------------------------
    void eval(std::size_t n, double *out, const double *const *vars) const {
        if (n == 0)
            return;

        bool has_branches = false;
        for (const auto &inst : code) {
            if (inst.op == Op::JmpZ || inst.op == Op::Jmp) {
                has_branches = true;
                break;
            }
        }
        if (has_branches) {
            thread_local std::vector<double> current_vars;
            current_vars.resize(var_names.size());
            for (std::size_t i = 0; i < n; ++i) {
                for (std::size_t j = 0; j < var_names.size(); ++j) {
                    current_vars[j] = vars[j][i];
                }
                out[i] = eval(current_vars.data());
            }
            return;
        }

        thread_local std::vector<double> stack_mem;
        if (stack_mem.size() < 256 * n) {
            stack_mem.resize(256 * n);
        }
        double *sp_storage = stack_mem.data();

        double *stack[256];
        double **sp = stack;
        const Inst *ip = code.data();

#if defined(__GNUC__) || defined(__clang__)
        static const void *dispatch_table[] = {
            &&op_pushc, &&op_loadvar, &&op_add,  &&op_sub, &&op_mul,  &&op_div,
            &&op_mod,   &&op_and,     &&op_or,   &&op_xor, &&op_shl,  &&op_shr,
            &&op_eq,    &&op_neq,     &&op_lt,   &&op_le,  &&op_gt,   &&op_ge,
            &&op_not,   &&op_tilde,   &&op_jmpz, &&op_jmp, &&op_call, &&op_end};

#define DISPATCH()                                                             \
    do {                                                                       \
        goto *dispatch_table[static_cast<uint32_t>(ip->op)];                   \
    } while (0)

        DISPATCH();

    op_pushc: {
        double val = consts[ip->arg];
        double *dest = sp_storage + (sp - stack) * n;
        std::fill_n(dest, n, val);
        *sp++ = dest;
        ++ip;
        DISPATCH();
    }
    op_loadvar: {
        *sp++ = const_cast<double *>(vars[ip->arg]);
        ++ip;
        DISPATCH();
    }
    op_add: {
        double *r = *--sp;
        double *l = *--sp;
        double *dest = sp_storage + (sp - stack) * n;
#pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            dest[i] = l[i] + r[i];
        }
        *sp++ = dest;
        ++ip;
        DISPATCH();
    }
    op_sub: {
        double *r = *--sp;
        double *l = *--sp;
        double *dest = sp_storage + (sp - stack) * n;
#pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            dest[i] = l[i] - r[i];
        }
        *sp++ = dest;
        ++ip;
        DISPATCH();
    }
    op_mul: {
        double *r = *--sp;
        double *l = *--sp;
        double *dest = sp_storage + (sp - stack) * n;
#pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            dest[i] = l[i] * r[i];
        }
        *sp++ = dest;
        ++ip;
        DISPATCH();
    }
    op_div: {
        double *r = *--sp;
        double *l = *--sp;
        for (std::size_t i = 0; i < n; ++i) {
            if (r[i] == 0.0)
                throw std::runtime_error("Division by zero");
        }
        double *dest = sp_storage + (sp - stack) * n;
#pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            dest[i] = l[i] / r[i];
        }
        *sp++ = dest;
        ++ip;
        DISPATCH();
    }
    op_mod: {
        double *r = *--sp;
        double *l = *--sp;
        for (std::size_t i = 0; i < n; ++i) {
            if (r[i] == 0.0)
                throw std::runtime_error("Modulo by zero");
        }
        double *dest = sp_storage + (sp - stack) * n;
#pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            dest[i] = std::fmod(l[i], r[i]);
        }
        *sp++ = dest;
        ++ip;
        DISPATCH();
    }
    op_and: {
        double *r = *--sp;
        double *l = *--sp;
        double *dest = sp_storage + (sp - stack) * n;
#pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            dest[i] = static_cast<double>(static_cast<int64_t>(l[i]) &
                                          static_cast<int64_t>(r[i]));
        }
        *sp++ = dest;
        ++ip;
        DISPATCH();
    }
    op_or: {
        double *r = *--sp;
        double *l = *--sp;
        double *dest = sp_storage + (sp - stack) * n;
#pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            dest[i] = static_cast<double>(static_cast<int64_t>(l[i]) |
                                          static_cast<int64_t>(r[i]));
        }
        *sp++ = dest;
        ++ip;
        DISPATCH();
    }
    op_xor: {
        double *r = *--sp;
        double *l = *--sp;
        double *dest = sp_storage + (sp - stack) * n;
        evalpp::vec_pow(l, r, dest, n);
        *sp++ = dest;
        ++ip;
        DISPATCH();
    }
    op_shl: {
        double *r = *--sp;
        double *l = *--sp;
        double *dest = sp_storage + (sp - stack) * n;
#pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            dest[i] = static_cast<double>(static_cast<int64_t>(l[i])
                                          << static_cast<int64_t>(r[i]));
        }
        *sp++ = dest;
        ++ip;
        DISPATCH();
    }
    op_shr: {
        double *r = *--sp;
        double *l = *--sp;
        double *dest = sp_storage + (sp - stack) * n;
#pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            dest[i] = static_cast<double>(static_cast<int64_t>(l[i]) >>
                                          static_cast<int64_t>(r[i]));
        }
        *sp++ = dest;
        ++ip;
        DISPATCH();
    }
    op_eq: {
        double *r = *--sp;
        double *l = *--sp;
        double *dest = sp_storage + (sp - stack) * n;
#pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            dest[i] = (l[i] == r[i]) ? 1.0 : 0.0;
        }
        *sp++ = dest;
        ++ip;
        DISPATCH();
    }
    op_neq: {
        double *r = *--sp;
        double *l = *--sp;
        double *dest = sp_storage + (sp - stack) * n;
#pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            dest[i] = (l[i] != r[i]) ? 1.0 : 0.0;
        }
        *sp++ = dest;
        ++ip;
        DISPATCH();
    }
    op_lt: {
        double *r = *--sp;
        double *l = *--sp;
        double *dest = sp_storage + (sp - stack) * n;
#pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            dest[i] = (l[i] < r[i]) ? 1.0 : 0.0;
        }
        *sp++ = dest;
        ++ip;
        DISPATCH();
    }
    op_le: {
        double *r = *--sp;
        double *l = *--sp;
        double *dest = sp_storage + (sp - stack) * n;
#pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            dest[i] = (l[i] <= r[i]) ? 1.0 : 0.0;
        }
        *sp++ = dest;
        ++ip;
        DISPATCH();
    }
    op_gt: {
        double *r = *--sp;
        double *l = *--sp;
        double *dest = sp_storage + (sp - stack) * n;
#pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            dest[i] = (l[i] > r[i]) ? 1.0 : 0.0;
        }
        *sp++ = dest;
        ++ip;
        DISPATCH();
    }
    op_ge: {
        double *r = *--sp;
        double *l = *--sp;
        double *dest = sp_storage + (sp - stack) * n;
#pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            dest[i] = (l[i] >= r[i]) ? 1.0 : 0.0;
        }
        *sp++ = dest;
        ++ip;
        DISPATCH();
    }
    op_not: {
        double *l = *--sp;
        double *dest = sp_storage + (sp - stack) * n;
#pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            dest[i] = (l[i] == 0.0) ? 1.0 : 0.0;
        }
        *sp++ = dest;
        ++ip;
        DISPATCH();
    }
    op_tilde: {
        double *l = *--sp;
        double *dest = sp_storage + (sp - stack) * n;
#pragma omp simd
        for (std::size_t i = 0; i < n; ++i) {
            dest[i] = static_cast<double>(~static_cast<int64_t>(l[i]));
        }
        *sp++ = dest;
        ++ip;
        DISPATCH();
    }
    op_jmpz: {
        double *c = *--sp;
        bool all_zero = true;
        for (std::size_t i = 0; i < n; ++i) {
            if (c[i] != 0.0) {
                all_zero = false;
                break;
            }
        }
        if (all_zero)
            ip += ip->arg;
        else
            ++ip;
        DISPATCH();
    }
    op_jmp: {
        ip += ip->arg;
        DISPATCH();
    }
    op_call: {
        Builtin b = static_cast<Builtin>(ip->arg >> 16);
        uint32_t arity = ip->arg & 0xFFFF;
        double *args[8];
        for (uint32_t i = arity; i > 0; --i) {
            args[i - 1] = *--sp;
        }
        double *dest = sp_storage + (sp - stack) * n;
        switch (b) {
        case Builtin::Sin:
            evalpp::vec_sin(args[0], dest, n);
            break;
        case Builtin::Cos:
            evalpp::vec_cos(args[0], dest, n);
            break;
        case Builtin::Tan:
#pragma omp simd
            for (std::size_t i = 0; i < n; ++i) {
                dest[i] = std::tan(args[0][i]);
            }
            break;
        case Builtin::Asin:
#pragma omp simd
            for (std::size_t i = 0; i < n; ++i) {
                dest[i] = std::asin(args[0][i]);
            }
            break;
        case Builtin::Acos:
#pragma omp simd
            for (std::size_t i = 0; i < n; ++i) {
                dest[i] = std::acos(args[0][i]);
            }
            break;
        case Builtin::Atan:
#pragma omp simd
            for (std::size_t i = 0; i < n; ++i) {
                dest[i] = std::atan(args[0][i]);
            }
            break;
        case Builtin::Sqrt:
#pragma omp simd
            for (std::size_t i = 0; i < n; ++i) {
                dest[i] = std::sqrt(args[0][i]);
            }
            break;
        case Builtin::Pow:
            evalpp::vec_pow(args[0], args[1], dest, n);
            break;
        case Builtin::Log:
            evalpp::vec_log(args[0], dest, n);
            break;
        case Builtin::Log10:
#pragma omp simd
            for (std::size_t i = 0; i < n; ++i) {
                dest[i] = std::log10(args[0][i]);
            }
            break;
        case Builtin::Exp:
#pragma omp simd
            for (std::size_t i = 0; i < n; ++i) {
                dest[i] = std::exp(args[0][i]);
            }
            break;
        case Builtin::Abs:
#pragma omp simd
            for (std::size_t i = 0; i < n; ++i) {
                dest[i] = std::abs(args[0][i]);
            }
            break;
        case Builtin::Floor:
#pragma omp simd
            for (std::size_t i = 0; i < n; ++i) {
                dest[i] = std::floor(args[0][i]);
            }
            break;
        case Builtin::Ceil:
#pragma omp simd
            for (std::size_t i = 0; i < n; ++i) {
                dest[i] = std::ceil(args[0][i]);
            }
            break;
        case Builtin::Round:
#pragma omp simd
            for (std::size_t i = 0; i < n; ++i) {
                dest[i] = std::round(args[0][i]);
            }
            break;
        case Builtin::Min:
#pragma omp simd
            for (std::size_t i = 0; i < n; ++i) {
                double m = args[0][i];
                for (uint32_t j = 1; j < arity; ++j) {
                    m = std::min(m, args[j][i]);
                }
                dest[i] = m;
            }
            break;
        case Builtin::Max:
#pragma omp simd
            for (std::size_t i = 0; i < n; ++i) {
                double m = args[0][i];
                for (uint32_t j = 1; j < arity; ++j) {
                    m = std::max(m, args[j][i]);
                }
                dest[i] = m;
            }
            break;
        }
        *sp++ = dest;
        ++ip;
        DISPATCH();
    }
    op_end:
        if (sp == stack) {
            std::fill_n(out, n, 0.0);
        } else {
            std::copy_n(*(sp - 1), n, out);
        }
        return;

#undef DISPATCH

#else
        // Fallback switch-based VM for MSVC.
        while (true) {
            switch (ip->op) {
            case Op::PushC: {
                double val = consts[ip->arg];
                double *dest = sp_storage + (sp - stack) * n;
                std::fill_n(dest, n, val);
                *sp++ = dest;
                ++ip;
                break;
            }
            case Op::LoadVar: {
                *sp++ = const_cast<double *>(vars[ip->arg]);
                ++ip;
                break;
            }
            case Op::Add: {
                double *r = *--sp;
                double *l = *--sp;
                double *dest = sp_storage + (sp - stack) * n;
                for (std::size_t i = 0; i < n; ++i) {
                    dest[i] = l[i] + r[i];
                }
                *sp++ = dest;
                ++ip;
                break;
            }
            case Op::Sub: {
                double *r = *--sp;
                double *l = *--sp;
                double *dest = sp_storage + (sp - stack) * n;
                for (std::size_t i = 0; i < n; ++i) {
                    dest[i] = l[i] - r[i];
                }
                *sp++ = dest;
                ++ip;
                break;
            }
            case Op::Mul: {
                double *r = *--sp;
                double *l = *--sp;
                double *dest = sp_storage + (sp - stack) * n;
                for (std::size_t i = 0; i < n; ++i) {
                    dest[i] = l[i] * r[i];
                }
                *sp++ = dest;
                ++ip;
                break;
            }
            case Op::Div: {
                double *r = *--sp;
                double *l = *--sp;
                for (std::size_t i = 0; i < n; ++i) {
                    if (r[i] == 0.0)
                        throw std::runtime_error("Division by zero");
                }
                double *dest = sp_storage + (sp - stack) * n;
                for (std::size_t i = 0; i < n; ++i) {
                    dest[i] = l[i] / r[i];
                }
                *sp++ = dest;
                ++ip;
                break;
            }
            case Op::Mod: {
                double *r = *--sp;
                double *l = *--sp;
                for (std::size_t i = 0; i < n; ++i) {
                    if (r[i] == 0.0)
                        throw std::runtime_error("Modulo by zero");
                }
                double *dest = sp_storage + (sp - stack) * n;
                for (std::size_t i = 0; i < n; ++i) {
                    dest[i] = std::fmod(l[i], r[i]);
                }
                *sp++ = dest;
                ++ip;
                break;
            }
            case Op::And: {
                double *r = *--sp;
                double *l = *--sp;
                double *dest = sp_storage + (sp - stack) * n;
                for (std::size_t i = 0; i < n; ++i) {
                    dest[i] = static_cast<double>(static_cast<int64_t>(l[i]) &
                                                  static_cast<int64_t>(r[i]));
                }
                *sp++ = dest;
                ++ip;
                break;
            }
            case Op::Or: {
                double *r = *--sp;
                double *l = *--sp;
                double *dest = sp_storage + (sp - stack) * n;
                for (std::size_t i = 0; i < n; ++i) {
                    dest[i] = static_cast<double>(static_cast<int64_t>(l[i]) |
                                                  static_cast<int64_t>(r[i]));
                }
                *sp++ = dest;
                ++ip;
                break;
            }
            case Op::Xor: {
                double *r = *--sp;
                double *l = *--sp;
                double *dest = sp_storage + (sp - stack) * n;
                evalpp::vec_pow(l, r, dest, n);
                *sp++ = dest;
                ++ip;
                break;
            }
            case Op::Shl: {
                double *r = *--sp;
                double *l = *--sp;
                double *dest = sp_storage + (sp - stack) * n;
                for (std::size_t i = 0; i < n; ++i) {
                    dest[i] =
                        static_cast<double>(static_cast<int64_t>(l[i])
                                            << static_cast<int64_t>(r[i]));
                }
                *sp++ = dest;
                ++ip;
                break;
            }
            case Op::Shr: {
                double *r = *--sp;
                double *l = *--sp;
                double *dest = sp_storage + (sp - stack) * n;
                for (std::size_t i = 0; i < n; ++i) {
                    dest[i] = static_cast<double>(static_cast<int64_t>(l[i]) >>
                                                  static_cast<int64_t>(r[i]));
                }
                *sp++ = dest;
                ++ip;
                break;
            }
            case Op::Eq: {
                double *r = *--sp;
                double *l = *--sp;
                double *dest = sp_storage + (sp - stack) * n;
                for (std::size_t i = 0; i < n; ++i) {
                    dest[i] = (l[i] == r[i]) ? 1.0 : 0.0;
                }
                *sp++ = dest;
                ++ip;
                break;
            }
            case Op::Neq: {
                double *r = *--sp;
                double *l = *--sp;
                double *dest = sp_storage + (sp - stack) * n;
                for (std::size_t i = 0; i < n; ++i) {
                    dest[i] = (l[i] != r[i]) ? 1.0 : 0.0;
                }
                *sp++ = dest;
                ++ip;
                break;
            }
            case Op::Lt: {
                double *r = *--sp;
                double *l = *--sp;
                double *dest = sp_storage + (sp - stack) * n;
                for (std::size_t i = 0; i < n; ++i) {
                    dest[i] = (l[i] < r[i]) ? 1.0 : 0.0;
                }
                *sp++ = dest;
                ++ip;
                break;
            }
            case Op::Le: {
                double *r = *--sp;
                double *l = *--sp;
                double *dest = sp_storage + (sp - stack) * n;
                for (std::size_t i = 0; i < n; ++i) {
                    dest[i] = (l[i] <= r[i]) ? 1.0 : 0.0;
                }
                *sp++ = dest;
                ++ip;
                break;
            }
            case Op::Gt: {
                double *r = *--sp;
                double *l = *--sp;
                double *dest = sp_storage + (sp - stack) * n;
                for (std::size_t i = 0; i < n; ++i) {
                    dest[i] = (l[i] > r[i]) ? 1.0 : 0.0;
                }
                *sp++ = dest;
                ++ip;
                break;
            }
            case Op::Ge: {
                double *r = *--sp;
                double *l = *--sp;
                double *dest = sp_storage + (sp - stack) * n;
                for (std::size_t i = 0; i < n; ++i) {
                    dest[i] = (l[i] >= r[i]) ? 1.0 : 0.0;
                }
                *sp++ = dest;
                ++ip;
                break;
            }
            case Op::Not: {
                double *l = *--sp;
                double *dest = sp_storage + (sp - stack) * n;
                for (std::size_t i = 0; i < n; ++i) {
                    dest[i] = (l[i] == 0.0) ? 1.0 : 0.0;
                }
                *sp++ = dest;
                ++ip;
                break;
            }
            case Op::Tilde: {
                double *l = *--sp;
                double *dest = sp_storage + (sp - stack) * n;
                for (std::size_t i = 0; i < n; ++i) {
                    dest[i] = static_cast<double>(~static_cast<int64_t>(l[i]));
                }
                *sp++ = dest;
                ++ip;
                break;
            }
            case Op::JmpZ: {
                double *c = *--sp;
                bool all_zero = true;
                for (std::size_t i = 0; i < n; ++i) {
                    if (c[i] != 0.0) {
                        all_zero = false;
                        break;
                    }
                }
                if (all_zero)
                    ip += ip->arg;
                else
                    ++ip;
                break;
            }
            case Op::Jmp: {
                ip += ip->arg;
                break;
            }
            case Op::Call: {
                Builtin b = static_cast<Builtin>(ip->arg >> 16);
                uint32_t arity = ip->arg & 0xFFFF;
                double *args[8];
                for (uint32_t i = arity; i > 0; --i) {
                    args[i - 1] = *--sp;
                }
                double *dest = sp_storage + (sp - stack) * n;
                switch (b) {
                case Builtin::Sin:
                    evalpp::vec_sin(args[0], dest, n);
                    break;
                case Builtin::Cos:
                    evalpp::vec_cos(args[0], dest, n);
                    break;
                case Builtin::Tan:
                    for (std::size_t i = 0; i < n; ++i) {
                        dest[i] = std::tan(args[0][i]);
                    }
                    break;
                case Builtin::Asin:
                    for (std::size_t i = 0; i < n; ++i) {
                        dest[i] = std::asin(args[0][i]);
                    }
                    break;
                case Builtin::Acos:
                    for (std::size_t i = 0; i < n; ++i) {
                        dest[i] = std::acos(args[0][i]);
                    }
                    break;
                case Builtin::Atan:
                    for (std::size_t i = 0; i < n; ++i) {
                        dest[i] = std::atan(args[0][i]);
                    }
                    break;
                case Builtin::Sqrt:
                    for (std::size_t i = 0; i < n; ++i) {
                        dest[i] = std::sqrt(args[0][i]);
                    }
                    break;
                case Builtin::Pow:
                    evalpp::vec_pow(args[0], args[1], dest, n);
                    break;
                case Builtin::Log:
                    evalpp::vec_log(args[0], dest, n);
                    break;
                case Builtin::Log10:
                    for (std::size_t i = 0; i < n; ++i) {
                        dest[i] = std::log10(args[0][i]);
                    }
                    break;
                case Builtin::Exp:
                    for (std::size_t i = 0; i < n; ++i) {
                        dest[i] = std::exp(args[0][i]);
                    }
                    break;
                case Builtin::Abs:
                    for (std::size_t i = 0; i < n; ++i) {
                        dest[i] = std::abs(args[0][i]);
                    }
                    break;
                case Builtin::Floor:
                    for (std::size_t i = 0; i < n; ++i) {
                        dest[i] = std::floor(args[0][i]);
                    }
                    break;
                case Builtin::Ceil:
                    for (std::size_t i = 0; i < n; ++i) {
                        dest[i] = std::ceil(args[0][i]);
                    }
                    break;
                case Builtin::Round:
                    for (std::size_t i = 0; i < n; ++i) {
                        dest[i] = std::round(args[0][i]);
                    }
                    break;
                case Builtin::Min:
                    for (std::size_t i = 0; i < n; ++i) {
                        double m = args[0][i];
                        for (uint32_t j = 1; j < arity; ++j) {
                            m = std::min(m, args[j][i]);
                        }
                        dest[i] = m;
                    }
                    break;
                case Builtin::Max:
                    for (std::size_t i = 0; i < n; ++i) {
                        double m = args[0][i];
                        for (uint32_t j = 1; j < arity; ++j) {
                            m = std::max(m, args[j][i]);
                        }
                        dest[i] = m;
                    }
                    break;
                }
                *sp++ = dest;
                ++ip;
                break;
            }
            case Op::EndProg:
                if (sp == stack) {
                    std::fill_n(out, n, 0.0);
                } else {
                    std::copy_n(*(sp - 1), n, out);
                }
                return;
            }
        }
#endif
    }

    // -----------------------------------------------------------------------
    // eval(vars)  — low-level fast path (unchanged from original).
    //   The caller manages the variable array; the VM loop accesses
    //   vars[ip->arg] with zero overhead — no map lookups, no string ops.
    // -----------------------------------------------------------------------
    double eval(const double *vars) const {
        double stack[256];
        double *sp = stack;
        const Inst *ip = code.data();

#if defined(__GNUC__) || defined(__clang__)
        static const void *dispatch_table[] = {
            &&op_pushc, &&op_loadvar, &&op_add,  &&op_sub, &&op_mul,  &&op_div,
            &&op_mod,   &&op_and,     &&op_or,   &&op_xor, &&op_shl,  &&op_shr,
            &&op_eq,    &&op_neq,     &&op_lt,   &&op_le,  &&op_gt,   &&op_ge,
            &&op_not,   &&op_tilde,   &&op_jmpz, &&op_jmp, &&op_call, &&op_end};

#define DISPATCH()                                                             \
    do {                                                                       \
        goto *dispatch_table[static_cast<uint32_t>(ip->op)];                   \
    } while (0)

        DISPATCH();

    op_pushc: {
        *sp++ = consts[ip->arg];
        ++ip;
        DISPATCH();
    }
    op_loadvar: {
        *sp++ = vars[ip->arg]; // ← still a plain array index; zero overhead
        ++ip;
        DISPATCH();
    }
    op_add: {
        double r = *--sp;
        *(sp - 1) += r;
        ++ip;
        DISPATCH();
    }
    op_sub: {
        double r = *--sp;
        *(sp - 1) -= r;
        ++ip;
        DISPATCH();
    }
    op_mul: {
        double r = *--sp;
        *(sp - 1) *= r;
        ++ip;
        DISPATCH();
    }
    op_div: {
        double r = *--sp;
        if (r == 0.0)
            throw std::runtime_error("Division by zero");
        *(sp - 1) /= r;
        ++ip;
        DISPATCH();
    }
    op_mod: {
        double r = *--sp;
        if (r == 0.0)
            throw std::runtime_error("Modulo by zero");
        *(sp - 1) = std::fmod(*(sp - 1), r);
        ++ip;
        DISPATCH();
    }
    op_and: {
        uint64_t r = *--sp;
        *(sp - 1) = static_cast<double>(static_cast<int64_t>(*(sp - 1)) &
                                        static_cast<int64_t>(r));
        ++ip;
        DISPATCH();
    }
    op_or: {
        uint64_t r = *--sp;
        *(sp - 1) = static_cast<double>(static_cast<int64_t>(*(sp - 1)) |
                                        static_cast<int64_t>(r));
        ++ip;
        DISPATCH();
    }
    op_xor: {
        double r = *--sp;
        *(sp - 1) = std::pow(*(sp - 1), r);
        ++ip;
        DISPATCH();
    }
    op_shl: {
        uint64_t r = *--sp;
        *(sp - 1) = static_cast<double>(static_cast<int64_t>(*(sp - 1))
                                        << static_cast<int64_t>(r));
        ++ip;
        DISPATCH();
    }
    op_shr: {
        uint64_t r = *--sp;
        *(sp - 1) = static_cast<double>(static_cast<int64_t>(*(sp - 1)) >>
                                        static_cast<int64_t>(r));
        ++ip;
        DISPATCH();
    }
    op_eq: {
        double r = *--sp;
        *(sp - 1) = (*(sp - 1) == r) ? 1.0 : 0.0;
        ++ip;
        DISPATCH();
    }
    op_neq: {
        double r = *--sp;
        *(sp - 1) = (*(sp - 1) != r) ? 1.0 : 0.0;
        ++ip;
        DISPATCH();
    }
    op_lt: {
        double r = *--sp;
        *(sp - 1) = (*(sp - 1) < r) ? 1.0 : 0.0;
        ++ip;
        DISPATCH();
    }
    op_le: {
        double r = *--sp;
        *(sp - 1) = (*(sp - 1) <= r) ? 1.0 : 0.0;
        ++ip;
        DISPATCH();
    }
    op_gt: {
        double r = *--sp;
        *(sp - 1) = (*(sp - 1) > r) ? 1.0 : 0.0;
        ++ip;
        DISPATCH();
    }
    op_ge: {
        double r = *--sp;
        *(sp - 1) = (*(sp - 1) >= r) ? 1.0 : 0.0;
        ++ip;
        DISPATCH();
    }
    op_not: {
        *(sp - 1) = (*(sp - 1) == 0.0) ? 1.0 : 0.0;
        ++ip;
        DISPATCH();
    }
    op_tilde: {
        *(sp - 1) = static_cast<double>(~static_cast<int64_t>(*(sp - 1)));
        ++ip;
        DISPATCH();
    }
    op_jmpz: {
        double c = *--sp;
        if (c == 0.0)
            ip += ip->arg;
        else
            ++ip;
        DISPATCH();
    }
    op_jmp: {
        ip += ip->arg;
        DISPATCH();
    }
    op_call: {
        Builtin b = static_cast<Builtin>(ip->arg >> 16);
        uint32_t c = ip->arg & 0xFFFF;
        double *args = sp - c;
        double res = 0;
        switch (b) {
        case Builtin::Sin:
            res = std::sin(args[0]);
            break;
        case Builtin::Cos:
            res = std::cos(args[0]);
            break;
        case Builtin::Tan:
            res = std::tan(args[0]);
            break;
        case Builtin::Asin:
            res = std::asin(args[0]);
            break;
        case Builtin::Acos:
            res = std::acos(args[0]);
            break;
        case Builtin::Atan:
            res = std::atan(args[0]);
            break;
        case Builtin::Sqrt:
            res = std::sqrt(args[0]);
            break;
        case Builtin::Pow:
            res = std::pow(args[0], args[1]);
            break;
        case Builtin::Log:
            res = std::log(args[0]);
            break;
        case Builtin::Log10:
            res = std::log10(args[0]);
            break;
        case Builtin::Exp:
            res = std::exp(args[0]);
            break;
        case Builtin::Abs:
            res = std::abs(args[0]);
            break;
        case Builtin::Floor:
            res = std::floor(args[0]);
            break;
        case Builtin::Ceil:
            res = std::ceil(args[0]);
            break;
        case Builtin::Round:
            res = std::round(args[0]);
            break;
        case Builtin::Min:
            res = args[0];
            for (uint32_t i = 1; i < c; ++i)
                res = std::min(res, args[i]);
            break;
        case Builtin::Max:
            res = args[0];
            for (uint32_t i = 1; i < c; ++i)
                res = std::max(res, args[i]);
            break;
        }
        sp -= c;
        *sp++ = res;
        ++ip;
        DISPATCH();
    }
    op_end:
        return sp == stack ? 0.0 : *(sp - 1);

#undef DISPATCH

#else
        // Fallback switch-based VM for MSVC (unchanged).
        while (true) {
            switch (ip->op) {
            case Op::PushC:
                *sp++ = consts[ip->arg];
                break;
            case Op::LoadVar:
                *sp++ = vars[ip->arg];
                break;
            case Op::Add: {
                double r = *--sp;
                *(sp - 1) += r;
                break;
            }
            case Op::Sub: {
                double r = *--sp;
                *(sp - 1) -= r;
                break;
            }
            case Op::Mul: {
                double r = *--sp;
                *(sp - 1) *= r;
                break;
            }
            case Op::Div: {
                double r = *--sp;
                if (r == 0.0)
                    throw std::runtime_error("Division by zero");
                *(sp - 1) /= r;
                break;
            }
            case Op::Mod: {
                double r = *--sp;
                if (r == 0.0)
                    throw std::runtime_error("Modulo by zero");
                *(sp - 1) = std::fmod(*(sp - 1), r);
                break;
            }
            case Op::And: {
                uint64_t r = *--sp;
                *(sp - 1) = static_cast<double>(
                    static_cast<int64_t>(*(sp - 1)) & static_cast<int64_t>(r));
                break;
            }
            case Op::Or: {
                uint64_t r = *--sp;
                *(sp - 1) = static_cast<double>(
                    static_cast<int64_t>(*(sp - 1)) | static_cast<int64_t>(r));
                break;
            }
            case Op::Xor: {
                double r = *--sp;
                *(sp - 1) = std::pow(*(sp - 1), r);
                break;
            }
            case Op::Shl: {
                uint64_t r = *--sp;
                *(sp - 1) = static_cast<double>(static_cast<int64_t>(*(sp - 1))
                                                << static_cast<int64_t>(r));
                break;
            }
            case Op::Shr: {
                uint64_t r = *--sp;
                *(sp - 1) = static_cast<double>(
                    static_cast<int64_t>(*(sp - 1)) >> static_cast<int64_t>(r));
                break;
            }
            case Op::Eq: {
                double r = *--sp;
                *(sp - 1) = (*(sp - 1) == r) ? 1.0 : 0.0;
                break;
            }
            case Op::Neq: {
                double r = *--sp;
                *(sp - 1) = (*(sp - 1) != r) ? 1.0 : 0.0;
                break;
            }
            case Op::Lt: {
                double r = *--sp;
                *(sp - 1) = (*(sp - 1) < r) ? 1.0 : 0.0;
                break;
            }
            case Op::Le: {
                double r = *--sp;
                *(sp - 1) = (*(sp - 1) <= r) ? 1.0 : 0.0;
                break;
            }
            case Op::Gt: {
                double r = *--sp;
                *(sp - 1) = (*(sp - 1) > r) ? 1.0 : 0.0;
                break;
            }
            case Op::Ge: {
                double r = *--sp;
                *(sp - 1) = (*(sp - 1) >= r) ? 1.0 : 0.0;
                break;
            }
            case Op::Not: {
                *(sp - 1) = (*(sp - 1) == 0.0) ? 1.0 : 0.0;
                break;
            }
            case Op::Tilde: {
                *(sp - 1) =
                    static_cast<double>(~static_cast<int64_t>(*(sp - 1)));
                break;
            }
            case Op::JmpZ: {
                double c = *--sp;
                if (c == 0.0) {
                    ip += ip->arg;
                    continue;
                }
                break;
            }
            case Op::Jmp: {
                ip += ip->arg;
                continue;
            }
            case Op::Call: {
                Builtin b = static_cast<Builtin>(ip->arg >> 16);
                uint32_t c = ip->arg & 0xFFFF;
                double *args = sp - c;
                double res = 0;
                switch (b) {
                case Builtin::Sin:
                    res = std::sin(args[0]);
                    break;
                case Builtin::Cos:
                    res = std::cos(args[0]);
                    break;
                case Builtin::Tan:
                    res = std::tan(args[0]);
                    break;
                case Builtin::Asin:
                    res = std::asin(args[0]);
                    break;
                case Builtin::Acos:
                    res = std::acos(args[0]);
                    break;
                case Builtin::Atan:
                    res = std::atan(args[0]);
                    break;
                case Builtin::Sqrt:
                    res = std::sqrt(args[0]);
                    break;
                case Builtin::Pow:
                    res = std::pow(args[0], args[1]);
                    break;
                case Builtin::Log:
                    res = std::log(args[0]);
                    break;
                case Builtin::Log10:
                    res = std::log10(args[0]);
                    break;
                case Builtin::Exp:
                    res = std::exp(args[0]);
                    break;
                case Builtin::Abs:
                    res = std::abs(args[0]);
                    break;
                case Builtin::Floor:
                    res = std::floor(args[0]);
                    break;
                case Builtin::Ceil:
                    res = std::ceil(args[0]);
                    break;
                case Builtin::Round:
                    res = std::round(args[0]);
                    break;
                case Builtin::Min:
                    res = args[0];
                    for (uint32_t i = 1; i < c; ++i)
                        res = std::min(res, args[i]);
                    break;
                case Builtin::Max:
                    res = args[0];
                    for (uint32_t i = 1; i < c; ++i)
                        res = std::max(res, args[i]);
                    break;
                }
                sp -= c;
                *sp++ = res;
                break;
            }
            case Op::EndProg:
                return sp == stack ? 0.0 : *(sp - 1);
            }
            ++ip;
        }
#endif
    }
};

struct ASTNode {
    enum class Type { Const, Var, Unary, Binary, Ternary, Call } type;
    double val;
    TokenType op;
    uint32_t a, b, c;
    Builtin builtin;
    uint8_t arg_count;
    uint32_t args[8];
};

class Compiler {
  public:
    explicit Compiler(std::vector<Token> tk,
                      const std::vector<std::string_view> &vars)
        : tk_(std::move(tk)), var_names_(vars) {}

    Program compile() {
        uint32_t root = parseTernary();
        if (tk_[pos_].type != TokenType::End)
            throw std::runtime_error("Parse error");

        Program p;
        for (auto v : var_names_)
            p.var_names.emplace_back(v);
        emit(p, root);
        p.code.push_back({Op::EndProg, 0});
        return p;
    }

  private:
    std::vector<Token> tk_;
    std::size_t pos_{0};
    std::vector<ASTNode> ast_;
    std::vector<std::string_view> var_names_;

    bool match(TokenType t) {
        if (tk_[pos_].type == t) {
            ++pos_;
            return true;
        }
        return false;
    }
    void expect(TokenType t) {
        if (!match(t))
            throw std::runtime_error("Parse error");
    }

    uint32_t makeNode(ASTNode n) {
        ast_.push_back(std::move(n));
        return ast_.size() - 1;
    }

    static bool iequals(std::string_view a, std::string_view b) {
        if (a.size() != b.size())
            return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::tolower(static_cast<unsigned char>(a[i])) !=
                std::tolower(static_cast<unsigned char>(b[i])))
                return false;
        }
        return true;
    }

    uint32_t parseTernary() {
        uint32_t cond = parseOr();
        if (match(TokenType::Question)) {
            uint32_t t = parseTernary();
            expect(TokenType::Colon);
            uint32_t f = parseTernary();
            if (ast_[cond].type == ASTNode::Type::Const)
                return ast_[cond].val != 0.0 ? t : f;
            return makeNode({ASTNode::Type::Ternary,
                             0,
                             TokenType::Question,
                             cond,
                             t,
                             f,
                             Builtin::Sin,
                             0,
                             {}});
        }
        return cond;
    }

    uint32_t parseOr() {
        uint32_t n = parseAnd();
        while (match(TokenType::OrOr)) {
            uint32_t r = parseAnd();
            if (ast_[n].type == ASTNode::Type::Const &&
                ast_[r].type == ASTNode::Type::Const) {
                ast_[n].val =
                    (ast_[n].val != 0.0 || ast_[r].val != 0.0) ? 1.0 : 0.0;
                continue;
            }
            n = makeNode({ASTNode::Type::Binary,
                          0,
                          TokenType::OrOr,
                          n,
                          r,
                          0,
                          Builtin::Sin,
                          0,
                          {}});
        }
        return n;
    }

    uint32_t parseAnd() {
        uint32_t n = parseBitOr();
        while (match(TokenType::AndAnd)) {
            uint32_t r = parseBitOr();
            if (ast_[n].type == ASTNode::Type::Const &&
                ast_[r].type == ASTNode::Type::Const) {
                ast_[n].val =
                    (ast_[n].val != 0.0 && ast_[r].val != 0.0) ? 1.0 : 0.0;
                continue;
            }
            n = makeNode({ASTNode::Type::Binary,
                          0,
                          TokenType::AndAnd,
                          n,
                          r,
                          0,
                          Builtin::Sin,
                          0,
                          {}});
        }
        return n;
    }

    uint32_t parseBitOr() {
        uint32_t n = parseBitXor();
        while (match(TokenType::Pipe)) {
            uint32_t r = parseBitXor();
            if (ast_[n].type == ASTNode::Type::Const &&
                ast_[r].type == ASTNode::Type::Const) {
                ast_[n].val =
                    static_cast<double>(static_cast<int64_t>(ast_[n].val) |
                                        static_cast<int64_t>(ast_[r].val));
                continue;
            }
            n = makeNode({ASTNode::Type::Binary,
                          0,
                          TokenType::Pipe,
                          n,
                          r,
                          0,
                          Builtin::Sin,
                          0,
                          {}});
        }
        return n;
    }

    uint32_t parseBitXor() { return parseBitAnd(); }

    uint32_t parseBitAnd() {
        uint32_t n = parseEq();
        while (match(TokenType::Amp)) {
            uint32_t r = parseEq();
            if (ast_[n].type == ASTNode::Type::Const &&
                ast_[r].type == ASTNode::Type::Const) {
                ast_[n].val =
                    static_cast<double>(static_cast<int64_t>(ast_[n].val) &
                                        static_cast<int64_t>(ast_[r].val));
                continue;
            }
            n = makeNode({ASTNode::Type::Binary,
                          0,
                          TokenType::Amp,
                          n,
                          r,
                          0,
                          Builtin::Sin,
                          0,
                          {}});
        }
        return n;
    }

    uint32_t parseEq() {
        uint32_t n = parseRel();
        while (true) {
            if (match(TokenType::EqualEqual)) {
                uint32_t r = parseRel();
                if (ast_[n].type == ASTNode::Type::Const &&
                    ast_[r].type == ASTNode::Type::Const) {
                    ast_[n].val = (ast_[n].val == ast_[r].val) ? 1.0 : 0.0;
                } else
                    n = makeNode({ASTNode::Type::Binary,
                                  0,
                                  TokenType::EqualEqual,
                                  n,
                                  r,
                                  0,
                                  Builtin::Sin,
                                  0,
                                  {}});
            } else if (match(TokenType::NotEqual)) {
                uint32_t r = parseRel();
                if (ast_[n].type == ASTNode::Type::Const &&
                    ast_[r].type == ASTNode::Type::Const) {
                    ast_[n].val = (ast_[n].val != ast_[r].val) ? 1.0 : 0.0;
                } else
                    n = makeNode({ASTNode::Type::Binary,
                                  0,
                                  TokenType::NotEqual,
                                  n,
                                  r,
                                  0,
                                  Builtin::Sin,
                                  0,
                                  {}});
            } else
                break;
        }
        return n;
    }

    uint32_t parseRel() {
        uint32_t n = parseShift();
        while (true) {
            if (match(TokenType::Less)) {
                uint32_t r = parseShift();
                if (ast_[n].type == ASTNode::Type::Const &&
                    ast_[r].type == ASTNode::Type::Const) {
                    ast_[n].val = (ast_[n].val < ast_[r].val) ? 1.0 : 0.0;
                } else
                    n = makeNode({ASTNode::Type::Binary,
                                  0,
                                  TokenType::Less,
                                  n,
                                  r,
                                  0,
                                  Builtin::Sin,
                                  0,
                                  {}});
            } else if (match(TokenType::LessEqual)) {
                uint32_t r = parseShift();
                if (ast_[n].type == ASTNode::Type::Const &&
                    ast_[r].type == ASTNode::Type::Const) {
                    ast_[n].val = (ast_[n].val <= ast_[r].val) ? 1.0 : 0.0;
                } else
                    n = makeNode({ASTNode::Type::Binary,
                                  0,
                                  TokenType::LessEqual,
                                  n,
                                  r,
                                  0,
                                  Builtin::Sin,
                                  0,
                                  {}});
            } else if (match(TokenType::Greater)) {
                uint32_t r = parseShift();
                if (ast_[n].type == ASTNode::Type::Const &&
                    ast_[r].type == ASTNode::Type::Const) {
                    ast_[n].val = (ast_[n].val > ast_[r].val) ? 1.0 : 0.0;
                } else
                    n = makeNode({ASTNode::Type::Binary,
                                  0,
                                  TokenType::Greater,
                                  n,
                                  r,
                                  0,
                                  Builtin::Sin,
                                  0,
                                  {}});
            } else if (match(TokenType::GreaterEqual)) {
                uint32_t r = parseShift();
                if (ast_[n].type == ASTNode::Type::Const &&
                    ast_[r].type == ASTNode::Type::Const) {
                    ast_[n].val = (ast_[n].val >= ast_[r].val) ? 1.0 : 0.0;
                } else
                    n = makeNode({ASTNode::Type::Binary,
                                  0,
                                  TokenType::GreaterEqual,
                                  n,
                                  r,
                                  0,
                                  Builtin::Sin,
                                  0,
                                  {}});
            } else
                break;
        }
        return n;
    }

    uint32_t parseShift() {
        uint32_t n = parseAdd();
        while (true) {
            if (match(TokenType::ShiftLeft)) {
                uint32_t r = parseAdd();
                if (ast_[n].type == ASTNode::Type::Const &&
                    ast_[r].type == ASTNode::Type::Const) {
                    ast_[n].val = static_cast<double>(
                        static_cast<int64_t>(ast_[n].val)
                        << static_cast<int64_t>(ast_[r].val));
                } else
                    n = makeNode({ASTNode::Type::Binary,
                                  0,
                                  TokenType::ShiftLeft,
                                  n,
                                  r,
                                  0,
                                  Builtin::Sin,
                                  0,
                                  {}});
            } else if (match(TokenType::ShiftRight)) {
                uint32_t r = parseAdd();
                if (ast_[n].type == ASTNode::Type::Const &&
                    ast_[r].type == ASTNode::Type::Const) {
                    ast_[n].val =
                        static_cast<double>(static_cast<int64_t>(ast_[n].val) >>
                                            static_cast<int64_t>(ast_[r].val));
                } else
                    n = makeNode({ASTNode::Type::Binary,
                                  0,
                                  TokenType::ShiftRight,
                                  n,
                                  r,
                                  0,
                                  Builtin::Sin,
                                  0,
                                  {}});
            } else
                break;
        }
        return n;
    }

    uint32_t parseAdd() {
        uint32_t n = parseMul();
        while (true) {
            if (match(TokenType::Plus)) {
                uint32_t r = parseMul();
                if (ast_[n].type == ASTNode::Type::Const &&
                    ast_[r].type == ASTNode::Type::Const) {
                    ast_[n].val += ast_[r].val;
                } else
                    n = makeNode({ASTNode::Type::Binary,
                                  0,
                                  TokenType::Plus,
                                  n,
                                  r,
                                  0,
                                  Builtin::Sin,
                                  0,
                                  {}});
            } else if (match(TokenType::Minus)) {
                uint32_t r = parseMul();
                if (ast_[n].type == ASTNode::Type::Const &&
                    ast_[r].type == ASTNode::Type::Const) {
                    ast_[n].val -= ast_[r].val;
                } else
                    n = makeNode({ASTNode::Type::Binary,
                                  0,
                                  TokenType::Minus,
                                  n,
                                  r,
                                  0,
                                  Builtin::Sin,
                                  0,
                                  {}});
            } else
                break;
        }
        return n;
    }

    uint32_t parsePow() {
        uint32_t n = parseUnary();
        while (true) {
            if (match(TokenType::Caret)) {
                uint32_t r = parseUnary();
                if (ast_[n].type == ASTNode::Type::Const &&
                    ast_[r].type == ASTNode::Type::Const) {
                    ast_[n].val = std::pow(ast_[n].val, ast_[r].val);
                } else {
                    n = makeNode({ASTNode::Type::Binary,
                                  0,
                                  TokenType::Caret,
                                  n,
                                  r,
                                  0,
                                  Builtin::Sin,
                                  0,
                                  {}});
                }
            } else {
                break;
            }
        }
        return n;
    }

    uint32_t parseMul() {
        uint32_t n = parsePow();
        while (true) {
            if (match(TokenType::Star)) {
                uint32_t r = parsePow();
                if (ast_[n].type == ASTNode::Type::Const &&
                    ast_[r].type == ASTNode::Type::Const) {
                    ast_[n].val *= ast_[r].val;
                } else
                    n = makeNode({ASTNode::Type::Binary,
                                  0,
                                  TokenType::Star,
                                  n,
                                  r,
                                  0,
                                  Builtin::Sin,
                                  0,
                                  {}});
            } else if (match(TokenType::Slash)) {
                uint32_t r = parsePow();
                if (ast_[n].type == ASTNode::Type::Const &&
                    ast_[r].type == ASTNode::Type::Const) {
                    if (ast_[r].val == 0.0)
                        throw std::runtime_error("Division by zero");
                    ast_[n].val /= ast_[r].val;
                } else
                    n = makeNode({ASTNode::Type::Binary,
                                  0,
                                  TokenType::Slash,
                                  n,
                                  r,
                                  0,
                                  Builtin::Sin,
                                  0,
                                  {}});
            } else if (match(TokenType::Percent)) {
                uint32_t r = parsePow();
                if (ast_[n].type == ASTNode::Type::Const &&
                    ast_[r].type == ASTNode::Type::Const) {
                    if (ast_[r].val == 0.0)
                        throw std::runtime_error("Modulo by zero");
                    ast_[n].val = std::fmod(ast_[n].val, ast_[r].val);
                } else
                    n = makeNode({ASTNode::Type::Binary,
                                  0,
                                  TokenType::Percent,
                                  n,
                                  r,
                                  0,
                                  Builtin::Sin,
                                  0,
                                  {}});
            } else
                break;
        }
        return n;
    }

    uint32_t parseUnary() {
        if (match(TokenType::Plus)) {
            uint32_t n = parseUnary();
            if (ast_[n].type == ASTNode::Type::Const)
                return n;
            return makeNode({ASTNode::Type::Unary,
                             0,
                             TokenType::Plus,
                             n,
                             0,
                             0,
                             Builtin::Sin,
                             0,
                             {}});
        }
        if (match(TokenType::Minus)) {
            uint32_t n = parseUnary();
            if (ast_[n].type == ASTNode::Type::Const) {
                ast_[n].val = -ast_[n].val;
                return n;
            }
            return makeNode({ASTNode::Type::Unary,
                             0,
                             TokenType::Minus,
                             n,
                             0,
                             0,
                             Builtin::Sin,
                             0,
                             {}});
        }
        if (match(TokenType::Not)) {
            uint32_t n = parseUnary();
            if (ast_[n].type == ASTNode::Type::Const) {
                ast_[n].val = ast_[n].val == 0.0 ? 1.0 : 0.0;
                return n;
            }
            return makeNode({ASTNode::Type::Unary,
                             0,
                             TokenType::Not,
                             n,
                             0,
                             0,
                             Builtin::Sin,
                             0,
                             {}});
        }
        if (match(TokenType::Tilde)) {
            uint32_t n = parseUnary();
            if (ast_[n].type == ASTNode::Type::Const) {
                ast_[n].val =
                    static_cast<double>(~static_cast<int64_t>(ast_[n].val));
                return n;
            }
            return makeNode({ASTNode::Type::Unary,
                             0,
                             TokenType::Tilde,
                             n,
                             0,
                             0,
                             Builtin::Sin,
                             0,
                             {}});
        }
        return parsePrimary();
    }

    uint32_t parsePrimary() {
        Token t = tk_[pos_];
        if (match(TokenType::Number))
            return makeNode({ASTNode::Type::Const,
                             t.number,
                             TokenType::End,
                             0,
                             0,
                             0,
                             Builtin::Sin,
                             0,
                             {}});
        if (match(TokenType::Identifier)) {
            if (match(TokenType::LParen)) {
                ASTNode call_node{ASTNode::Type::Call,
                                  0,
                                  TokenType::End,
                                  0,
                                  0,
                                  0,
                                  Builtin::Sin,
                                  0,
                                  {}};

                if (!match(TokenType::RParen)) {
                    while (true) {
                        if (call_node.arg_count >= 8)
                            throw std::runtime_error("Too many arguments");
                        call_node.args[call_node.arg_count++] = parseTernary();
                        if (match(TokenType::Comma))
                            continue;
                        break;
                    }
                    expect(TokenType::RParen);
                }

                struct FuncMap {
                    std::string_view name;
                    Builtin b;
                };
                static constexpr FuncMap funcs[] = {
                    {"sin", Builtin::Sin},     {"cos", Builtin::Cos},
                    {"tan", Builtin::Tan},     {"asin", Builtin::Asin},
                    {"acos", Builtin::Acos},   {"atan", Builtin::Atan},
                    {"sqrt", Builtin::Sqrt},   {"pow", Builtin::Pow},
                    {"log", Builtin::Log},     {"log10", Builtin::Log10},
                    {"exp", Builtin::Exp},     {"abs", Builtin::Abs},
                    {"floor", Builtin::Floor}, {"ceil", Builtin::Ceil},
                    {"round", Builtin::Round}, {"min", Builtin::Min},
                    {"max", Builtin::Max}};

                bool found = false;
                for (const auto &f : funcs) {
                    if (iequals(t.text, f.name)) {
                        call_node.builtin = f.b;
                        found = true;
                        break;
                    }
                }
                if (!found)
                    throw std::runtime_error("Unknown func");

                // Validate function arity
                switch (call_node.builtin) {
                case Builtin::Pow:
                    if (call_node.arg_count != 2)
                        throw std::runtime_error("pow expects 2 arguments");
                    break;
                case Builtin::Min:
                case Builtin::Max:
                    if (call_node.arg_count < 1)
                        throw std::runtime_error(
                            "min/max expect at least 1 argument");
                    break;
                default:
                    if (call_node.arg_count != 1)
                        throw std::runtime_error("Function expects 1 argument");
                    break;
                }

                bool allConst = true;
                for (uint8_t i = 0; i < call_node.arg_count; ++i) {
                    if (ast_[call_node.args[i]].type != ASTNode::Type::Const) {
                        allConst = false;
                        break;
                    }
                }

                if (allConst) {
                    double res = 0;
                    std::array<double, 8> v;
                    for (uint8_t i = 0; i < call_node.arg_count; ++i)
                        v[i] = ast_[call_node.args[i]].val;

                    switch (call_node.builtin) {
                    case Builtin::Sin:
                        res = std::sin(v[0]);
                        break;
                    case Builtin::Cos:
                        res = std::cos(v[0]);
                        break;
                    case Builtin::Tan:
                        res = std::tan(v[0]);
                        break;
                    case Builtin::Asin:
                        res = std::asin(v[0]);
                        break;
                    case Builtin::Acos:
                        res = std::acos(v[0]);
                        break;
                    case Builtin::Atan:
                        res = std::atan(v[0]);
                        break;
                    case Builtin::Sqrt:
                        res = std::sqrt(v[0]);
                        break;
                    case Builtin::Pow:
                        res = std::pow(v[0], v[1]);
                        break;
                    case Builtin::Log:
                        res = std::log(v[0]);
                        break;
                    case Builtin::Log10:
                        res = std::log10(v[0]);
                        break;
                    case Builtin::Exp:
                        res = std::exp(v[0]);
                        break;
                    case Builtin::Abs:
                        res = std::abs(v[0]);
                        break;
                    case Builtin::Floor:
                        res = std::floor(v[0]);
                        break;
                    case Builtin::Ceil:
                        res = std::ceil(v[0]);
                        break;
                    case Builtin::Round:
                        res = std::round(v[0]);
                        break;
                    case Builtin::Min:
                        res = v[0];
                        for (uint8_t i = 1; i < call_node.arg_count; ++i)
                            res = std::min(res, v[i]);
                        break;
                    case Builtin::Max:
                        res = v[0];
                        for (uint8_t i = 1; i < call_node.arg_count; ++i)
                            res = std::max(res, v[i]);
                        break;
                    }
                    return makeNode({ASTNode::Type::Const,
                                     res,
                                     TokenType::End,
                                     0,
                                     0,
                                     0,
                                     Builtin::Sin,
                                     0,
                                     {}});
                }
                return makeNode(std::move(call_node));
            }

            if (iequals(t.text, "pi"))
                return makeNode({ASTNode::Type::Const,
                                 3.14159265358979323846,
                                 TokenType::End,
                                 0,
                                 0,
                                 0,
                                 Builtin::Sin,
                                 0,
                                 {}});
            if (iequals(t.text, "e"))
                return makeNode({ASTNode::Type::Const,
                                 2.71828182845904523536,
                                 TokenType::End,
                                 0,
                                 0,
                                 0,
                                 Builtin::Sin,
                                 0,
                                 {}});

            for (size_t i = 0; i < var_names_.size(); ++i) {
                if (iequals(t.text, var_names_[i])) {
                    return makeNode({ASTNode::Type::Var,
                                     0.0,
                                     TokenType::End,
                                     static_cast<uint32_t>(i),
                                     0,
                                     0,
                                     Builtin::Sin,
                                     0,
                                     {}});
                }
            }
            throw std::runtime_error("Unknown id/variable");
        }
        if (match(TokenType::LParen)) {
            uint32_t n = parseTernary();
            expect(TokenType::RParen);
            return n;
        }
        throw std::runtime_error("Parse error");
    }

    void emit(Program &p, uint32_t n) {
        const auto &node = ast_[n];
        if (node.type == ASTNode::Type::Const) {
            p.consts.push_back(node.val);
            p.code.push_back(
                {Op::PushC, static_cast<uint32_t>(p.consts.size() - 1)});
        } else if (node.type == ASTNode::Type::Var) {
            p.code.push_back({Op::LoadVar, node.a});
        } else if (node.type == ASTNode::Type::Unary) {
            emit(p, node.a);
            if (node.op == TokenType::Minus) {
                p.consts.push_back(-1);
                p.code.push_back(
                    {Op::PushC, static_cast<uint32_t>(p.consts.size() - 1)});
                p.code.push_back({Op::Mul, 0});
            } else if (node.op == TokenType::Not)
                p.code.push_back({Op::Not, 0});
            else if (node.op == TokenType::Tilde)
                p.code.push_back({Op::Tilde, 0});
        } else if (node.type == ASTNode::Type::Binary) {
            if (node.op == TokenType::AndAnd) {
                emit(p, node.a);
                uint32_t jmp1 = p.code.size();
                p.code.push_back({Op::JmpZ, 0});
                emit(p, node.b);
                uint32_t jmp2 = p.code.size();
                p.code.push_back({Op::JmpZ, 0});
                p.consts.push_back(1);
                p.code.push_back(
                    {Op::PushC, static_cast<uint32_t>(p.consts.size() - 1)});
                uint32_t jmp3 = p.code.size();
                p.code.push_back({Op::Jmp, 0});
                p.code[jmp1].arg = p.code.size() - jmp1;
                p.code[jmp2].arg = p.code.size() - jmp2;
                p.consts.push_back(0);
                p.code.push_back(
                    {Op::PushC, static_cast<uint32_t>(p.consts.size() - 1)});
                p.code[jmp3].arg = p.code.size() - jmp3;
            } else if (node.op == TokenType::OrOr) {
                emit(p, node.a);
                uint32_t jmp1 = p.code.size();
                p.code.push_back({Op::JmpZ, 0});
                p.consts.push_back(1);
                p.code.push_back(
                    {Op::PushC, static_cast<uint32_t>(p.consts.size() - 1)});
                uint32_t jmp2 = p.code.size();
                p.code.push_back({Op::Jmp, 0});
                p.code[jmp1].arg = p.code.size() - jmp1;
                emit(p, node.b);
                uint32_t jmp3 = p.code.size();
                p.code.push_back({Op::JmpZ, 0});
                p.consts.push_back(1);
                p.code.push_back(
                    {Op::PushC, static_cast<uint32_t>(p.consts.size() - 1)});
                uint32_t jmp4 = p.code.size();
                p.code.push_back({Op::Jmp, 0});
                p.code[jmp3].arg = p.code.size() - jmp3;
                p.consts.push_back(0);
                p.code.push_back(
                    {Op::PushC, static_cast<uint32_t>(p.consts.size() - 1)});
                p.code[jmp2].arg = p.code.size() - jmp2;
                p.code[jmp4].arg = p.code.size() - jmp4;
            } else {
                emit(p, node.a);
                emit(p, node.b);
                switch (node.op) {
                case TokenType::Plus:
                    p.code.push_back({Op::Add, 0});
                    break;
                case TokenType::Minus:
                    p.code.push_back({Op::Sub, 0});
                    break;
                case TokenType::Star:
                    p.code.push_back({Op::Mul, 0});
                    break;
                case TokenType::Slash:
                    p.code.push_back({Op::Div, 0});
                    break;
                case TokenType::Percent:
                    p.code.push_back({Op::Mod, 0});
                    break;
                case TokenType::Amp:
                    p.code.push_back({Op::And, 0});
                    break;
                case TokenType::Pipe:
                    p.code.push_back({Op::Or, 0});
                    break;
                case TokenType::Caret:
                    p.code.push_back({Op::Xor, 0});
                    break;
                case TokenType::ShiftLeft:
                    p.code.push_back({Op::Shl, 0});
                    break;
                case TokenType::ShiftRight:
                    p.code.push_back({Op::Shr, 0});
                    break;
                case TokenType::EqualEqual:
                    p.code.push_back({Op::Eq, 0});
                    break;
                case TokenType::NotEqual:
                    p.code.push_back({Op::Neq, 0});
                    break;
                case TokenType::Less:
                    p.code.push_back({Op::Lt, 0});
                    break;
                case TokenType::LessEqual:
                    p.code.push_back({Op::Le, 0});
                    break;
                case TokenType::Greater:
                    p.code.push_back({Op::Gt, 0});
                    break;
                case TokenType::GreaterEqual:
                    p.code.push_back({Op::Ge, 0});
                    break;
                default:
                    break;
                }
            }
        } else if (node.type == ASTNode::Type::Ternary) {
            emit(p, node.a);
            uint32_t jmp1 = p.code.size();
            p.code.push_back({Op::JmpZ, 0});
            emit(p, node.b);
            uint32_t jmp2 = p.code.size();
            p.code.push_back({Op::Jmp, 0});
            p.code[jmp1].arg = p.code.size() - jmp1;
            emit(p, node.c);
            p.code[jmp2].arg = p.code.size() - jmp2;
        } else if (node.type == ASTNode::Type::Call) {
            for (uint8_t i = 0; i < node.arg_count; ++i)
                emit(p, node.args[i]);
            p.code.push_back(
                {Op::Call, static_cast<uint32_t>(
                               (static_cast<uint32_t>(node.builtin) << 16) |
                               node.arg_count)});
        }
    }
};

// -------------------------------------------------------------------------
// compile()
//   Parses and compiles the expression, then:
//     1. Builds var_lookup (name → slot) using the case-insensitive map.
//     2. Resizes values[] to one double per variable, zero-initialised.
//
//   Both structures are populated exactly once and reused across every
//   subsequent set() / eval() call — no reallocations, no recompilation.
// -------------------------------------------------------------------------
inline Program compile(std::string_view expression,
                       const std::vector<std::string_view> &var_names = {}) {
    Lexer lexer(expression);
    Compiler compiler(lexer.tokenize(), var_names);
    Program p = compiler.compile();

    // Build the name→slot lookup table. Keys are stored as-is in the
    // compiler's original capitalisation; the case-insensitive hash/equal
    // make every lookup transparent to case.
    p.var_lookup.reserve(p.var_names.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(p.var_names.size()); ++i)
        p.var_lookup.emplace(p.var_names[i], i);

    // Pre-allocate the values array so eval() never re-allocates.
    p.values.assign(p.var_names.size(), 0.0);

    return p;
}

// -------------------------------------------------------------------------
// eval() — one-off convenience helper (no variables).
// -------------------------------------------------------------------------
inline double eval(std::string_view expression) {
    Program prog = compile(expression);
    return prog.eval();
}

} // namespace evalpp
