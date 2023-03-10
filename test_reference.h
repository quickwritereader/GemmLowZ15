#pragma once
#include <utils.h>
#include <type_traits>
#include <dnnl_thread.hpp>
#include <iostream>
using data_type_t = int;
namespace data_type {
const data_type_t undef = 0;
const data_type_t f16 = 1;
const data_type_t bf16 = 2;
const data_type_t f32 = 3;
const data_type_t s32 = 4;
const data_type_t s8 = 5;
const data_type_t u8 = 6;
const data_type_t f64 = 7;
} // namespace data_type

template <data_type_t>
struct prec_traits {}; /* ::type -> float */
template <typename>
struct data_traits_t {}; /* ::data_type -> f32 */
template <int>
struct typesize_traits_t {}; /* ::data_type_size -> f32 */

template <>
struct prec_traits<data_type::f32> {
    using type = float;
};
template <>
struct prec_traits<data_type::f64> {
    using type = double;
};
template <>
struct prec_traits<data_type::s32> {
    using type = int32_t;
};
template <>
struct prec_traits<data_type::s8> {
    using type = int8_t;
};
template <>
struct prec_traits<data_type::u8> {
    using type = uint8_t;
};

template <>
struct data_traits_t<float> {
    static constexpr data_type_t data_type = data_type::f32;
};
template <>
struct data_traits_t<int32_t> {
    static constexpr data_type_t data_type = data_type::s32;
};
template <>
struct data_traits_t<int8_t> {
    static constexpr data_type_t data_type = data_type::s8;
};
template <>
struct data_traits_t<uint8_t> {
    static constexpr data_type_t data_type = data_type::u8;
};

template <>
struct typesize_traits_t<4> {
    using type = float;
};
template <>
struct typesize_traits_t<2> {
    using type = int16_t;
};
template <>
struct typesize_traits_t<1> {
    using type = uint8_t;
};

namespace types {
inline size_t data_type_size(data_type_t data_type) {
    using namespace data_type;
    switch ((int)data_type) {

        case f32: return sizeof(prec_traits<f32>::type);
        case f64: return sizeof(prec_traits<f64>::type);
        case s32: return sizeof(prec_traits<s32>::type);
        case s8: return sizeof(prec_traits<s8>::type);
        case u8:
            return sizeof(prec_traits<u8>::type);
            // case data_type::undef:
            // default:
    }
    return (size_t)-1; /* not supposed to be reachable */
}

template <typename T>
inline T max_value(data_type_t data_type) {
    using namespace data_type;
#define CASE(x) \
    case x: \
        return static_cast<T>(std::numeric_limits<prec_traits<x>::type>::max())
    switch (data_type) {
        // CASE(f16);
        // CASE(bf16);
        CASE(s32);
        CASE(s8);
        CASE(u8);
        // case data_type::undef:
        // default: assert(!"unknown data_type");
    }
    return static_cast<T>(0); /* not supposed to be reachable */
#undef CASE
}

// This is a hack to comply with a big comment below.
template <>
inline float max_value(data_type_t data_type) {
    using namespace data_type;
#define CASE(x) \
    case x: \
        return static_cast<float>( \
                std::numeric_limits<prec_traits<x>::type>::max())
    switch (data_type) {

        CASE(s8);
        CASE(u8);
        // INT_MAX is not representable in float. The nearest float to it is
        // INT_MAX + 1 = 2^31 (0x4f000000). Regular conversion instructions such
        // as `cvtps2dq` or `cvtss2si` will convert this number to INT_MIN
        // making the result negative. We on purpose choose the previous float
        // number (0x4effffff) to return leaving the output close to INT_MAX but
        // still positive. In addition, we adjust validation of this approach.
        // The main concern against `real` saturation is performance, which
        // likely to drop (but it was not proved). The only drawback of current
        // approach is saturating on some integer values before it should happen
        // in the reality.
        case s32:
            return 2147483520.f;
            // case data_type::undef:
    }
    return 0.f; /* not supposed to be reachable */
#undef CASE
}
} // namespace types
template <typename data_t, typename acc_t>
inline typename std::enable_if<!std::is_integral<data_t>::value,
        typename std::remove_reference<acc_t>::type>::type
saturate(const acc_t &x) {
    acc_t v = x;
    return v;
}

template <typename data_t, typename acc_t>
inline typename std::enable_if<std::is_integral<data_t>::value,
        typename std::remove_reference<acc_t>::type>::type
saturate(const acc_t &x) {
    acc_t v = x;
    acc_t lbound = (acc_t)std::numeric_limits<data_t>::lowest();
    // Pick up a modified version of max value when do f32 -> s32.
    acc_t ubound = types::max_value<acc_t>(data_traits_t<data_t>::data_type);
    if (v < lbound) v = lbound;
    if (v > ubound) v = ubound;
    return v;
}

template <>
inline uint8_t saturate<int8_t, uint8_t>(const uint8_t &x) {
    return x <= 127u ? x : 127;
}

template <>
inline int8_t saturate<uint8_t, int8_t>(const int8_t &x) {
    return x >= 0 ? x : 0;
}
inline float mxcsr_round(float f) {
    return nearbyintf(f);
}

/** converts @p f to an integer according to the mxcsr register */
inline int mxcsr_cvt(float f) {
    return (int)mxcsr_round(f);
}

template <typename out_t>
inline typename std::enable_if<std::is_integral<out_t>::value,
        typename std::remove_reference<out_t>::type>::type
out_round(float v) {
    return (out_t)mxcsr_cvt(v);
}

template <typename out_t>
inline typename std::enable_if<!std::is_integral<out_t>::value,
        typename std::remove_reference<out_t>::type>::type
out_round(float v) {
    return v;
}

template <typename out_t, typename acc_t = float>
inline out_t saturate_and_round(acc_t f) {
    return out_round<out_t>(saturate<out_t, acc_t>(f));
}

struct test_igemm_params_t {
    char offsetc;
    int8_t _oa = 0;
    int8_t _ob = 0;
    bool nonzero_oc;

    int8_t oa() const { return _oa; }
    int8_t ob() const { return _ob; }
};

struct test_pack_params_t {
    bool pack_a;
    bool pack_b;
};

struct gemm_offset_t {
    int64_t a;
    int64_t b;
    int64_t c;
    int64_t co;
};
struct test_params_t {
    char transA;
    char transB;
    int64_t M;
    int64_t N;
    int64_t K;
    float alpha;
    float beta;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;

    test_igemm_params_t igemm_params;
    test_pack_params_t pack_params;
    bool expect_to_fail;
    //dnnl_status_t expected_status;

    gemm_offset_t off;

    bool tr_a() const { return transA == 'T' || transA == 't'; }
    bool tr_b() const { return transB == 'T' || transB == 't'; }
    int64_t sizeC() const { return M * ldc; }

    bool oc_is_R() const {
        auto c = igemm_params.offsetc;
        return c == 'R' || c == 'r';
    }
    bool oc_is_C() const {
        auto c = igemm_params.offsetc;
        return c == 'C' || c == 'c';
    }
    int64_t size_oc() const { return oc_is_R() ? N : oc_is_C() ? M : 1; }
};

template <typename a_dt, typename b_dt>
struct ref_gemm_t {
    static void call(const test_params_t &p, int64_t M, int64_t N, const a_dt *A,
            const b_dt *B, int32_t *C, const int32_t *oc) {

        const bool tr_a = p.transA && (p.transA == 'T' || p.transA == 't');
        const bool tr_b = p.transB && (p.transB == 'T' || p.transB == 't');
        bool OCisR = (p.igemm_params.offsetc == 'R'
                || p.igemm_params.offsetc == 'r');
        bool OCisC = (p.igemm_params.offsetc == 'C'
                || p.igemm_params.offsetc == 'c');

        auto pa = matrix_ptr_t<const a_dt, ISLASTINDEX_FAST, double> {
                &(A[p.off.a]), p.lda};
        auto pb = matrix_ptr_t<const b_dt, ISLASTINDEX_FAST, double> {
                &(B[p.off.b]), p.ldb};
        auto pc = matrix_ptr_t<int32_t, ISLASTINDEX_FAST> {&C[p.off.c], p.ldc};
        int8_t oa = p.igemm_params.oa();
        int8_t ob = p.igemm_params.ob();
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_IGNORE
      for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
#else
        dnnl::impl::parallel_nd(M, N, [&](int64_t m, int64_t n) {
#endif
            double c_elem = 0;
            for (int64_t k = 0; k < p.K; k++) {
                const double a_elem = (tr_a ? pa(k, m) : pa(m, k)) - oa;
                const double b_elem = (tr_b ? pb(n, k) : pb(k, n)) - ob;
                c_elem += a_elem * b_elem;
            }

            double coffset = OCisR ? oc[n] : OCisC ? oc[m] : oc[0];

            double val = (p.beta == 0.f ? 0. : p.beta * (double)pc(m, n))
                    + p.alpha * c_elem + coffset;
            pc(m, n) = static_cast<int32_t>(
                    nearbyint(saturate<int32_t, double>(val)));
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_IGNORE
        }}
#else
        });
#endif
            }
};
