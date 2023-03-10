#pragma once
#include <utils.h>

#include <iostream>
template <int G, int KK, bool accessSide, typename T, typename DT = T>
typename std::enable_if<(KK == 4), void>::type pack_G_KK(int k,
        const T *__restrict__ src, int srcLD, DT *__restrict__ dst,
        DT add_val = 0) {
    auto Src = matrix_ptr_t<const T, accessSide> {src, srcLD};

    for (int p = 0; p < k / 4; p++) {
        for (int j = 0; j < G; j++) {
            dst[0] = Src(4 * p, j) + add_val;
            dst[1] = Src(4 * p + 1, j) + add_val;
            dst[2] = Src(4 * p + 2, j) + add_val;
            dst[3] = Src(4 * p + 3, j) + add_val;
            // for (int aa=0;aa<4;aa++)

            dst += 4;
        }
    }
    int k_4 = k & (-4);
    if ((k & 3) == 3) {
        for (int j = 0; j < G; j++) {
            dst[0] = Src(k_4, j) + add_val;
            dst[1] = Src(k_4 + 1, j) + add_val;
            dst[2] = Src(k_4 + 2, j) + add_val;
            dst[3] = add_val;
            dst += 4;
        }
    } else if ((k & 2) == 2) {
        for (int j = 0; j < G; j++) {
            dst[0] = Src(k_4, j) + add_val;
            dst[1] = Src(k_4 + 1, j) + add_val;
            dst[2] = add_val;
            dst[3] = add_val;
            dst += 4;
        }
    } else if ((k & 1) == 1) {
        for (int j = 0; j < G; j++) {
            dst[0] = Src(k_4, j);
            dst[1] = add_val;
            dst[2] = add_val;
            dst[3] = add_val;
            dst += 4;
        }
    }
}

#include <iostream>
template <int G, int KK, bool accessSide, typename T, typename DT>
typename std::enable_if<(KK == 2), void>::type pack_G_KK(int k,
        const T *__restrict__ src, int srcLD, DT *__restrict__ dst,
        DT add_val = 0) {
    auto Src = matrix_ptr_t<const T, accessSide> {src, srcLD};

    for (int p = 0; p < k / 2; p++) {
        for (int j = 0; j < G; j++) {
            dst[0] = (DT)Src(2 * p, j) + add_val;
            dst[1] = (DT)Src(2 * p + 1, j) + add_val;
            dst += 2;
        }
    }
    int k_2 = k & (-2);
    if ((k & 1) == 1) {
        for (int j = 0; j < G; j++) {
            dst[0] = (DT)Src(k_2, j) + add_val;
            dst[1] = (DT)add_val;
            dst += 2;
        }
    }
}

#include <iostream>
template <int G, int KK, bool accessSide, typename T, typename DT>
typename std::enable_if<(KK == 1), void>::type pack_G_KK(
        int k, const T *src, int srcLD, T *dst, DT add_val) {
    auto Src = matrix_ptr_t<const T, accessSide> {src, srcLD};

    for (int p = 0; p < k; p++) {
        for (int j = 0; j < G; j++) {
            *dst++ = (DT)Src(p, j) + add_val;
        }
    }
}

template <int N, int G, int KK, bool accessSide, typename T, typename DT>
typename std::enable_if<(N >= G), void>::type pack_KK_TAILS(const int k,
        const int n, const T *__restrict__ src, int srcLD, DT *__restrict__ dst,
        DT add_val = 0) {
    // no need to unroll
}

template <int N, int G, int KK, bool accessSide, typename T, typename DT>
typename std::enable_if<(N < G), void>::type pack_KK_TAILS(const int k,
        const int n, const T *__restrict__ src, int srcLD, DT *__restrict__ dst,
        DT add_val = 0) {
    if (n & N) {

        // for example n&1, n&2 and et cetera
        pack_G_KK<N, KK, accessSide>(k, src, srcLD, dst, add_val);
        int kk = (k + KK - 1) & (-KK);
        auto Src = matrix_ptr_t<const T, accessSide> {src, srcLD};
        src = Src.ptr(0, N);
        dst += N * kk;
    }

    pack_KK_TAILS<N * 2, G, KK, accessSide>(k, n, src, srcLD, dst, add_val);
}

template <typename T, typename DT, int G, int KK = 4, bool trans>
void __attribute__((noinline))
pack_K(const int k, const int n, const T *__restrict__ src, int srcLD,
        DT *__restrict__ dst, DT add_val = 0) {

    // if k is not divisible by 4 , the rest will be 0-ed
    constexpr bool accessSide = trans ? (!ISLASTINDEX_FAST) : ISLASTINDEX_FAST;
    int kk = (k + KK - 1) & (-KK);
    int nn = n - n % G; // n & (-G); G should be power of 2, so let the
            // compiler to decide
    auto Src = matrix_ptr_t<const T, accessSide> {src, srcLD};
    for (int j = 0; j < nn; j += G) {
        pack_G_KK<G, KK, accessSide>(k, Src.ptr(0, j), srcLD, dst, add_val);
        // last dst will not be accessed
        dst += G * kk;
    }

    // if not padded fully
    // unroll with those conditions:
    // k&1 k&2 k&4 k&8 and so on
    pack_KK_TAILS<1, G, KK, accessSide>(
            k, n - nn, Src.ptr(0, nn), srcLD, dst, add_val);
}
