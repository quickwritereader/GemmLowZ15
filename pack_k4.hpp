#pragma once
#include <utils.h>

#include <iostream>
template <const int G, bool accessSide, typename T>
void pack_G_PAD_K4(int k, const T *src, int srcLD, T *dst) {
    auto Src = MatrixPtr<const T, accessSide>{src, srcLD};

    // std::cout<<__LINE__<<") "<<G<<"  "<<k<<std::endl;
    for (int p = 0; p < k / 4; p++) {
        for (int j = 0; j < G; j++) {
            dst[0] = Src(4 * p, j);
            dst[1] = Src(4 * p + 1, j);
            dst[2] = Src(4 * p + 2, j);
            dst[3] = Src(4 * p + 3, j);
            // for (int aa=0;aa<4;aa++)
            // std::cout<<dst[aa]<<",";
            // std::cout<<"\n";
            dst += 4;
        }
    }
    int k_4 = k & (-4);
    if ((k & 3) == 3) {
        for (int j = 0; j < G; j++) {
            dst[0] = Src(k_4, j);
            dst[1] = Src(k_4 + 1, j);
            dst[2] = Src(k_4 + 2, j);
            dst[3] = 0;
            dst += 4;
        }
    } else if ((k & 2) == 2) {
        for (int j = 0; j < G; j++) {
            dst[0] = Src(k_4, j);
            dst[1] = Src(k_4 + 1, j);
            dst[2] = 0;
            dst[3] = 0;
            dst += 4;
        }
    } else if ((k & 1) == 1) {
        for (int j = 0; j < G; j++) {
            dst[0] = Src(k_4, j);
            dst[1] = 0;
            dst[2] = 0;
            dst[3] = 0;
            dst += 4;
        }
    }
}

template <const int G, bool accessSide, typename T>
void pack_G_PAD_G_K4(int k, int n, const T *src, int srcLD, T *dst) {
    auto Src = MatrixPtr<const T, accessSide>{src, srcLD};
    int k_4 = k & (-4);

    for (int p = 0; p < k_4; p += 4) {
        for (int j = 0; j < n; j++) {
            dst[0] = Src(p, j);
            dst[1] = Src(p + 1, j);
            dst[2] = Src(p + 2, j);
            dst[3] = Src(p + 3, j);
            dst += 4;
        }
        for (int j = n; j < G; j++) {
            dst[0] = 0;
            dst[1] = 0;
            dst[2] = 0;
            dst[3] = 0;
            dst += 4;
        }
    }
    if ((k & 3) == 3) {
        for (int j = 0; j < n; j++) {
            dst[0] = Src(k_4, j);
            dst[1] = Src(k_4 + 1, j);
            dst[2] = Src(k_4 + 2, j);
            dst[3] = 0;
            // std::cout<<dst[0]
            // <<","<<dst[1]<<","<<dst[2]<<","<<dst[3]<<std::endl;
            dst += 4;
        }
        for (int j = n; j < G; j++) {
            dst[0] = 0;
            dst[1] = 0;
            dst[2] = 0;
            dst[3] = 0;
            dst += 4;
        }
    } else if ((k & 2) == 2) {
        for (int j = 0; j < n; j++) {
            dst[0] = Src(k_4, j);
            dst[1] = Src(k_4 + 1, j);
            dst[2] = 0;
            dst[3] = 0;
            dst += 4;
        }
        for (int j = n; j < G; j++) {
            dst[0] = 0;
            dst[1] = 0;
            dst[2] = 0;
            dst[3] = 0;
            dst += 4;
        }
    } else if ((k & 1) == 1) {
        for (int j = 0; j < n; j++) {
            dst[0] = Src(k_4, j);
            dst[1] = 0;
            dst[2] = 0;
            dst[3] = 0;
            dst += 4;
        }
        for (int j = n; j < G; j++) {
            dst[0] = 0;
            dst[1] = 0;
            dst[2] = 0;
            dst[3] = 0;
            dst += 4;
        }
    }
}

template <typename T, int N, int G, bool accessSide>
typename std::enable_if<(N >= G), void>::type pack_K4_TAILS(const int k,
                                                            const int n,
                                                            const T *src,
                                                            int srcLD, T *dst) {
    // no need to unroll
}

template <typename T, int N, int G, bool accessSide>
typename std::enable_if<(N < G), void>::type pack_K4_TAILS(const int k,
                                                           const int n,
                                                           const T *src,
                                                           int srcLD, T *dst) {
    if (n & N) {
        // std::cout<<__LINE__<<") "<<N<<std::endl;
        // for example n&1, n&2 and et cetera
        pack_G_PAD_K4<N, accessSide>(k, src, srcLD, dst);
        int kk = (k + 3) & (-4);
        auto Src = MatrixPtr<const T, accessSide>{src, srcLD};
        src = Src.ptr(0, N);
        dst += N * kk;
    }

    pack_K4_TAILS<T, N * 2, G, accessSide>(k, n, src, srcLD, dst);
}

template <typename T, int G, bool trans, bool PAD_G = false>
void pack_K4(const int k, const int n, const T *src, int srcLD, T *dst) {
    // std::cout<<"____________PAD_________ "<<k<<", "<<n<<std::endl;
    // if k is not divisible by 4 , the rest will be 0-ed
    constexpr bool accessSide = trans ? (!ISLASTINDEX_FAST) : ISLASTINDEX_FAST;
    int kk = (k + 3) & (-4);
    int nn = n - n % G;  // n & (-G); G should be power of 2, so let the
                         // compiler to decide
    auto Src = MatrixPtr<const T, accessSide>{src, srcLD};
    for (int j = 0; j < nn; j += G) {
        pack_G_PAD_K4<G, accessSide>(k, Src.ptr(0, j), srcLD, dst);
        // last dst will not be accessed
        dst += G * kk;
    }

    if (!PAD_G) {
        // if not padded fully
        // unroll with those conditions:
        // k&1 k&2 k&4 k&8 and so on
        pack_K4_TAILS<T, 1, G, accessSide>(k, n - nn, Src.ptr(0, nn), srcLD,
                                           dst);
    } else {
        if (n > nn) {
            pack_G_PAD_G_K4<G, accessSide>(k, n - nn, Src.ptr(0, nn), srcLD,
                                           dst);
        }
    }
}
