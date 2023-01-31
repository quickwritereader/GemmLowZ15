#include <stdio.h>
#include <stdlib.h>
#include <utils.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <pack.hpp>
#include <kernel_s16s16s32.hpp>
#include <test_reference.h>
#include <string.h>
#include <stdexcept>

// constexpr int MR=12;
// constexpr int NR=4;

constexpr dim_t MC = 256;
constexpr dim_t KC = 2048;
constexpr dim_t NC = 256;

enum class offset_type {
    none,
    fixed,
    column,
    row,
};

void addResults(offset_type offsetType, dim_t m, dim_t n, double alpha, double beta,
        int32_t *C, dim_t ldC, int32_t *Ctemp, dim_t ldCtemp, const int32_t *co) {

    if (offsetType == offset_type::fixed) {
        if (beta == 0) {
            for (dim_t j = 0; j < n; j++) {
                for (dim_t i = 0; i < m; i++) {
                    double val  =  alpha *  (double)Ctemp[j*ldCtemp + i];
                    gPtr(i, j) = static_cast<int32_t>(nearbyint(
                                         saturate<int32_t, double>(val)))
                            + co[0];
                }
            }
        } else if (beta != 1) {
            for (dim_t j = 0; j < n; j++) {
                for (dim_t i = 0; i < m; i++) {
                    double val = beta * (double)gPtr(i, j) + alpha *  (double)Ctemp[j*ldCtemp + i];
                    gPtr(i, j) = static_cast<int32_t>(nearbyint(
                                         saturate<int32_t, double>(val)))
                            + co[0];
                }
            }
        }
    } else if (offsetType == offset_type::column) {
        if (beta == 0) {
            for (dim_t j = 0; j < n; j++) {
                for (dim_t i = 0; i < m; i++) {
                    double val  =  alpha *  (double)Ctemp[j*ldCtemp + i];
                    gPtr(i, j) = static_cast<int32_t>(nearbyint(
                                         saturate<int32_t, double>(val))) + co[j];
                }
            }
        } else if (beta != 1) {
            for (dim_t j = 0; j < n; j++) {
                for (dim_t i = 0; i < m; i++) {
                    double val = beta * (double)gPtr(i, j) + alpha *  (double)Ctemp[j*ldCtemp + i];
                    gPtr(i, j) = static_cast<int32_t>(nearbyint(
                                         saturate<int32_t, double>(val)))
                            + co[j];
                }
            }
        }

    } else if (offsetType == offset_type::row) {
        if (beta == 0) {
            for (dim_t j = 0; j < n; j++) {
                for (dim_t i = 0; i < m; i++) {
                    double val  =  alpha *  (double)Ctemp[j*ldCtemp + i];
                    gPtr(i, j) = static_cast<int32_t>(nearbyint(
                                         saturate<int32_t, double>(val))) +co[i];
                }
            }
        } else if (beta != 1) {
            for (dim_t j = 0; j < n; j++) {
                for (dim_t i = 0; i < m; i++) {
                    double val = beta * (double)gPtr(i, j) + alpha *  (double)Ctemp[j*ldCtemp + i];
                    gPtr(i, j) = static_cast<int32_t>(nearbyint(
                                         saturate<int32_t, double>(val)))
                            + co[i];
                }
            }
        }
    } else {
        if (beta == 0) {
            for (dim_t j = 0; j < n; j++) {
                for (dim_t i = 0; i < m; i++) {
                    gPtr(i, j) = static_cast<int32_t>(
                            nearbyint(saturate<int32_t, double>( alpha *  (double)Ctemp[j*ldCtemp + i])));
                }
            }
        } else if (beta != 1) {
            for (dim_t j = 0; j < n; j++) {
                for (dim_t i = 0; i < m; i++) {
                    double val = beta * (double)gPtr(i, j) + alpha *  (double)Ctemp[j*ldCtemp + i];
                    gPtr(i, j) = static_cast<int32_t>(
                            nearbyint(saturate<int32_t, double>(val)));
                }
            }
        }
    }
}

template <typename TA>
inline void LoopThree(bool transA, bool transB, dim_t m, dim_t n, dim_t k,
        const TA *A, dim_t ldA, int16_t *Bpacked, int32_t *C,
        dim_t ldC, int16_t *Apacked) {
    constexpr int VLEN = vec_type_t<int32_t>::size();
    for (dim_t i = 0; i < m; i += MC) {
        dim_t ib = std::min(MC, m - i);
        if (transA) {
            pack_K<TA, int16_t, MR, 2, false>(k, ib, &aPtr(0, i), ldA, Apacked);
        } else {
            pack_K<TA, int16_t, MR, 2, true>(k, ib, &aPtr(i, 0), ldA, Apacked);
        }

        showMatrix(2, ((k + 1) & (-2)) * ib / 2, Apacked, 1, "Apack");
        dim_t kk = (k + 1) & -2;
        LoopTwo<NR>(ib, n, kk, Apacked, Bpacked, &gPtr(i, 0), ldC);
    }
}

template <typename TA, typename TB>
inline void LoopFour(bool transA, bool transB, dim_t m, dim_t n, dim_t k,
        const TA *A, dim_t ldA, const TB *B, dim_t ldB, int32_t *C,
        dim_t ldC, int16_t *Apacked, int16_t *Bpacked) {
    constexpr int VLEN = vec_type_t<int32_t>::size();
    for (dim_t p = 0; p < k; p += KC) {
        dim_t pb = std::min(KC, k - p);
        if (transB) {
            pack_K<TB, int16_t, NR, 2, true>(pb, n, &bPtr(0, p), ldB, Bpacked);
        } else {
            pack_K<TB, int16_t, NR, 2, false>(pb, n, &bPtr(p, 0), ldB, Bpacked);
        }

        showMatrix(2, ((pb + 1) & (-2)) * n / 2, Bpacked, 1, "Bpack");

        LoopThree(transA, transB, m, n, pb,
                transA ? &aPtr(p, 0) : &aPtr(0, p), ldA, Bpacked, C, ldC,
                Apacked);
    }
}

template <typename TA, typename TB>
inline void LoopFive(offset_type offsetType, bool transA, bool transB, dim_t m, dim_t n, dim_t k,
        float alpha, const TA *A, dim_t ldA, const TB *B, dim_t ldB,  float beta, int32_t *C,
        dim_t ldC,const int32_t *co) {
    
    //lets restrict sizes by KC,MC,NC
    int kC = k>KC ? KC : ((k+3) & -4 );
    int mC = m>MC ? MC : m;
    int nC = n>NC ? NC : n;
    std::unique_ptr<int16_t[]> Bpack(new int16_t[kC * nC]);
    std::unique_ptr<int16_t[]> Apack(new int16_t[mC * kC]);
    // unfortunately we have create memory for C as well for the correctness
    // scaling C with beta beforehand is not possible here 
    // and also we have k blocked which makes it safer to allocate for C
    std::unique_ptr<int32_t[]> CtempMem(new int32_t[m * nC]);

    int32_t* Ctemp = CtempMem.get();
    if(!Ctemp || !Apack.get() || !Bpack.get()){
        throw std::runtime_error("error");
    }
    for (dim_t j = 0; j < n; j += NC) {

        //set all Ctemp zero
        for(int y = 0; y<m*nC;y++){
            Ctemp[y]=0;
        }
        dim_t jb = std::min(
                NC, n - j); /* Last loop may not involve a full block */
        LoopFour(transA, transB, m, jb, k, A, ldA,
                transB ? &bPtr(j, 0) : &bPtr(0, j), ldB, Ctemp, m/*ldC*/,
                Apack.get(), Bpack.get());
        addResults(offsetType, m, jb, (double)alpha, (double)beta, &gPtr(0,j), ldC, Ctemp, m, co);
        
    }
}

void gemmu8u8s32(const char *transa, const char *transb, const char *offsetc,
        dim_t M, dim_t N, dim_t K, float alpha, const uint8_t *A, dim_t LDA,
        const uint8_t *ao, const uint8_t *B, dim_t LDB, const uint8_t *bo,
        float beta, int32_t *C, dim_t LDC, const int32_t *co) {

    offset_type offType = offset_type::none;
    if (*offsetc == 'F' || *offsetc == 'f') offType = offset_type::fixed;
    if (*offsetc == 'R' || *offsetc == 'r') offType = offset_type::row;
    if (*offsetc == 'C' || *offsetc == 'c') offType = offset_type::column;
    bool trA = *transa == 't' || *transa == 'T';
    bool trB = *transb == 't' || *transb == 'T';

    LoopFive<uint8_t, uint8_t>(
            offType, trA, trB, M, N, K, alpha, A, LDA, B, LDB, beta, C, LDC, co);
}
