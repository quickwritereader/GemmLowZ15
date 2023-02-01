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

constexpr int MC = 96 * 8  ;
constexpr int KC = 164 * 4 * 2;
constexpr int NC = 1024;

enum class offset_type {
    none,
    fixed,
    column,
    row,
};

__attribute__ ((noinline)) void addResults(offset_type offsetType, dim_t m, dim_t n, double alpha, double beta,
        int32_t * __restrict__ C, dim_t ldC, int32_t * __restrict__ Ctemp, dim_t ldCtemp, const int32_t * __restrict__ co)   {

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


template <typename TA, typename TB>
inline void LoopKC(bool transA, bool transB, dim_t m, dim_t n, dim_t k,
        const TA *A, dim_t ldA, const TA *ao, const TB *B, dim_t ldB, const TB *bo, int32_t *C,
        dim_t ldC, int16_t *Apacked, int16_t *Bpacked) {
    for (dim_t p = 0; p < k; p += KC) {
        dim_t pb = std::min(KC, k - p);
        dim_t kk = (pb + 1) & -2;
        if(bo){
            int16_t add_val = -(*bo);
            if (transB) {
                pack_K<TB, int16_t, NR, 2, true>(pb, n, &bPtr(0, p), ldB, Bpacked, add_val);
            } else {
                pack_K<TB, int16_t, NR, 2, false>(pb, n, &bPtr(p, 0), ldB, Bpacked, add_val);
            }
        }else{
            if (transB) {
                pack_K<TB, int16_t, NR, 2, true>(pb, n, &bPtr(0, p), ldB, Bpacked);
            } else {
                pack_K<TB, int16_t, NR, 2, false>(pb, n, &bPtr(p, 0), ldB, Bpacked);
            }
        }

        if(ao){
            int16_t add_val = -(*ao);
            if (transA) {
                pack_K<TA, int16_t, MR, 2, false>(pb, m, &aPtr(p, 0), ldA, Apacked, add_val);
            } else {
                pack_K<TA, int16_t, MR, 2, true>(pb, m, &aPtr(0, p), ldA, Apacked, add_val);
            }
        }else{
            if (transA) {
                pack_K<TA, int16_t, MR, 2, false>(pb, m, &aPtr(p, 0), ldA, Apacked);
            } else {
                pack_K<TA, int16_t, MR, 2, true>(pb, m, &aPtr(0, p), ldA, Apacked);
            }
        }

        showMatrix(2, ((pb + 1) & (-2)) * n / 2, Bpacked, 1, "Bpack");

        showMatrix(2, ((pb + 1) & (-2)) * m / 2, Apacked, 1, "Apack");
        
        LoopTwo<NR>(m, n, kk, Apacked, Bpacked, C, ldC);

    }
}

template <typename TA, typename TB>
inline void LoopMC(offset_type offsetType,bool transA, bool transB, dim_t m, dim_t n, dim_t k, float alpha,
        const TA *A, dim_t ldA,  const TA *ao,const TB *B, dim_t ldB, const TB *bo, float beta, int32_t *C,
        dim_t ldC, int16_t *Apacked,int16_t *Bpacked, int32_t *Ctemp, dim_t ldCtemp,const int32_t *co) {
    for (dim_t i = 0; i < m; i += MC) {
        dim_t ib = std::min(MC, m - i);

        for(dim_t u=0; u < ib*n; u++){
            Ctemp[u] = 0;
        }
        LoopKC(transA, transB, ib, n, k, transA?&aPtr(0,i):&aPtr(i,0), ldA, ao, B, ldB, bo, Ctemp, ib,
                Apacked, Bpacked);

                
        addResults(offsetType, ib, n, (double)alpha, (double)beta, &gPtr(i, 0), ldC, Ctemp, ib, co);
    }
}


template <typename TA, typename TB>
inline void LoopNC(offset_type offsetType, bool transA, bool transB, dim_t m, dim_t n, dim_t k,
        float alpha, const TA *A, dim_t ldA, const TA *ao, const TB *B, dim_t ldB,  const TB *bo,float beta, int32_t *C,
        dim_t ldC,const int32_t *co) {
    
    //lets restrict sizes by KC 
    int kC = (k+4)>KC ? KC : ((k+3) & -4 );


    
    auto Bpack = (int16_t *)malloc((kC * NC) * sizeof(int16_t)+16);
    auto Apack = (int16_t *)malloc((MC * kC) * sizeof(int16_t)+16); 
    // unfortunately we have create memory for C as well for the correctness
    // scaling C with beta beforehand is not possible here 
    // and also we have k blocked which makes it safer to allocate for C
    int mC = m+16>MC ? MC : (m + 15) &(-16);
    int nC = n+16>NC ? NC : (n + 15) &(-16);
    auto Ctemp = (int32_t *)malloc((mC * nC) * sizeof(int32_t)+16);

    //align
    auto AP = align_ptr(Apack, 16);
    auto BP = align_ptr(Bpack, 16);
    auto CP = align_ptr(Ctemp, 16);
 
    if (any_null(Apack, Bpack, Ctemp)) {
        free(Apack);
        free(Bpack);
        free(Ctemp);
        return ;
    }
    //we will use (NC->MC->KC) blocking  instead of (NC->KC->MC )to control memory for C temp
    //

    for (dim_t j = 0; j < n; j += NC) {


        dim_t jb = std::min(
                NC, n - j); /* Last loop may not involve a full block */
        LoopMC(offsetType,transA, transB, m, jb, k, alpha, A, ldA, ao,
                transB ? &bPtr(j, 0) : &bPtr(0, j), ldB, bo, beta, &gPtr(0,j), ldC,
                AP,BP, CP, mC, co);
        
    }

    free(Apack);
    free(Bpack);
    free(Ctemp);
}

void gemmX8X8s32(const char *transa, const char *transb, const char *offsetc,
        dim_t M, dim_t N, dim_t K, float alpha, const uint8_t *A, dim_t LDA,
        const uint8_t *ao, const uint8_t *B, dim_t LDB, const uint8_t *bo,
        float beta, int32_t *C, dim_t LDC, const int32_t *co) {

    offset_type offType = offset_type::none;
    if (*offsetc == 'F' || *offsetc == 'f') offType = offset_type::fixed;
    if (*offsetc == 'R' || *offsetc == 'r') offType = offset_type::row;
    if (*offsetc == 'C' || *offsetc == 'c') offType = offset_type::column;
    bool trA = *transa == 't' || *transa == 'T';
    bool trB = *transb == 't' || *transb == 'T';

    LoopNC<uint8_t, uint8_t>(
            offType, trA, trB, M, N, K, alpha, A, LDA, ao, B, LDB, bo, beta, C, LDC, co);
}


void gemmX8X8s32(const char *transa, const char *transb, const char *offsetc,
        dim_t M, dim_t N, dim_t K, float alpha, const int8_t *A, dim_t LDA,
        const int8_t *ao, const uint8_t *B, dim_t LDB, const uint8_t *bo,
        float beta, int32_t *C, dim_t LDC, const int32_t *co){

    offset_type offType = offset_type::none;
    if (*offsetc == 'F' || *offsetc == 'f') offType = offset_type::fixed;
    if (*offsetc == 'R' || *offsetc == 'r') offType = offset_type::row;
    if (*offsetc == 'C' || *offsetc == 'c') offType = offset_type::column;
    bool trA = *transa == 't' || *transa == 'T';
    bool trB = *transb == 't' || *transb == 'T';

    LoopNC<int8_t, uint8_t>(
            offType, trA, trB, M, N, K, alpha, A, LDA, ao, B, LDB, bo, beta, C, LDC, co);
}